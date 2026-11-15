# Commandes essentielles du CLI MLX

Options vérifiées sur `mlx-lm` 0.32.2 et `mlx-vlm` 0.7.12 (versions du projet).
Pour tout le reste : `<commande> --help`.

MLX n'est installé que dans ce projet : préfixe chaque commande par `uv run` (ou active
l'environnement avec `source .venv/bin/activate`).

Toutes les commandes prennent `--model` : un repo Hugging Face (`mlx-community/gemma-4-e4b-it-4bit`,
téléchargé au premier appel) ou un chemin local. Sans `--model`, `generate` et `benchmark`
prennent `mlx-community/Llama-3.2-3B-Instruct-4bit` et le **téléchargent**.

## `mlx_lm.generate` — une réponse à un prompt

Le test le plus rapide d'un modèle. Affiche la réponse puis les stats : tokens/s du prompt
(prefill), tokens/s de génération et **pic mémoire** (le vrai chiffre de RAM consommée).

```bash
uv run mlx_lm.generate --model mlx-community/gemma-4-e4b-it-4bit -p "Explique MLX en 2 phrases" -m 200
echo "Résume ce texte : ..." | uv run mlx_lm.generate --model <repo> -p -    # prompt lu sur stdin
```

| Option | Rôle |
|---|---|
| `-p`, `--prompt` | Le message (`-` = stdin) |
| `-m`, `--max-tokens` | Longueur maximale de la réponse (défaut 100) |
| `--system-prompt` | Prompt système |
| `--temp`, `--top-p`, `--top-k`, `--min-p` | Échantillonnage. `--temp 0` = déterministe |
| `--seed` | Graine, pour des résultats reproductibles |
| `--max-kv-size` | Plafonne le KV cache (en tokens) : borne la RAM sur les longs contextes |
| `--kv-bits 8` / `4` (2 à 8) | Quantifie le KV cache : divise sa RAM par ~2 ou ~3,5. Échoue sur les modèles à fenêtre glissante (Gemma 3/4) |
| `--prefill-step-size` | Taille des lots du prefill. Plus petit = pic mémoire plus bas, plus lent |
| `--prompt-cache-file` | Réutilise un long préfixe pré-calculé (voir `mlx_lm.cache_prompt`) |
| `--draft-model` | Décodage spéculatif avec un petit modèle de la même famille |
| `--ignore-chat-template` | Envoie le prompt brut (modèles de base, non instruct) |
| `--image` | Images pour les modèles multimodaux pris en charge par mlx_lm |
| `--verbose False` | N'affiche que la réponse, sans stats |

## `mlx_lm.chat` — conversation interactive dans le terminal

Garde l'historique entre les tours. Tape `q` pour quitter, `r` pour remettre la conversation à zéro, `h` pour l'aide.

```bash
uv run mlx_lm.chat --model mlx-community/gemma-4-e4b-it-4bit --max-tokens 1000
uv run mlx_lm.chat --model <repo> --system-prompt "Tu réponds en français, brièvement."
```

Options utiles : `--max-tokens`, `--temp`, `--system-prompt`, `--max-kv-size`, `--prefill-step-size`.

## `mlx_lm.server` — API HTTP compatible OpenAI

Expose `/v1/chat/completions`, `/v1/completions` et `/v1/models`. N'importe quel client
OpenAI (SDK, LangChain, outils…) s'y branche en changeant la `base_url`. Le serveur charge
le modèle demandé dans le champ `model` de la requête : `/v1/models` liste tout ton cache,
pas seulement le modèle passé au lancement.

```bash
uv run mlx_lm.server --model mlx-community/gemma-4-e4b-it-4bit --port 8080
# ou, avec choix interactif du modèle :
uv run python scripts/serve.py
```

| Option | Rôle |
|---|---|
| `--host` | `127.0.0.1` par défaut. Ne mets `0.0.0.0` que si tu veux l'ouvrir au réseau : aucune authentification |
| `--port` | 8080 par défaut |
| `--max-tokens` | Longueur max par défaut d'une réponse (512) |
| `--temp`, `--top-p`, `--top-k`, `--min-p` | Échantillonnage par défaut (temp 0 par défaut) |
| `--chat-template-args '{"enable_thinking":false}'` | Coupe le mode « réflexion » pour toutes les requêtes |
| `--kv-bits`, `--max-kv-size`… | Comme pour `generate`. `--kv-bits` désactive le batching |
| `--decode-concurrency`, `--prompt-concurrency` | Requêtes traitées en parallèle |
| `--prompt-cache-size`, `--prompt-cache-bytes` | Taille du cache des préfixes déjà vus (conversations multi-tours) |
| `--draft-model` | Décodage spéculatif |
| `--adapter-path` | Charge un adaptateur LoRA |

Par requête, le champ non standard `chat_template_kwargs` (ex. `{"enable_thinking": false}`)
est transmis au chat template : voir `examples/03_openai_client.py`.

## `mlx_lm.manage` — voir et supprimer les modèles du cache

```bash
uv run mlx_lm.manage --scan                    # modèles dont le repo contient "mlx"
uv run mlx_lm.manage --scan --pattern MLX      # le filtre tient compte de la casse !
uv run mlx_lm.manage --delete --pattern LFM2-24B-A2B-MLX-8bit   # demande confirmation
```

Piège : `--scan` sans pattern rate les repos nommés `...-MLX-4bit`. Pour une vue complète,
utilise `scripts/list_models.py` (voir [modeles-installes.md](modeles-installes.md)).

## `mlx_lm.convert` — convertir et quantifier un modèle Hugging Face

Prend un modèle au format Hugging Face (PyTorch/safetensors) et produit un dossier MLX,
quantifié ou non. Utile quand aucune version MLX n'existe déjà sur `mlx-community`.

```bash
uv run mlx_lm.convert --hf-path LiquidAI/LFM2-2.6B --mlx-path ./models/lfm2-2.6b-4bit -q
uv run mlx_lm.convert --hf-path <repo> --mlx-path ./models/x-8bit -q --q-bits 8
uv run mlx_lm.convert --hf-path <repo> --mlx-path ./models/x-mxfp4 -q --q-mode mxfp4
uv run mlx_lm.convert --hf-path <repo> --mlx-path ./models/x-mixed -q --quant-predicate mixed_3_6
```

| Option | Rôle |
|---|---|
| `-q` | Quantifier (sinon : conversion en float16/bfloat16) |
| `--q-bits` | Bits par poids (4 par défaut) |
| `--q-group-size` | Taille des groupes de quantification (64 par défaut) |
| `--q-mode` | `affine` (défaut), `mxfp4`, `nvfp4`, `mxfp8` |
| `--quant-predicate` | Recettes mixtes : `mixed_3_6`, `mixed_4_6`… (couches sensibles en plus de bits) |
| `--dtype` | Type des paramètres non quantifiés |
| `-d` | Déquantifier un modèle |
| `--upload-repo` | Publier le résultat sur Hugging Face |

La conversion charge le modèle complet en float16 : il faut environ 2 octets par paramètre de RAM
(un 8B demande ~16 Go). Voir [quantification.md](quantification.md).

## `mlx_lm.benchmark` — mesurer la vitesse

Mesure prefill et génération sur des tailles fixées, plusieurs essais : le bon outil pour
comparer deux modèles ou deux quantifications.

```bash
uv run mlx_lm.benchmark --model mlx-community/gemma-4-e4b-it-4bit -p 2048 -g 256 -n 3
```

| Option | Rôle |
|---|---|
| `-p`, `--prompt-tokens` | Longueur du prompt simulé |
| `-g`, `--generation-tokens` | Tokens à générer |
| `-n`, `--num-trials` | Nombre d'essais |
| `-b`, `--batch-size` | Requêtes simultanées |

## Modèles vision : `mlx_vlm`

Pour les modèles qui lisent des images (type `VLM` dans `scripts/list_models.py`).

```bash
uv run mlx_vlm.generate --model mlx-community/nanoLLaVA-1.5-8bit --image photo.jpg --prompt "Décris l'image" --max-tokens 200
uv run mlx_vlm.chat --model mlx-community/nanoLLaVA-1.5-8bit
uv run mlx_vlm.server --model <repo> --host 127.0.0.1 --port 8080
uv run python scripts/serve.py <repo> --vlm
```

`mlx_vlm.server` écoute sur `0.0.0.0` par défaut, donc il est visible du réseau local : passe
`--host 127.0.0.1` (c'est ce que fait `scripts/serve.py`). Options en plus : `--audio`, `--system`,
`--temperature`, `--enable-thinking`, `--kv-bits`.

Un modèle multimodal comme Gemma 4 marche aussi en texte seul avec `mlx_lm`, qui ne charge alors
que la partie texte.

## Les autres commandes, en une ligne

| Commande | À quoi elle sert |
|---|---|
| `mlx_lm.cache_prompt` | Pré-calcule le KV cache d'un long préfixe (document, prompt système) dans un fichier |
| `mlx_lm.lora` / `mlx_lm.fuse` | Fine-tuning LoRA, puis fusion de l'adaptateur dans le modèle |
| `mlx_lm.awq`, `mlx_lm.gptq`, `mlx_lm.dwq`, `mlx_lm.dynamic_quant` | Quantifications qui préservent mieux la qualité, avec des données de calibration |
| `mlx_lm.evaluate` / `mlx_lm.perplexity` | Évaluer la qualité (benchmarks lm-eval, perplexité) |
| `mlx_lm.upload` / `mlx_lm.share` | Publier un modèle sur Hugging Face / le partager entre Macs |
| `mlx_vlm.convert` / `mlx_vlm.chat_ui` | Convertir un VLM / interface web de chat |

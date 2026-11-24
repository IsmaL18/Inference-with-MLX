# inference-with-mlx

Bac à sable et base de connaissance pour faire tourner des LLM en local sur Apple Silicon avec
[MLX](https://github.com/ml-explore/mlx) (`mlx-lm` pour le texte, `mlx-vlm` pour la vision).

## Prérequis

- Un Mac Apple Silicon (M1 ou plus récent).
- [uv](https://docs.astral.sh/uv/) : `curl -LsSf https://astral.sh/uv/install.sh | sh`

## Installation

MLX est une dépendance du projet : rien n'est installé globalement.

```bash
uv sync                            # crée .venv avec mlx-lm, mlx-vlm, openai
uv lock --upgrade && uv sync       # plus tard, pour mettre MLX à jour
```

Lance ensuite chaque commande avec `uv run` (par exemple `uv run mlx_lm.chat …`), ou active
l'environnement une fois par terminal avec `source .venv/bin/activate`.

## Démarrage

```bash
uv run python scripts/list_models.py                            # modèles installés, et lesquels tiennent en RAM
uv run mlx_lm.chat --model mlx-community/gemma-4-e4b-it-4bit    # discuter avec un modèle (téléchargé au 1er lancement)
```

## Commandes essentielles

| Commande | À quoi elle sert |
|---|---|
| `mlx_lm.generate --model <repo> -p "…"` | Une réponse à un prompt, avec tokens/s et pic mémoire |
| `mlx_lm.chat --model <repo>` | Conversation interactive dans le terminal |
| `mlx_lm.server --model <repo>` | API HTTP compatible OpenAI sur `127.0.0.1:8080` |
| `mlx_lm.manage --scan --pattern MLX` | Lister les modèles du cache (`--delete` pour supprimer) |
| `mlx_lm.convert --hf-path <repo> -q` | Convertir un modèle Hugging Face en MLX quantifié |
| `mlx_lm.benchmark --model <repo>` | Mesurer la vitesse (prefill, génération, mémoire) |
| `mlx_vlm.generate --model <repo> --image x.jpg --prompt "…"` | Interroger un modèle vision sur une image |

Le détail, les options et les exemples sont dans [docs/commandes-cli.md](docs/commandes-cli.md).

## RAM : l'essentiel

- **RAM nécessaire ≈ poids du modèle en Go + KV cache (grandit avec le contexte) + ~1 Go.**
- macOS ne donne au GPU qu'environ **70 à 75 % de la RAM**. C'est ce budget qui compte, pas la RAM totale.
- Poids selon la quantification : **4bit ≈ 0,56 Go par milliard de paramètres**, 8bit ≈ 1,06 Go, bf16 = 2 Go.
- Garde une marge : au-delà du budget GPU, macOS swappe et la génération s'effondre.
- Un MoE (`35B-A3B`) doit tenir **en entier** en RAM, mais génère à la vitesse de ses paramètres actifs.

Calculs, tableaux par quantification et réglage de la limite GPU : [docs/ram.md](docs/ram.md).

## Quels modèles sont installés ?

```bash
uv run python scripts/list_models.py         # modèles MLX : type, quantification, poids, RAM estimée, verdict
uv run python scripts/list_models.py --all   # tout le cache Hugging Face
```

Les modèles sont stockés dans `~/.cache/huggingface/hub`. Commandes alternatives et
suppression : [docs/modeles-installes.md](docs/modeles-installes.md).

## Servir un modèle en local

```bash
uv run python scripts/serve.py                     # propose la liste des modèles installés
uv run python scripts/serve.py mlx-community/gemma-4-e4b-it-4bit --port 8080
uv run python scripts/serve.py <repo> -- --max-tokens 2048   # après --, options de mlx_lm.server
uv run python scripts/serve.py <repo-vision> --vlm           # mlx_vlm.server
```

Le script lance le serveur officiel de MLX, écoute en local uniquement (`127.0.0.1`) et passe
en mode hors ligne si le modèle est déjà en cache. Ensuite, n'importe quel client OpenAI fonctionne
avec `base_url="http://127.0.0.1:8080/v1"`.

## Exemples Python

| Script | Montre |
|---|---|
| `examples/01_generate.py` | `load` + `generate`, chat template, échantillonnage |
| `examples/02_stream.py` | `stream_generate` : affichage token par token et métriques |
| `examples/03_openai_client.py` | Appeler le serveur local avec le SDK OpenAI, en streaming |
| `examples/04_prompt_cache_agent.py` | Simuler un agent de code et mesurer le prompt cache (tokens en cache, latence) |

```bash
uv run python examples/01_generate.py --prompt "Ta question"
```

Modèle par défaut : `mlx-community/gemma-4-e4b-it-4bit` (~5 Go). Change-le avec `--model`.

## Base de connaissance

| Fiche | Contenu |
|---|---|
| [commandes-cli.md](docs/commandes-cli.md) | Commandes essentielles de `mlx_lm` et `mlx_vlm`, options, exemples |
| [ram.md](docs/ram.md) | Calcul de la RAM : poids et KV cache par format, limite GPU, ce qui tient selon la machine |
| [contexte-performance.md](docs/contexte-performance.md) | Prefill/decode, contexte, KV cache, prompt cache, mode réflexion |
| [serveur-agent-cache.md](docs/serveur-agent-cache.md) | Servir un agent de code : prompt cache borné, budget RAM d'un Mac 24 Go |
| [quantification.md](docs/quantification.md) | De bf16 à moins de 1 bit, quantifications avancées, lire un nom de modèle |
| [modeles-installes.md](docs/modeles-installes.md) | Où sont les modèles, comment les lister, les supprimer, les télécharger |

# Contexte et performance

## Les deux phases d'une génération

| Phase | Ce qui se passe | Limitée par | Métrique |
|---|---|---|---|
| **Prefill** | Le modèle lit tout le prompt d'un coup | La puissance de calcul du GPU | `Prompt: … tokens-per-sec` |
| **Decode** | Il produit la réponse, un token à la fois | La **bande passante mémoire** | `Generation: … tokens-per-sec` |

Le prefill est rapide (des centaines de tokens/s), mais un prompt de 30 000 tokens met quand
même plusieurs secondes avant le premier mot : c'est le *time to first token*.

Le decode relit tous les poids actifs à chaque token. D'où l'ordre de grandeur :

**tokens/s ≈ bande passante mémoire ÷ poids actifs lus par token**

Un M4 de base offre ~120 Go/s, un M4 Pro ~270 Go/s, un M4 Max ~410-546 Go/s. Conséquences :
- un modèle 2× plus petit (4bit au lieu de 8bit) génère presque 2× plus vite ;
- un MoE `A3B` génère à la vitesse d'un modèle de ~3B, même s'il en pèse 35B en RAM ;
- mesuré sur ton M3 avec `gemma-4-e4b-it-4bit` : ~35 tokens/s en génération.

## La fenêtre de contexte

C'est le nombre maximal de tokens (prompt + historique + réponse) que le modèle a appris à
gérer : `max_position_embeddings` dans `config.json`, de 8 000 à plus d'un million selon les modèles.
Un token vaut environ ¾ de mot en anglais, un peu moins en français.

Le contexte maximal est un plafond, pas une consommation : la RAM du KV cache grandit avec
les tokens réellement utilisés (voir le calcul dans [ram.md](ram.md)).

## Les leviers

| Levier | Option | Effet | Coût |
|---|---|---|---|
| Quantifier le KV cache | `--kv-bits 8` ou `4` (`--quantized-kv-start` : à partir de quel token) | KV cache ÷2 ou ÷4 | Légère perte de qualité ; sur le serveur, désactive le batching ; non pris en charge par mlx_lm sur les modèles à fenêtre glissante (Gemma 3/4) |
| Plafonner le KV cache | `--max-kv-size N` | RAM bornée : les plus vieux tokens sont oubliés (cache rotatif) | Le modèle « oublie » le début |
| Prefill par petits lots | `--prefill-step-size 512` | Pic mémoire réduit | Prefill plus lent |
| Réutiliser un préfixe | `mlx_lm.cache_prompt` puis `--prompt-cache-file` | Plus de prefill pour un document ou un prompt système récurrent | Fichier lié à un modèle précis |
| Décodage spéculatif | `--draft-model <petit modèle même famille>` | Decode 1,5-2× plus rapide quand le petit modèle devine bien | RAM du second modèle |
| Modèle plus quantifié | 4bit au lieu de 8bit | Plus rapide, moins de RAM | Qualité (voir [quantification.md](quantification.md)) |

Exemple de prompt cache :

```bash
uv run mlx_lm.cache_prompt --model <repo> --prompt "$(cat long_document.txt)" --prompt-cache-file doc.safetensors
uv run mlx_lm.generate --prompt-cache-file doc.safetensors -p "Résume la section 3"
```

`mlx_lm.server` fait ça automatiquement entre les tours d'une même conversation
(`--prompt-cache-size`, `--prompt-cache-bytes`). Sans plafond en octets, il peut saturer la RAM :
voir [serveur-agent-cache.md](serveur-agent-cache.md).

## Le mode « réflexion »

Les modèles récents (Gemma 4, Qwen 3…) produisent d'abord un raisonnement (`<think>…` ou
`<|channel>thought`) avant la réponse. C'est plus de tokens, donc plus de temps et de KV cache.
Pour le couper :

- en Python : `tokenizer.apply_chat_template(..., enable_thinking=False)` ;
- serveur, globalement : `--chat-template-args '{"enable_thinking":false}'` ;
- serveur, par requête : `"chat_template_kwargs": {"enable_thinking": false}`.

## Comparer des modèles

```bash
uv run mlx_lm.benchmark --model <repo> -p 2048 -g 256 -n 3
```

Compare toujours à prompt et longueur de génération identiques, sur une machine au repos, en
regardant trois chiffres : prefill tokens/s, generation tokens/s et pic mémoire.

## Éviter le réseau au démarrage

Même quand le modèle est en cache, `mlx_lm` interroge le Hub Hugging Face pour vérifier les
fichiers, ce qui peut retarder le démarrage. Constaté ici : plus de 3 minutes avec une
connexion lente. Pour travailler hors ligne :

```bash
export HF_HUB_OFFLINE=1
```

`scripts/serve.py` le fait automatiquement quand le modèle est déjà en cache.

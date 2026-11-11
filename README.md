# inference-with-mlx

Bac à sable et base de connaissance pour faire tourner des LLM en local sur Apple Silicon avec
[MLX](https://github.com/ml-explore/mlx) (`mlx-lm` pour le texte, `mlx-vlm` pour la vision).

## Démarrage en 3 commandes

```bash
uv sync                                            # installe mlx-lm, mlx-vlm, openai dans .venv
uv run python scripts/list_models.py               # quels modèles j'ai, et lesquels tiennent en RAM
uv run mlx_lm.chat --model mlx-community/gemma-4-e4b-it-4bit   # discuter avec un modèle
```

## Installation : projet ou globale

MLX est installé de deux façons sur cette machine. Elles cohabitent sans conflit.

| | Dans le projet (`pyproject.toml`) | Globale (`uv tool`) |
|---|---|---|
| Installation | `uv sync` (déjà fait) | `uv tool install mlx-lm` et `uv tool install mlx-vlm` |
| Usage | `uv run mlx_lm.chat …` depuis ce dossier | `mlx_lm.chat …` depuis n'importe où |
| Python (`import mlx_lm`) | Oui | Non |
| Mise à jour | `uv lock --upgrade && uv sync` | `uv tool upgrade mlx-lm mlx-vlm` |
| Version actuelle | mlx-lm 0.32.2, mlx-vlm 0.7.12 | mlx-lm 0.31.3, mlx-vlm 0.4.4 |

Dans ce dossier, utilise `uv run` pour que doc, scripts et CLI soient sur la même version.

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
- macOS ne donne au GPU que **~75 % de la RAM** : 19,1 Go sur ce Mac (M4, 24 Go).
- Ordre de grandeur des poids : **4bit ≈ 0,56 Go par milliard de paramètres**, 8bit ≈ 1,06 Go.
- Sur ce Mac : jusqu'à ~10 Go de poids, c'est confortable ; 10-15 Go passent en fermant les
  apps ; au-delà de ~18 Go, ça swappe.
- Un MoE (`35B-A3B`) doit tenir **en entier** en RAM, mais génère à la vitesse de ses paramètres actifs.

Calculs, tableaux et réglage de la limite GPU : [docs/ram.md](docs/ram.md).

## Quels modèles sont installés ?

```bash
uv run python scripts/list_models.py         # modèles MLX : type, quantification, poids, RAM estimée, verdict
uv run python scripts/list_models.py --all   # tout le cache Hugging Face
```

Ils sont stockés dans `~/.cache/huggingface/hub`. Commandes alternatives et suppression :
[docs/modeles-installes.md](docs/modeles-installes.md).

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

```bash
uv run python examples/01_generate.py --prompt "Ta question"
```

Modèle par défaut partout : `mlx-community/gemma-4-e4b-it-4bit` (5 Go, ~35 tokens/s sur ce Mac).

## Base de connaissance

| Fiche | Contenu |
|---|---|
| [commandes-cli.md](docs/commandes-cli.md) | Commandes essentielles de `mlx_lm` et `mlx_vlm`, options, exemples |
| [ram.md](docs/ram.md) | Calcul de la RAM : poids, KV cache, limite GPU, ce qui tient sur 24 Go |
| [contexte-performance.md](docs/contexte-performance.md) | Prefill/decode, contexte, KV cache, prompt cache, mode réflexion |
| [quantification.md](docs/quantification.md) | 8/4/3bit, mxfp4, quantifications avancées, lire un nom de modèle |
| [modeles-installes.md](docs/modeles-installes.md) | Où sont les modèles, comment les lister, les supprimer, les télécharger |

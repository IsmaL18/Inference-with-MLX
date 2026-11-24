# inference-with-mlx

Bac à sable et base de connaissance MLX (LLM en local sur Apple Silicon). Doc en français.

## Commandes

- `uv sync` : installe les dépendances (mlx-lm, mlx-vlm, openai).
- `uv run python scripts/list_models.py [--all]` : modèles du cache HF, poids, RAM estimée.
- `uv run python scripts/serve.py [repo] [--vlm] [-- options mlx_lm.server]` : serveur OpenAI local.
- `uv run python examples/0X_*.py` : exemples de l'API Python (`04_` : simulation d'agent, mesure du prompt cache).
- `uvx ruff check . && uvx ruff format --check .` (config dans pyproject) : lint.

## Conventions

- MLX est installé uniquement dans le projet (aucune install globale) : toujours `uv run`.
- Toute option CLI citée dans `docs/` doit être vérifiée avec `uv run <commande> --help` sur la version du projet.
- `scripts/mlx_models.py` est partagé par `list_models.py` et `serve.py` (lancés depuis `scripts/`, import local).

## Pièges

- `mlx_lm.manage --scan` filtre « mlx » en tenant compte de la casse : il rate les repos `...-MLX-...`.
- mlx_lm contacte le Hub même pour un modèle en cache (lent hors ligne) : `HF_HUB_OFFLINE=1`.
- Gemma 4 / Qwen 3 réfléchissent par défaut : `enable_thinking=False` dans le chat template.
- `--kv-bits` échoue sur les modèles à fenêtre glissante (Gemma 3/4) : `RotatingKVCache Quantization NYI`.
- `mlx_lm.server` n'a aucun plafond d'octets pour son prompt cache par défaut (10 entrées max) : toujours passer `--prompt-cache-bytes` (voir `docs/serveur-agent-cache.md`).
- `mlx_vlm.server` écoute sur 0.0.0.0 par défaut : toujours passer `--host 127.0.0.1`.

"""Lance un serveur local compatible OpenAI qui expose un modèle MLX.

    uv run python scripts/serve.py                          # choix interactif parmi les modèles installés
    uv run python scripts/serve.py mlx-community/gemma-4-e4b-it-4bit
    uv run python scripts/serve.py <modèle> --port 8081 -- --max-tokens 2048   # options passées à mlx_lm.server
    uv run python scripts/serve.py <modèle-vision> --vlm    # utilise mlx_vlm.server (images)

Ce script n'est qu'un lanceur : le serveur est celui de MLX (mlx_lm.server ou mlx_vlm.server).
"""

import argparse
import os
import sys

from mlx_models import DEFAULT_MODEL, fit_verdict, gpu_budget_gb, scan_cache


def choose_model() -> str:
    budget = gpu_budget_gb()
    models = sorted(
        (m for m in scan_cache() if m.is_mlx and not m.incomplete),
        key=lambda m: m.weights_bytes,
    )
    print(f"Modèles MLX installés (budget GPU : {budget:.1f} Go) :\n")
    for i, m in enumerate(models, 1):
        kind = "VLM" if m.is_vlm else "LLM"
        print(f"  {i:>2}. {m.repo_id:<72} {kind}  {m.weights_gb:5.1f} Go  tient : {fit_verdict(m, budget)}")
    answer = input(f"\nNuméro du modèle [Entrée = {DEFAULT_MODEL}] : ").strip()
    if not answer:
        return DEFAULT_MODEL
    if answer.isdigit() and 1 <= int(answer) <= len(models):
        return models[int(answer) - 1].repo_id
    sys.exit(f"Choix invalide : {answer!r}")


def main() -> None:
    argv = sys.argv[1:]
    passthrough = []
    if "--" in argv:
        split = argv.index("--")
        argv, passthrough = argv[:split], argv[split + 1 :]

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", nargs="?", help="repo Hugging Face ou chemin local (défaut : choix interactif)")
    parser.add_argument("--host", default="127.0.0.1", help="127.0.0.1 = accessible depuis ce Mac uniquement")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--vlm", action="store_true", help="servir avec mlx_vlm.server (modèles vision)")
    args = parser.parse_args(argv)

    model = args.model or choose_model()
    module = "mlx_vlm.server" if args.vlm else "mlx_lm.server"
    cmd = [sys.executable, "-m", module, "--model", model, "--host", args.host, "--port", str(args.port), *passthrough]

    env = dict(os.environ)
    if any(m.repo_id == model and not m.incomplete for m in scan_cache()):
        # Modèle déjà en cache : pas d'aller-retour réseau vers le Hub au démarrage.
        env.setdefault("HF_HUB_OFFLINE", "1")

    url = f"http://{args.host}:{args.port}/v1"
    print(f"\nModèle  : {model}\nServeur : {module}\nAPI     : {url}  (compatible OpenAI)\n")
    print("Test rapide, depuis un autre terminal :")
    print(
        f"  curl {url}/chat/completions -H 'Content-Type: application/json' \\\n"
        f'    -d \'{{"model": "{model}", "messages": [{{"role": "user", "content": "Bonjour"}}]}}\'\n'
    )
    print("Ctrl+C pour arrêter.\n", flush=True)
    os.execve(sys.executable, cmd, env)


if __name__ == "__main__":
    main()

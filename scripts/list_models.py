"""Liste les modèles MLX du cache Hugging Face, avec leur poids et s'ils tiennent en RAM.

uv run python scripts/list_models.py          # modèles MLX seulement
uv run python scripts/list_models.py --all    # tout le cache (embeddings, modèles HF bruts…)
"""

import argparse

from mlx_models import RAM_OVERHEAD_FIXED_GB, RAM_OVERHEAD_RATIO, fit_verdict, gpu_budget_gb, hub_cache_dir, scan_cache


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--all", action="store_true", help="inclure les modèles non MLX")
    args = parser.parse_args()

    models = [m for m in scan_cache() if args.all or m.is_mlx]
    budget = gpu_budget_gb()

    print(f"Cache : {hub_cache_dir()}")
    print(f"Budget mémoire GPU (macOS) : {budget:.1f} Go\n")
    header = f"{'MODÈLE':<72} {'TYPE':<9} {'QUANT':<9} {'POIDS':>8} {'RAM EST.':>9}  TIENT ?"
    print(header)
    print("-" * len(header))
    for m in sorted(models, key=lambda m: m.weights_bytes):
        kind = "VLM" if m.is_vlm else "LLM"
        if m.is_moe:
            kind += "/MoE"
        if not m.is_mlx:
            kind = "non-MLX"
        print(
            f"{m.repo_id:<72} {kind:<9} {m.quantization:<9} "
            f"{m.weights_gb:>6.1f}Go {m.estimated_ram_gb:>7.1f}Go  {fit_verdict(m, budget)}"
        )
    print(
        f"\n{len(models)} modèle(s). RAM est. = poids × {1 + RAM_OVERHEAD_RATIO:.2f} + {RAM_OVERHEAD_FIXED_GB} Go "
        "(contexte modéré). « juste » = tient seul, en fermant les autres apps. Voir docs/ram.md."
    )


if __name__ == "__main__":
    main()

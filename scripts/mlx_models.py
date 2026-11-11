"""Inventaire des modèles du cache Hugging Face, partagé par list_models.py et serve.py."""

import json
import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_MODEL = "mlx-community/gemma-4-e4b-it-4bit"

# Marge ajoutée au poids des fichiers pour estimer la RAM au chargement :
# KV cache d'un contexte modéré (quelques milliers de tokens) + activations + runtime.
RAM_OVERHEAD_RATIO = 0.15
RAM_OVERHEAD_FIXED_GB = 0.5

GB = 1000**3


@dataclass
class CachedModel:
    repo_id: str
    path: Path
    weights_bytes: int
    is_mlx: bool
    is_vlm: bool
    is_moe: bool
    quantization: str
    incomplete: bool

    @property
    def weights_gb(self) -> float:
        return self.weights_bytes / GB

    @property
    def estimated_ram_gb(self) -> float:
        return self.weights_gb * (1 + RAM_OVERHEAD_RATIO) + RAM_OVERHEAD_FIXED_GB


def hub_cache_dir() -> Path:
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"])
    hf_home = os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")
    return Path(hf_home) / "hub"


def gpu_budget_gb() -> float:
    """Mémoire que macOS laisse au GPU (≈ 75 % de la RAM par défaut)."""
    import mlx.core as mx

    info = mx.device_info() if hasattr(mx, "device_info") else mx.metal.device_info()
    return info["max_recommended_working_set_size"] / GB


def fit_verdict(model: CachedModel, budget_gb: float) -> str:
    if model.incomplete:
        return "incomplet"
    if model.estimated_ram_gb <= 0.8 * budget_gb:
        return "oui"
    if model.estimated_ram_gb <= budget_gb:
        return "juste"
    return "non"


def _latest_snapshot(repo_dir: Path) -> Path | None:
    ref = repo_dir / "refs" / "main"
    if ref.is_file():
        snapshot = repo_dir / "snapshots" / ref.read_text().strip()
        if snapshot.is_dir():
            return snapshot
    snapshots = [p for p in (repo_dir / "snapshots").glob("*") if p.is_dir()]
    return max(snapshots, key=lambda p: p.stat().st_mtime, default=None)


def _describe_quantization(config: dict) -> str:
    quant = config.get("quantization") or config.get("quantization_config")
    if not isinstance(quant, dict) or "bits" not in quant:
        return config.get("torch_dtype") or config.get("dtype") or "?"
    mode = quant.get("mode", "affine")
    return f"{quant['bits']}bit" if mode == "affine" else f"{mode}"


def _has_experts(config: dict) -> bool:
    keys = ("num_experts", "num_local_experts", "n_routed_experts", "moe_num_experts")
    for cfg in (config, config.get("text_config") or {}):
        if any(cfg.get(k) for k in keys):
            return True
    return False


def scan_cache(cache_dir: Path | None = None) -> list[CachedModel]:
    cache_dir = cache_dir or hub_cache_dir()
    models = []
    for repo_dir in sorted(cache_dir.glob("models--*")):
        snapshot = _latest_snapshot(repo_dir)
        if snapshot is None or not (snapshot / "config.json").exists():
            continue
        repo_id = repo_dir.name.removeprefix("models--").replace("--", "/", 1)
        try:
            config = json.loads((snapshot / "config.json").read_text())
        except (OSError, json.JSONDecodeError):
            config = {}
        weight_files = list(snapshot.rglob("*.safetensors")) or list(snapshot.rglob("*.bin"))
        # stat() suit le lien symbolique vers le blob.
        weights = sum(f.stat().st_size for f in weight_files if f.exists())
        incomplete = weights == 0 or any((repo_dir / "blobs").glob("*.incomplete"))
        models.append(
            CachedModel(
                repo_id=repo_id,
                path=snapshot,
                weights_bytes=weights,
                # MLX quantifié => clé "quantization" ; sinon on se fie au nom du dépôt.
                is_mlx="quantization" in config or "mlx" in repo_id.lower(),
                is_vlm="vision_config" in config,
                is_moe=_has_experts(config),
                quantization=_describe_quantization(config),
                incomplete=incomplete,
            )
        )
    return models

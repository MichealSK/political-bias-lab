from __future__ import annotations


def gpu_info() -> dict:
    import torch

    if not torch.cuda.is_available():
        return {"available": False, "name": None, "total_vram_gb": 0.0, "bf16_supported": False}
    props = torch.cuda.get_device_properties(0)
    return {
        "available": True,
        "name": torch.cuda.get_device_name(0),
        "total_vram_gb": round(props.total_memory / 1024**3, 2),
        "bf16_supported": bool(torch.cuda.is_bf16_supported()),
        "cuda": torch.version.cuda,
    }


def require_colab_gpu(min_vram_gb: float = 10.0) -> dict:
    info = gpu_info()
    if not info["available"]:
        raise RuntimeError("No CUDA GPU detected. In Colab choose Runtime > Change runtime type > GPU.")
    if float(info["total_vram_gb"]) < float(min_vram_gb):
        raise RuntimeError(
            f"This run requires approximately {min_vram_gb:.1f} GB of GPU VRAM, but the assigned GPU has "
            f"{info['total_vram_gb']:.1f} GB. Disconnect and request another Colab GPU later, or run a smaller profile."
        )
    return info

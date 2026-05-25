from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from typing import Any

import torch

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover - psutil is optional at runtime
    psutil = None


@dataclass
class HardwareSettings:
    device: str
    cpu_count: int
    cpu_threads: int
    ram_available_gb: float | None
    cuda_available: bool
    gpu_name: str | None
    gpu_total_memory_gb: float | None
    num_workers: int
    pin_memory: bool
    persistent_workers: bool
    prefetch_factor: int | None
    mixed_precision: bool
    autocast_dtype: str
    channels_last: bool


def _available_ram_gb() -> float | None:
    if psutil is None:
        return None
    try:
        return round(float(psutil.virtual_memory().available) / (1024**3), 2)
    except Exception:
        return None


def _gpu_total_memory_gb(device: torch.device) -> float | None:
    if device.type != "cuda":
        return None
    try:
        props = torch.cuda.get_device_properties(device)
        return round(float(props.total_memory) / (1024**3), 2)
    except Exception:
        return None


def _resolve_auto_threads(value: Any, cpu_count: int) -> int:
    if value is None or str(value).lower() in {"auto", "all", "-1"}:
        return max(1, cpu_count)
    return max(1, int(value))


def resolve_num_workers(value: Any, max_workers: int | None = None, reserve_ram_gb: float = 3.0) -> int:
    """Calcula workers para DataLoader sin consumir toda la RAM.

    En equipos con poca RAM conviene no usar todos los cores como workers, porque cada worker
    carga imágenes, máscaras y objetos del processor. Para una CPU con 14 GB de RAM, 3 o 4
    workers suele ser un punto seguro. Si se necesita forzar más, usar --num-workers N.
    """
    cpu_count = os.cpu_count() or 1
    if value is not None and str(value).lower() not in {"auto", "all", "-1"}:
        return max(0, int(value))

    ram_gb = _available_ram_gb()
    if ram_gb is None:
        ram_based = 4
    else:
        # Reserva memoria para el proceso principal, cache de imágenes, CUDA context y SO.
        usable = max(1.0, ram_gb - float(reserve_ram_gb))
        ram_based = max(1, int(usable // 2.0))

    worker_cap = max_workers if max_workers is not None else 4
    return max(0, min(cpu_count, ram_based, int(worker_cap)))


def configure_torch_runtime(cfg: dict[str, Any], device: torch.device) -> HardwareSettings:
    runtime_cfg = cfg.get("runtime", {}) or {}
    cpu_count = os.cpu_count() or 1
    cpu_threads = _resolve_auto_threads(runtime_cfg.get("cpu_threads", "auto"), cpu_count)

    # Variables que suelen leer OpenMP/MKL/OpenBLAS. Se configuran antes de llamar set_num_threads.
    for env_name in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        os.environ[env_name] = str(cpu_threads)

    torch.set_num_threads(cpu_threads)
    # Inter-op alto puede generar sobrecarga; 1 o 2 funciona mejor cuando DataLoader ya paraleliza.
    torch.set_num_interop_threads(int(runtime_cfg.get("interop_threads", 1)))

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = bool(runtime_cfg.get("cudnn_benchmark", True))
        torch.backends.cuda.matmul.allow_tf32 = bool(runtime_cfg.get("allow_tf32", True))
        torch.backends.cudnn.allow_tf32 = bool(runtime_cfg.get("allow_tf32", True))
        precision = str(runtime_cfg.get("float32_matmul_precision", "high"))
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision(precision)

    max_workers = runtime_cfg.get("max_workers", 4)
    num_workers = resolve_num_workers(runtime_cfg.get("num_workers", "auto"), max_workers=max_workers)

    autocast_dtype = str(runtime_cfg.get("autocast_dtype", "float16")).lower()
    if autocast_dtype not in {"float16", "bfloat16"}:
        autocast_dtype = "float16"
    if autocast_dtype == "bfloat16" and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        autocast_dtype = "float16"

    settings = HardwareSettings(
        device=str(device),
        cpu_count=cpu_count,
        cpu_threads=cpu_threads,
        ram_available_gb=_available_ram_gb(),
        cuda_available=torch.cuda.is_available(),
        gpu_name=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        gpu_total_memory_gb=_gpu_total_memory_gb(device),
        num_workers=num_workers,
        pin_memory=bool(runtime_cfg.get("pin_memory", device.type == "cuda")),
        persistent_workers=bool(runtime_cfg.get("persistent_workers", True)) and num_workers > 0,
        prefetch_factor=int(runtime_cfg.get("prefetch_factor", 2)) if num_workers > 0 else None,
        mixed_precision=bool(runtime_cfg.get("mixed_precision", cfg.get("training", {}).get("mixed_precision", True))) and device.type == "cuda",
        autocast_dtype=autocast_dtype,
        channels_last=bool(runtime_cfg.get("channels_last", True)) and device.type == "cuda",
    )
    return settings


def dataloader_kwargs(settings: HardwareSettings, shuffle: bool, worker_init_fn=None) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "shuffle": shuffle,
        "num_workers": settings.num_workers,
        "pin_memory": settings.pin_memory,
        "persistent_workers": settings.persistent_workers,
    }
    if worker_init_fn is not None:
        kwargs["worker_init_fn"] = worker_init_fn
    if settings.prefetch_factor is not None:
        kwargs["prefetch_factor"] = settings.prefetch_factor
    return kwargs


def autocast_dtype(settings: HardwareSettings) -> torch.dtype:
    return torch.bfloat16 if settings.autocast_dtype == "bfloat16" else torch.float16


def hardware_report(settings: HardwareSettings) -> str:
    return json.dumps(asdict(settings), ensure_ascii=False, indent=2)

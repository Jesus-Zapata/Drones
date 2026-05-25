from __future__ import annotations

import argparse

from sam_electric.hardware import configure_torch_runtime, hardware_report
from sam_electric.utils import get_device, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Muestra el perfil de hardware que usará el entrenamiento/evaluación.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--num-workers", default=None)
    parser.add_argument("--cpu-threads", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.num_workers is not None:
        cfg.setdefault("runtime", {})["num_workers"] = args.num_workers
    if args.cpu_threads is not None:
        cfg.setdefault("runtime", {})["cpu_threads"] = args.cpu_threads

    settings = configure_torch_runtime(cfg, get_device())
    print(hardware_report(settings))


if __name__ == "__main__":
    main()

from __future__ import annotations

import importlib.util
from pathlib import Path


def load_service(root: Path):
    service_path = root / "heartbeat-dss-app" / "backend" / "app" / "service.py"
    if not service_path.exists():
        raise FileNotFoundError(f"Service module not found: {service_path}")

    spec = importlib.util.spec_from_file_location("heartbeat_dss_service", service_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import service module from {service_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module.HeartDSSService()

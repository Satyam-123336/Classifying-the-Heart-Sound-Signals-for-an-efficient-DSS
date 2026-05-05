from __future__ import annotations

import importlib.util
import os
from pathlib import Path

ARTIFACTS_DIR = Path("huggingface-space-api/artifacts").resolve()
APP_PATH = Path("huggingface-space-api/app.py").resolve()

os.environ["HF_ARTIFACTS_DIR"] = str(ARTIFACTS_DIR)

spec = importlib.util.spec_from_file_location("hf_space_app", APP_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not load module spec from {APP_PATH}")

module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

service = module.RemoteHeartDSSService()
print(
    {
        "decision_threshold": service.decision_threshold,
        "invert_score": service.invert_score,
        "positive_label": service.positive_label,
        "models_loaded": list(service.model_order),
    }
)

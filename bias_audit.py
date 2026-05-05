from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import joblib


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "training-2022" / "training-2022"
RESULTS_DIR = ROOT / "results"
MODEL_PATH = ROOT / "saved_models" / "svm.pkl"
VALVES = ("AV", "MV", "PV", "TV")


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))

    total = int(y_true.size)
    acc = (tp + tn) / total if total else 0.0

    recall_normal = tp / (tp + fn) if (tp + fn) else 0.0
    specificity_abnormal = tn / (tn + fp) if (tn + fp) else 0.0
    precision_normal = tp / (tp + fp) if (tp + fp) else 0.0
    precision_abnormal = tn / (tn + fn) if (tn + fn) else 0.0

    f1_normal = (
        2 * precision_normal * recall_normal / (precision_normal + recall_normal)
        if (precision_normal + recall_normal)
        else 0.0
    )
    recall_abnormal = specificity_abnormal
    f1_abnormal = (
        2 * precision_abnormal * recall_abnormal / (precision_abnormal + recall_abnormal)
        if (precision_abnormal + recall_abnormal)
        else 0.0
    )

    balanced_accuracy = 0.5 * (recall_normal + specificity_abnormal)

    true_normal_rate = float(np.mean(y_true == 1)) if total else 0.0
    pred_normal_rate = float(np.mean(y_pred == 1)) if total else 0.0

    return {
        "samples": total,
        "tp_normal": tp,
        "fn_normal": fn,
        "tn_abnormal": tn,
        "fp_abnormal": fp,
        "accuracy": round(float(acc), 4),
        "balanced_accuracy": round(float(balanced_accuracy), 4),
        "recall_normal": round(float(recall_normal), 4),
        "specificity_abnormal": round(float(specificity_abnormal), 4),
        "precision_normal": round(float(precision_normal), 4),
        "precision_abnormal": round(float(precision_abnormal), 4),
        "f1_normal": round(float(f1_normal), 4),
        "f1_abnormal": round(float(f1_abnormal), 4),
        "true_normal_rate": round(float(true_normal_rate), 4),
        "pred_normal_rate": round(float(pred_normal_rate), 4),
        "normal_rate_gap": round(float(pred_normal_rate - true_normal_rate), 4),
    }


def calibrated_predictions(scores: np.ndarray, y_true: np.ndarray) -> Tuple[np.ndarray, float, float]:
    candidates = np.unique(np.quantile(scores, np.linspace(0.05, 0.95, 181)))

    best_bal = -1.0
    best_thr = 0.5
    best_pred = (scores >= 0.5).astype(int)

    for thr in candidates:
        pred = (scores >= thr).astype(int)
        m = compute_metrics(y_true, pred)
        bal = float(m["balanced_accuracy"])
        if bal > best_bal:
            best_bal = bal
            best_thr = float(thr)
            best_pred = pred

    return best_pred, best_thr, best_bal


def audit_cached_features() -> Dict[str, object]:
    x = np.load(RESULTS_DIR / "features_reduced.npy")
    y = np.load(RESULTS_DIR / "labels.npy").astype(int)
    clf = joblib.load(MODEL_PATH)

    raw_pred = clf.predict(x).astype(int)

    if hasattr(clf, "predict_proba"):
        scores = clf.predict_proba(x)[:, 1]
    elif hasattr(clf, "decision_function"):
        raw_scores = clf.decision_function(x)
        min_v = float(np.min(raw_scores))
        max_v = float(np.max(raw_scores))
        scores = (raw_scores - min_v) / (max_v - min_v + 1e-12)
    else:
        scores = raw_pred.astype(float)

    cal_pred, threshold, best_bal = calibrated_predictions(scores, y)

    return {
        "raw_svm": compute_metrics(y, raw_pred),
        "calibrated_threshold": {
            "threshold": round(float(threshold), 4),
            "best_balanced_accuracy": round(float(best_bal), 4),
            **compute_metrics(y, cal_pred),
        },
    }


def collect_labeled_wavs(max_per_class_per_valve: int = 10) -> List[Tuple[Path, int, str]]:
    import random

    rng = random.Random(42)
    by_group: Dict[Tuple[str, int], List[Path]] = {}

    for txt in DATA_DIR.glob("*.txt"):
        lines = txt.read_text(encoding="utf-8", errors="ignore").splitlines()
        outcome_line = next((line for line in lines if line.startswith("#Outcome:")), None)
        if not outcome_line:
            continue

        outcome = outcome_line.split(":", 1)[1].strip()
        if outcome not in {"Normal", "Abnormal"}:
            continue

        label = 1 if outcome == "Normal" else 0
        rid = txt.stem

        for valve in VALVES:
            wav = DATA_DIR / f"{rid}_{valve}.wav"
            if wav.exists():
                by_group.setdefault((valve, label), []).append(wav)

    selected: List[Tuple[Path, int, str]] = []
    for valve in VALVES:
        for label in (0, 1):
            group = by_group.get((valve, label), [])
            rng.shuffle(group)
            for wav in group[:max_per_class_per_valve]:
                selected.append((wav, label, valve))

    return selected


def audit_live_service_sample(max_per_class_per_valve: int = 10) -> Dict[str, object]:
    from heartbeat_dss_app_loader import load_service

    service = load_service(ROOT)
    sample = collect_labeled_wavs(max_per_class_per_valve=max_per_class_per_valve)

    y_true_all: List[int] = []
    y_pred_all: List[int] = []
    per_valve: Dict[str, Dict[str, List[int]]] = {v: {"true": [], "pred": []} for v in VALVES}

    for wav, truth, valve in sample:
        audio = wav.read_bytes()
        label, _, _, _, _ = service.predict_from_audio_bytes(audio)
        pred = 1 if label == "Normal" else 0

        y_true_all.append(truth)
        y_pred_all.append(pred)
        per_valve[valve]["true"].append(truth)
        per_valve[valve]["pred"].append(pred)

    y_true_np = np.array(y_true_all, dtype=int)
    y_pred_np = np.array(y_pred_all, dtype=int)

    valve_metrics: Dict[str, Dict[str, float]] = {}
    for valve in VALVES:
        vt = np.array(per_valve[valve]["true"], dtype=int)
        vp = np.array(per_valve[valve]["pred"], dtype=int)
        if vt.size:
            valve_metrics[valve] = compute_metrics(vt, vp)

    return {
        "sample_size": int(y_true_np.size),
        "sampling": {
            "max_per_class_per_valve": int(max_per_class_per_valve),
            "valves": list(VALVES),
        },
        "overall": compute_metrics(y_true_np, y_pred_np),
        "per_valve": valve_metrics,
    }


def main() -> None:
    report: Dict[str, object] = {
        "cached_feature_audit": audit_cached_features(),
    }

    try:
        report["live_service_sample_audit"] = audit_live_service_sample(max_per_class_per_valve=10)
    except Exception as exc:
        report["live_service_sample_audit_error"] = str(exc)

    out_path = ROOT / "bias_audit_report.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()

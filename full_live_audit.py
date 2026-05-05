from __future__ import annotations

import json
import argparse
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from heartbeat_dss_app_loader import load_service


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "training-2022" / "training-2022"
VALVES = ("AV", "MV", "PV", "TV")


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))

    total = int(y_true.size)
    acc = (tp + tn) / total if total else 0.0
    recall_normal = tp / (tp + fn) if (tp + fn) else 0.0
    specificity_abnormal = tn / (tn + fp) if (tn + fp) else 0.0
    bal = 0.5 * (recall_normal + specificity_abnormal)

    pred_normal_rate = float(np.mean(y_pred == 1)) if total else 0.0
    true_normal_rate = float(np.mean(y_true == 1)) if total else 0.0

    return {
        "samples": total,
        "accuracy": round(float(acc), 4),
        "balanced_accuracy": round(float(bal), 4),
        "recall_normal": round(float(recall_normal), 4),
        "specificity_abnormal": round(float(specificity_abnormal), 4),
        "true_normal_rate": round(float(true_normal_rate), 4),
        "pred_normal_rate": round(float(pred_normal_rate), 4),
        "normal_rate_gap": round(float(pred_normal_rate - true_normal_rate), 4),
        "tp": tp,
        "fn": fn,
        "tn": tn,
        "fp": fp,
    }


def collect_records() -> List[Tuple[str, Path, int, str]]:
    records: List[Tuple[str, Path, int, str]] = []

    for txt in DATA_DIR.glob("*.txt"):
        lines = txt.read_text(encoding="utf-8", errors="ignore").splitlines()
        outcome_line = next((line for line in lines if line.startswith("#Outcome:")), None)
        if not outcome_line:
            continue

        outcome = outcome_line.split(":", 1)[1].strip()
        if outcome not in {"Normal", "Abnormal"}:
            continue

        label = 1 if outcome == "Normal" else 0
        patient_id = txt.stem

        for valve in VALVES:
            wav = DATA_DIR / f"{patient_id}_{valve}.wav"
            if wav.exists():
                records.append((patient_id, wav, label, valve))

    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full live audit with progress output")
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Optional cap on number of WAV files to evaluate (0 means all)",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="Print progress every N files",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    service = load_service(ROOT)
    records = collect_records()

    if args.max_files and args.max_files > 0:
        rng = random.Random(42)
        normal = [r for r in records if r[2] == 1]
        abnormal = [r for r in records if r[2] == 0]
        rng.shuffle(normal)
        rng.shuffle(abnormal)

        n_normal = min(len(normal), args.max_files // 2)
        n_abnormal = min(len(abnormal), args.max_files - n_normal)

        selected = normal[:n_normal] + abnormal[:n_abnormal]
        rng.shuffle(selected)
        records = selected

    total_records = len(records)
    if total_records == 0:
        raise RuntimeError("No labeled WAV records found for audit.")

    print(f"[Audit] Starting evaluation for {total_records} files")

    y_true_all: List[int] = []
    y_pred_all: List[int] = []

    per_valve: Dict[str, Dict[str, List[int]]] = {
        valve: {"true": [], "pred": []} for valve in VALVES
    }
    per_patient: Dict[str, Dict[str, List[int]]] = {}
    errors: List[Dict[str, object]] = []

    for idx, (patient_id, wav, truth, valve) in enumerate(records, start=1):
        audio_bytes = wav.read_bytes()
        label, p_normal, p_abnormal, confidence, score = service.predict_from_audio_bytes(audio_bytes)
        pred = 1 if label == "Normal" else 0

        y_true_all.append(truth)
        y_pred_all.append(pred)

        per_valve[valve]["true"].append(truth)
        per_valve[valve]["pred"].append(pred)

        if patient_id not in per_patient:
            per_patient[patient_id] = {"true": [], "pred": []}
        per_patient[patient_id]["true"].append(truth)
        per_patient[patient_id]["pred"].append(pred)

        if pred != truth:
            errors.append(
                {
                    "patient_id": patient_id,
                    "valve": valve,
                    "truth": "Normal" if truth == 1 else "Abnormal",
                    "pred": label,
                    "prob_normal": round(float(p_normal), 4),
                    "prob_abnormal": round(float(p_abnormal), 4),
                    "confidence": round(float(confidence), 4),
                    "score": round(float(score), 4),
                    "wav": str(wav),
                }
            )

        if idx % max(args.progress_every, 1) == 0 or idx == total_records:
            print(f"[Audit] Progress {idx}/{total_records}")

    y_true_np = np.array(y_true_all, dtype=int)
    y_pred_np = np.array(y_pred_all, dtype=int)

    overall = compute_metrics(y_true_np, y_pred_np)

    valve_metrics = {}
    for valve in VALVES:
        vt = np.array(per_valve[valve]["true"], dtype=int)
        vp = np.array(per_valve[valve]["pred"], dtype=int)
        if vt.size:
            valve_metrics[valve] = compute_metrics(vt, vp)

    patient_error_counts: List[Tuple[str, int, int]] = []
    for patient_id, values in per_patient.items():
        yt = np.array(values["true"], dtype=int)
        yp = np.array(values["pred"], dtype=int)
        err = int(np.sum(yt != yp))
        patient_error_counts.append((patient_id, err, int(yt.size)))

    patient_error_counts.sort(key=lambda x: (-x[1], x[0]))

    report = {
        "total_files_evaluated": int(y_true_np.size),
        "overall": overall,
        "per_valve": valve_metrics,
        "worst_patients_top20": [
            {"patient_id": pid, "errors": err, "files": total}
            for pid, err, total in patient_error_counts[:20]
        ],
        "error_examples_top50": errors[:50],
    }

    out_path = ROOT / "full_live_audit_report.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()

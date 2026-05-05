from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROWS_PATH = Path("hf_threshold_live_rows.json")


def metrics_for_threshold(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    pred = (scores >= threshold).astype(int)
    tn = int(np.sum((y_true == 0) & (pred == 0)))
    fp = int(np.sum((y_true == 0) & (pred == 1)))
    fn = int(np.sum((y_true == 1) & (pred == 0)))
    tp = int(np.sum((y_true == 1) & (pred == 1)))

    recall_normal = tn / (tn + fp) if (tn + fp) else 0.0
    recall_abnormal = tp / (tp + fn) if (tp + fn) else 0.0
    accuracy = float(np.mean(pred == y_true))
    balanced_accuracy = 0.5 * (recall_normal + recall_abnormal)
    f1_normal = (2 * tn) / (2 * tn + fp + fn) if (2 * tn + fp + fn) else 0.0
    f1_abnormal = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    macro_f1 = 0.5 * (f1_normal + f1_abnormal)
    pred_normal_rate = float(np.mean(pred == 0))
    true_normal_rate = float(np.mean(y_true == 0))
    normal_rate_gap = abs(pred_normal_rate - true_normal_rate)

    return {
        "threshold": float(threshold),
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": macro_f1,
        "recall_normal": recall_normal,
        "recall_abnormal": recall_abnormal,
        "f1_normal": f1_normal,
        "f1_abnormal": f1_abnormal,
        "pred_normal_rate": pred_normal_rate,
        "true_normal_rate": true_normal_rate,
        "normal_rate_gap": normal_rate_gap,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def scan_thresholds(y_true: np.ndarray, scores: np.ndarray) -> dict:
    unique_scores = np.sort(np.unique(scores))
    if unique_scores.size == 0:
        raise ValueError("No scores found")

    candidates = np.unique(
        np.concatenate(
            [
                [unique_scores[0] - 1e-9],
                unique_scores,
                [unique_scores[-1] + 1e-9],
            ]
        )
    )

    best = None
    for threshold in candidates:
        m = metrics_for_threshold(y_true, scores, float(threshold))
        item = (
            m["balanced_accuracy"],
            m["macro_f1"],
            -m["normal_rate_gap"],
            m["accuracy"],
            -abs(m["pred_normal_rate"] - 0.5),
            -m["threshold"],
        )
        if best is None or item > best["sort_key"]:
            best = {"sort_key": item, "metrics": m}

    return best["metrics"]


def main() -> None:
    rows = json.loads(ROWS_PATH.read_text(encoding="utf-8"))
    y_true = np.array([int(row["truth"]) for row in rows], dtype=int)
    scores = np.array([float(row["score"]) for row in rows], dtype=float)

    direct = scan_thresholds(y_true, scores)
    inverted = scan_thresholds(y_true, 1.0 - scores)

    result = {
        "direct": direct,
        "inverted": inverted,
        "recommended": direct if direct["balanced_accuracy"] >= inverted["balanced_accuracy"] else inverted,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

import json
from pathlib import Path
import random
import requests
import numpy as np

random.seed(26)
root = Path.cwd()
data_dir = root / "training-2022" / "training-2022"
api_url = "http://127.0.0.1:8000/predict"

# Parse patient labels and wav paths from txt metadata files
normal_patients = []
abnormal_patients = []

for txt in data_dir.glob("*.txt"):
    lines = txt.read_text(encoding="utf-8", errors="ignore").splitlines()
    outcome = None
    wavs = []
    for line in lines:
        if line.startswith("#Outcome:"):
            outcome = line.split(":", 1)[1].strip()
        # e.g. "AV 13918_AV.hea 13918_AV.wav 13918_AV.tsv"
        parts = line.strip().split()
        if len(parts) >= 4 and parts[2].lower().endswith(".wav"):
            wav_path = data_dir / parts[2]
            if wav_path.exists():
                wavs.append(wav_path)

    if not outcome or not wavs:
        continue

    rec = {"patient": txt.stem, "outcome": outcome.lower(), "wavs": wavs}
    if outcome.lower() == "normal":
        normal_patients.append(rec)
    elif outcome.lower() == "abnormal":
        abnormal_patients.append(rec)

# Balanced patient sample for external bias check
n = min(60, len(normal_patients), len(abnormal_patients))
normal_sample = random.sample(normal_patients, n)
abnormal_sample = random.sample(abnormal_patients, n)
sample = normal_sample + abnormal_sample
random.shuffle(sample)

results = []
errors = []

for rec in sample:
    true_label = "Normal" if rec["outcome"] == "normal" else "Abnormal"
    for wav in rec["wavs"]:
        try:
            with wav.open("rb") as fh:
                resp = requests.post(api_url, files={"file": (wav.name, fh, "audio/wav")}, timeout=240)
            if resp.status_code != 200:
                errors.append({"file": wav.name, "status": resp.status_code, "body": resp.text[:200]})
                continue
            body = resp.json()
            pred = body.get("label")
            p_norm = float(body.get("probability_normal", np.nan))
            p_abn = float(body.get("probability_abnormal", np.nan))
            score = float(body.get("score", np.nan))
            thr = float(body.get("decision_threshold", np.nan))
            results.append({
                "patient": rec["patient"],
                "file": wav.name,
                "true": true_label,
                "pred": pred,
                "p_normal": p_norm,
                "p_abnormal": p_abn,
                "score": score,
                "threshold": thr,
            })
        except Exception as e:
            errors.append({"file": wav.name, "error": str(e)[:200]})

if not results:
    print(json.dumps({"error": "No successful predictions", "errors": errors[:10]}, indent=2))
    raise SystemExit(1)

true = np.array([r["true"] for r in results])
pred = np.array([r["pred"] for r in results])

# Confusion with Normal as positive class
tp = int(np.sum((pred == "Normal") & (true == "Normal")))
tn = int(np.sum((pred == "Abnormal") & (true == "Abnormal")))
fp = int(np.sum((pred == "Normal") & (true == "Abnormal")))
fn = int(np.sum((pred == "Abnormal") & (true == "Normal")))

normal_recall = tp / (tp + fn) if (tp + fn) else 0.0
abnormal_recall = tn / (tn + fp) if (tn + fp) else 0.0
bal_acc = 0.5 * (normal_recall + abnormal_recall)

actual_normal_rate = float(np.mean(true == "Normal"))
pred_normal_rate = float(np.mean(pred == "Normal"))

# Per-class predicted-normal tendency (bias indicator)
pred_normal_when_true_normal = float(np.mean(pred[true == "Normal"] == "Normal")) if np.any(true == "Normal") else None
pred_normal_when_true_abnormal = float(np.mean(pred[true == "Abnormal"] == "Normal")) if np.any(true == "Abnormal") else None

p_normal_vals = np.array([r["p_normal"] for r in results], dtype=float)
pnorm_true_normal = p_normal_vals[true == "Normal"]
pnorm_true_abnormal = p_normal_vals[true == "Abnormal"]

summary = {
    "sampled_patients_per_class": n,
    "patients_total": len(sample),
    "recordings_scored": len(results),
    "request_errors": len(errors),
    "actual_normal_rate": actual_normal_rate,
    "predicted_normal_rate": pred_normal_rate,
    "normal_recall": normal_recall,
    "abnormal_recall": abnormal_recall,
    "balanced_accuracy": bal_acc,
    "confusion": {"tp_normal": tp, "tn_abnormal": tn, "fp_normal": fp, "fn_normal": fn},
    "bias_indicators": {
        "pred_normal_given_true_normal": pred_normal_when_true_normal,
        "pred_normal_given_true_abnormal": pred_normal_when_true_abnormal
    },
    "p_normal_distribution": {
        "mean_true_normal": float(np.mean(pnorm_true_normal)) if pnorm_true_normal.size else None,
        "mean_true_abnormal": float(np.mean(pnorm_true_abnormal)) if pnorm_true_abnormal.size else None,
        "median_true_normal": float(np.median(pnorm_true_normal)) if pnorm_true_normal.size else None,
        "median_true_abnormal": float(np.median(pnorm_true_abnormal)) if pnorm_true_abnormal.size else None,
    },
    "first_10_errors": errors[:10],
}

print(json.dumps(summary, indent=2))

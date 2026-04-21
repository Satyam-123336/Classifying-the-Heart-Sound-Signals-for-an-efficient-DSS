---
title: Heart Disease Detection
emoji: stethoscope
colorFrom: blue
colorTo: red
sdk: docker
app_file: app.py
pinned: false
---

# Heart Health DSS - Hugging Face Space API

This folder is a Docker Space template that hosts your heartbeat model as a remote API.

## 1) Create Space

- Create a new Hugging Face Space
- SDK: `Docker`
- Visibility: public or private (your choice)

## 2) Upload files to the Space repo

Upload all files from this folder to the root of the Space repo:

- `app.py`
- `requirements.txt`
- `Dockerfile`

Then upload model artifacts under this exact structure:

- `artifacts/saved_models/svm.pkl`
- `artifacts/results/pca_cached.joblib`
- `artifacts/results/decision_threshold_cached.json`

If files are large, use Git LFS in the Space repo.

## 3) Wait for build

After build finishes, test:

- `GET https://<your-space>.hf.space/health`
- `POST https://<your-space>.hf.space/predict` with multipart `file`

## 4) Point local backend to remote Space API

Set these env vars before starting your local backend:

PowerShell:

```powershell
$env:HEARTDSS_REMOTE_API_URL = "https://<your-space>.hf.space/predict"
# Optional for private space/gateway setups:
$env:HEARTDSS_REMOTE_API_TOKEN = "<token>"
$env:HEARTDSS_REMOTE_TIMEOUT = "120"
```

Then run backend normally. Your local backend will forward `/predict` to the Space API and frontend stays unchanged.

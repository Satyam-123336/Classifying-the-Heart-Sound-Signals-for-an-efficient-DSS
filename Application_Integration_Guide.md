# Heartbeat DSS Application Guide (Backend, Frontend, Hugging Face)

Date: 21-Apr-2026

This document explains the application side in easy language:
- local backend API,
- frontend behavior,
- Hugging Face Space API hosting,
- and how all parts connect.

---

## 1) Application architecture

The deployed app has 3 layers:

1. Frontend (React + Vite)
- Folder: `heartbeat-dss-app/frontend`
- User uploads audio and sees prediction cards.

2. Local backend (FastAPI)
- Folder: `heartbeat-dss-app/backend`
- Exposes `/health` and `/predict`.
- Can run in two modes:
  - local model mode (loads `HeartDSSService`)
  - remote proxy mode (forwards requests to Hugging Face Space)

3. Hugging Face Space model API (Docker)
- Folder template: `huggingface-space-api`
- Live repo clone: `Heart-Disease-Detection`
- Runs same inference logic and serves `/health` and `/predict` remotely.

---

## 2) Backend details (FastAPI)

Main file: `heartbeat-dss-app/backend/app/main.py`

### Why backend exists
It is the stable API boundary for the frontend. Frontend never talks to ML internals directly.

### Important behavior

1. Remote mode configuration
- `HEARTDSS_REMOTE_API_URL`
- `HEARTDSS_REMOTE_API_TOKEN` (optional, for private Space)
- `HEARTDSS_REMOTE_TIMEOUT`

If `HEARTDSS_REMOTE_API_URL` is set, backend forwards `/predict` to that URL.

2. Local mode lazy initialization
- If remote URL is not set, backend starts local `HeartDSSService` in background thread.
- Returns warmup status from `/health` while loading.

3. Health endpoint
- `GET /health`
- In remote mode it also checks remote `/health` and reports:
  - `remote_reachable`
  - `remote_error`
  - `model_loaded`

4. Prediction endpoint
- `POST /predict` with multipart `file`
- Validates extension (`wav/mp3/flac/ogg/m4a`), checks non-empty file.
- Returns a structured response with:
  - label/probabilities,
  - decision threshold,
  - margin/strength,
  - human explanation string.

Schema file: `heartbeat-dss-app/backend/app/schemas.py`
- Keeps response contract explicit for frontend and remote validation.

Service file: `heartbeat-dss-app/backend/app/service.py`
- Loads model/PCA/threshold,
- caches PCA and threshold metadata,
- renders chromagram from uploaded audio,
- computes fused features and calibrated decision.

---

## 3) Frontend details (React)

Main file: `heartbeat-dss-app/frontend/src/App.jsx`

### Why frontend is structured this way
It keeps user flow simple: upload audio -> check readiness -> request prediction -> explain result.

### Important behavior

1. API base
- Uses `VITE_API_BASE` or default `/api`.

2. Continuous readiness polling
- Calls `/api/health` every 4 seconds.
- Displays:
  - backend unreachable,
  - model warming,
  - model ready.

3. Upload flow
- File selection and preview URL.
- Sends multipart form to `/api/predict`.

4. Result presentation
- `ResultCard.jsx` shows:
  - label,
  - class probabilities,
  - score,
  - threshold,
  - margin,
  - decision strength,
  - plain-language explanation.

Upload component: `frontend/src/components/UploadCard.jsx`
- Drag-drop + click upload.
- Disabled predict button until API/model readiness is true.

Vite config: `frontend/vite.config.js`
- Proxies `/api` to `http://127.0.0.1:8000`.
- Frontend and backend can run on separate ports without CORS pain.

---

## 4) Hugging Face Space integration

Template files:
- `huggingface-space-api/app.py`
- `huggingface-space-api/Dockerfile`
- `huggingface-space-api/requirements.txt`

Live Space clone:
- `Heart-Disease-Detection/`

### Why Space is used
It hosts the model remotely so local app can call a cloud endpoint instead of loading all ML dependencies locally.

### Space API behavior
- Exposes `/health` and `/predict`.
- Loads artifacts from `./artifacts` by default:
  - `artifacts/saved_models/svm.pkl`
  - `artifacts/saved_models/gradient_boosting.pkl`
  - `artifacts/saved_models/histogram_gradient_boosting.pkl`
  - `artifacts/saved_models/random_forest.pkl`
  - `artifacts/saved_models/adaboost.pkl`
  - `artifacts/results/features_raw.npy`
  - `artifacts/results/features_reduced.npy`
  - `artifacts/results/labels.npy`
  - `artifacts/results/ensemble_weights.json`

### Container/runtime details
- Docker SDK Space with metadata in README frontmatter.
- Uses `opencv-python-headless` for server compatibility.
- Dockerfile installs Linux runtime libs needed by OpenCV stack and copies artifacts into image.

### Local backend -> Space wiring
Set in backend terminal before startup:
- `HEARTDSS_REMOTE_API_URL=https://<space>.hf.space/predict`
- `HEARTDSS_REMOTE_API_TOKEN=<token>` (if Space is private)
- `HEARTDSS_REMOTE_TIMEOUT=180`

Then backend `/predict` becomes a proxy to Space `/predict`.

---

## 5) End-to-end request flow

1. User uploads audio on frontend.
2. Frontend sends multipart file to local backend `/api/predict`.
3. Backend checks mode:
- remote mode: forwards request to Space
- local mode: runs local service inference
4. Prediction response (label + explainability fields) returns to frontend.
5. Frontend renders readable result card.

---

## 6) Commands to run full app

From project root, open two terminals.

Terminal A (backend, remote mode):
```powershell
Set-Location "E:\Taneja's Research"
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
& ".\.venv\Scripts\Activate.ps1"

Set-Location "E:\Taneja's Research\heartbeat-dss-app\backend"
$env:HEARTDSS_REMOTE_API_URL = "https://satysam-26-heart-disease-detection.hf.space/predict"
$env:HEARTDSS_REMOTE_API_TOKEN = "<YOUR_TOKEN>"
$env:HEARTDSS_REMOTE_TIMEOUT = "180"
& "E:\Taneja's Research\.venv\Scripts\python.exe" -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Terminal B (frontend):
```powershell
Set-Location "E:\Taneja's Research"
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
& ".\.venv\Scripts\Activate.ps1"

Set-Location "E:\Taneja's Research\heartbeat-dss-app\frontend"
npm install
npm run dev
```

Open:
- `http://localhost:5173`

Health checks:
```powershell
Invoke-WebRequest "http://127.0.0.1:8000/health" -UseBasicParsing | Select-Object -ExpandProperty Content
Invoke-WebRequest "http://localhost:5173/api/health" -UseBasicParsing | Select-Object -ExpandProperty Content
```

---

## 7) Common errors and fixes

1. `libGL.so.1` missing in Space
- Use `opencv-python-headless` and proper Linux runtime libs in Dockerfile.

2. Space config error in README
- README must include valid YAML frontmatter metadata.

3. Backend starts but `remote_reachable=false`
- Check Space URL, privacy token, and Space build status.

4. Frontend proxy `ECONNREFUSED 127.0.0.1:8000`
- Backend is not running or wrong port.

5. `uvicorn` not found
- Run with venv interpreter:
  - `python -m uvicorn ...`
- Ensure backend requirements installed.

---

## 8) Recommended operational practice

1. Keep one source of truth for model artifacts.
2. Rotate HF token if it was exposed in command history/chat.
3. Use backend as stable API contract; avoid direct frontend-to-Space calls unless intentionally redesigning auth and CORS.
4. Keep `/health` checks in monitoring to detect remote Space downtime quickly.

---

End of application documentation.

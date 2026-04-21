from __future__ import annotations

import io
import json
import os
from pathlib import Path

import cv2
import joblib
import librosa
import librosa.display
import matplotlib
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field
from tensorflow.keras.applications import MobileNetV2, VGG19
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as preprocess_mobilenet
from tensorflow.keras.applications.vgg19 import preprocess_input as preprocess_vgg
from tensorflow.keras.models import Model

matplotlib.use("Agg")
import matplotlib.pyplot as plt


IMAGE_SIZE = (224, 224)


class PredictionResponse(BaseModel):
    label: str = Field(description="Predicted class label: Normal or Abnormal")
    probability_normal: float = Field(ge=0.0, le=1.0)
    probability_abnormal: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    score: float = Field(ge=0.0, le=1.0, description="Calibrated SVM score for Normal class")
    decision_threshold: float = Field(ge=0.0, le=1.0)
    decision_margin: float = Field(ge=0.0, le=1.0)
    decision_strength: str
    explanation: str
    message: str


class RemoteHeartDSSService:
    def __init__(self) -> None:
        artifacts_dir = Path(os.environ.get("HF_ARTIFACTS_DIR", "./artifacts")).resolve()

        self.model_path = artifacts_dir / "saved_models" / "svm.pkl"
        self.pca_cache_path = artifacts_dir / "results" / "pca_cached.joblib"
        self.threshold_cache_path = artifacts_dir / "results" / "decision_threshold_cached.json"

        if not self.model_path.exists():
            raise FileNotFoundError(f"Missing model file: {self.model_path}")
        if not self.pca_cache_path.exists():
            raise FileNotFoundError(f"Missing PCA cache: {self.pca_cache_path}")
        if not self.threshold_cache_path.exists():
            raise FileNotFoundError(f"Missing threshold cache: {self.threshold_cache_path}")

        self.classifier = joblib.load(self.model_path)
        self.pca = joblib.load(self.pca_cache_path)

        threshold_payload = json.loads(self.threshold_cache_path.read_text(encoding="utf-8"))
        self.decision_threshold = float(threshold_payload.get("threshold", 0.5))

        vgg_base = VGG19(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
        self.vgg_model = Model(inputs=vgg_base.input, outputs=vgg_base.output)
        self.vgg_model.trainable = False

        mobilenet_base = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
        self.mobilenet_model = Model(inputs=mobilenet_base.input, outputs=mobilenet_base.output)
        self.mobilenet_model.trainable = False

    @staticmethod
    def _render_chromagram_to_rgb(waveform: np.ndarray, sr: int) -> np.ndarray:
        chroma = librosa.feature.chroma_stft(y=waveform, sr=sr, hop_length=512, n_fft=2048)

        fig = plt.figure(figsize=(15, 5), dpi=100)
        ax = plt.axes([0.0, 0.0, 1.0, 1.0], frameon=False, xticks=[], yticks=[])
        librosa.display.specshow(chroma, sr=sr, hop_length=512, cmap="coolwarm", ax=ax)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches=None, pad_inches=0)
        plt.close(fig)

        img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        bgr = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("Failed to render chromagram image")

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return rgb

    def _extract_fused_features(self, chroma_rgb: np.ndarray) -> np.ndarray:
        resized = cv2.resize(chroma_rgb, IMAGE_SIZE).astype(np.float32)
        batch = np.expand_dims(resized, axis=0)

        vgg_batch = preprocess_vgg(batch.copy())
        mobile_batch = preprocess_mobilenet(batch.copy())

        vgg_feat = self.vgg_model.predict(vgg_batch, verbose=0).reshape(1, -1)
        mobile_feat = self.mobilenet_model.predict(mobile_batch, verbose=0).reshape(1, -1)

        return np.hstack([vgg_feat, mobile_feat]).astype(np.float32, copy=False)

    def predict_from_audio_bytes(self, audio_bytes: bytes) -> PredictionResponse:
        waveform, sr = librosa.load(io.BytesIO(audio_bytes), sr=None, mono=True)
        if waveform.size == 0:
            raise ValueError("Uploaded audio is empty or unreadable")

        chroma_rgb = self._render_chromagram_to_rgb(waveform, sr)
        fused = self._extract_fused_features(chroma_rgb)
        reduced = self.pca.transform(fused).astype(np.float32, copy=False)

        if hasattr(self.classifier, "predict_proba"):
            probability_normal = float(self.classifier.predict_proba(reduced)[0, 1])
        elif hasattr(self.classifier, "decision_function"):
            raw = float(self.classifier.decision_function(reduced)[0])
            probability_normal = 1.0 / (1.0 + np.exp(-raw))
        else:
            probability_normal = float(self.classifier.predict(reduced)[0])

        probability_normal = min(max(probability_normal, 0.0), 1.0)
        score = probability_normal

        label = "Normal" if score >= self.decision_threshold else "Abnormal"
        probability_abnormal = 1.0 - probability_normal
        confidence = probability_normal if label == "Normal" else probability_abnormal

        margin = abs(score - self.decision_threshold)
        if margin < 0.03:
            strength = "borderline"
        elif margin < 0.10:
            strength = "moderate"
        else:
            strength = "strong"

        if label == "Abnormal":
            explanation = (
                f"Abnormal because Normal score {score:.2%} is below threshold {self.decision_threshold:.2%}."
            )
        else:
            explanation = (
                f"Normal because Normal score {score:.2%} is at or above threshold {self.decision_threshold:.2%}."
            )

        return PredictionResponse(
            label=label,
            probability_normal=probability_normal,
            probability_abnormal=probability_abnormal,
            confidence=confidence,
            score=score,
            decision_threshold=self.decision_threshold,
            decision_margin=margin,
            decision_strength=strength,
            explanation=explanation,
            message=f"The uploaded heartbeat appears {label.lower()}.",
        )


app = FastAPI(title="Heart Health DSS Remote API", version="1.0.0")
service: RemoteHeartDSSService | None = None
startup_error: str | None = None


@app.on_event("startup")
def startup_event() -> None:
    global service, startup_error
    try:
        service = RemoteHeartDSSService()
        startup_error = None
    except Exception as exc:
        startup_error = str(exc)
        service = None


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok" if service is not None else "error",
        "model_loaded": service is not None,
        "error": startup_error,
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)) -> PredictionResponse:
    if service is None:
        raise HTTPException(status_code=503, detail=f"Service not ready: {startup_error}")

    ext = Path(file.filename or "").suffix.lower()
    if ext not in {".wav", ".mp3", ".flac", ".ogg", ".m4a"}:
        raise HTTPException(status_code=400, detail="Unsupported audio format")

    audio = await file.read()
    if not audio:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    try:
        return service.predict_from_audio_bytes(audio)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

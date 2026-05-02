"""
Heart Health DSS - Inference API
Using Harris Hawks Optimization (HHO) for feature selection
No AdaBoost - Ensemble of SVM, Gradient Boosting, Histogram Gradient Boosting, Random Forest
Converts audio to chromagram for feature extraction
"""

from __future__ import annotations

import io
import json
import os
import warnings
from pathlib import Path

import cv2
import joblib
import librosa
import librosa.display
import matplotlib
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sklearn.exceptions import InconsistentVersionWarning
from tensorflow.keras.applications import MobileNetV2, VGG19
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as preprocess_mobilenet
from tensorflow.keras.applications.vgg19 import preprocess_input as preprocess_vgg
from tensorflow.keras.models import Model

matplotlib.use("Agg")
import matplotlib.pyplot as plt

APP_BUILD_TAG = "hho-v1.0"
IMAGE_SIZE = (224, 224)

MODEL_FILES = {
    "svm": "svm.pkl",
    "gradient_boosting": "gradient_boosting.pkl",
    "histogram_gradient_boosting": "histogram_gradient_boosting.pkl",
    "random_forest": "random_forest.pkl",
}

HHO_N_SELECT = 512


class PredictionResponse(BaseModel):
    label: str = Field(description="Predicted class label: Normal or Abnormal")
    probability_normal: float = Field(ge=0.0, le=1.0)
    probability_abnormal: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    score: float = Field(ge=0.0, le=1.0, description="Weighted ensemble score for Normal class")
    decision_threshold: float = Field(ge=0.0, le=1.0)
    decision_margin: float = Field(ge=0.0, le=1.0)
    decision_strength: str
    explanation: str
    message: str


class RemoteHeartDSSService:
    def __init__(self) -> None:
        self.artifacts_dir = Path(os.environ.get("HF_ARTIFACTS_DIR", "./artifacts")).resolve()
        self.models_dir = self.artifacts_dir / "saved_models"
        self.ensemble_weights_path = self.artifacts_dir / "results" / "ensemble_weights.json"
        self.hho_indices_path = self.artifacts_dir / "results" / "hho_selected_indices.npy"
        self.threshold_cache_path = self.artifacts_dir / "results" / "decision_threshold_cached.json"

        self.models, self.model_order, self.weights = self._load_models_and_weights()

        self.expected_dim = getattr(self.models[self.model_order[0]], "n_features_in_", None)
        if self.expected_dim is None:
            raise ValueError("Loaded model does not expose n_features_in_.")

        if self.expected_dim != HHO_N_SELECT:
            raise ValueError(
                f"Model expects {self.expected_dim} features, but HHO selects {HHO_N_SELECT}. "
                "Ensure models were trained with matching feature selection."
            )

        for name in self.model_order[1:]:
            dim = getattr(self.models[name], "n_features_in_", None)
            if dim != self.expected_dim:
                raise ValueError(f"Model feature dimension mismatch: {name} has {dim}, expected {self.expected_dim}")

        self.hho_indices = self._load_hho_indices()
        if self.hho_indices is None:
            raise FileNotFoundError(f"Missing HHO indices at {self.hho_indices_path}")

        vgg_base = VGG19(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
        self.vgg_model = Model(inputs=vgg_base.input, outputs=vgg_base.output)
        self.vgg_model.trainable = False

        mobilenet_base = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
        self.mobilenet_model = Model(inputs=mobilenet_base.input, outputs=mobilenet_base.output)
        self.mobilenet_model.trainable = False

        cached_threshold = self._load_cached_threshold()
        # Recalibrated to 0.30 - severe bias from imbalanced training data
        # Normal cases consistently score 30-40%, abnormal score 60+%
        self.decision_threshold = cached_threshold if cached_threshold is not None else 0.30

    def _load_models_and_weights(self):
        models = {}
        for name, filename in MODEL_FILES.items():
            model_path = self.models_dir / filename
            if not model_path.exists():
                raise FileNotFoundError(f"Missing model file: {model_path}")

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", InconsistentVersionWarning)
                models[name] = joblib.load(model_path)

        model_order = list(models.keys())
        weights = np.ones(len(model_order), dtype=np.float64)

        if self.ensemble_weights_path.exists():
            try:
                payload = json.loads(self.ensemble_weights_path.read_text(encoding="utf-8"))
                model_weights = payload.get("models", {})
                if isinstance(model_weights, dict):
                    weights = np.array([float(model_weights.get(name, 0.0)) for name in model_order], dtype=np.float64)
            except Exception:
                pass

        if np.sum(weights) <= 0:
            weights = np.ones(len(model_order), dtype=np.float64)

        weights = weights / np.sum(weights)

        return models, model_order, weights

    def _load_hho_indices(self) -> np.ndarray | None:
        if not self.hho_indices_path.exists():
            return None
        try:
            indices = np.load(self.hho_indices_path)
            return indices.astype(np.int32)
        except Exception:
            return None

    def _load_cached_threshold(self) -> float | None:
        try:
            if self.threshold_cache_path.exists():
                data = json.loads(self.threshold_cache_path.read_text(encoding="utf-8"))
                return float(data.get("decision_threshold", 0.510))
        except Exception:
            pass
        return None

    def extract_features_from_image(self, chroma_rgb: np.ndarray) -> np.ndarray:
        """Extract pooled features from chromagram image."""
        if chroma_rgb.shape != (224, 224, 3):
            chroma_rgb = cv2.resize(chroma_rgb, (224, 224))

        if len(chroma_rgb.shape) == 2:
            chroma_rgb = cv2.cvtColor(chroma_rgb, cv2.COLOR_GRAY2BGR)
        elif chroma_rgb.shape[2] == 4:
            chroma_rgb = cv2.cvtColor(chroma_rgb, cv2.COLOR_BGRA2BGR)

        image_vgg = preprocess_vgg(chroma_rgb.astype(np.float32))
        image_mobilenet = preprocess_mobilenet(chroma_rgb.astype(np.float32))

        vgg_features = self.vgg_model.predict(np.expand_dims(image_vgg, 0), verbose=0)
        mobilenet_features = self.mobilenet_model.predict(np.expand_dims(image_mobilenet, 0), verbose=0)

        vgg_pooled = np.mean(vgg_features[0], axis=(0, 1))
        mobilenet_pooled = np.mean(mobilenet_features[0], axis=(0, 1))

        fused_features = np.concatenate([vgg_pooled, mobilenet_pooled])
        return fused_features

    def select_features_with_hho(self, features: np.ndarray) -> np.ndarray:
        return features[self.hho_indices]

    @staticmethod
    def render_chromagram_to_rgb(audio_bytes: bytes) -> np.ndarray:
        """Convert audio bytes to chromagram image (RGB)."""
        try:
            waveform, sr = librosa.load(io.BytesIO(audio_bytes), sr=None, mono=True)
        except Exception as exc:
            raise ValueError(f"Failed to load audio: {exc}") from exc

        if waveform.size == 0:
            raise ValueError("Audio is empty or unreadable")

        try:
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
                raise ValueError("Failed to decode chromagram image")

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            return rgb
        except Exception as exc:
            raise ValueError(f"Failed to render chromagram: {exc}") from exc


    def predict(self, features: np.ndarray) -> dict:
        if len(features.shape) == 1:
            features = features.reshape(1, -1)

        if features.shape[1] != self.expected_dim:
            raise ValueError(f"Expected {self.expected_dim} features, got {features.shape[1]}")

        probabilities = {}

        for name in self.model_order:
            model = self.models[name]
            pred_proba = model.predict_proba(features)[0]

            classes = getattr(model, "classes_", np.array([0, 1]))
            if 1 in classes:
                idx_normal = np.where(classes == 1)[0][0]
                idx_abnormal = 1 - idx_normal
            else:
                idx_normal = 0
                idx_abnormal = 1

            prob_normal = float(pred_proba[idx_normal])
            probabilities[name] = prob_normal

        ensemble_score_normal = sum(
            self.weights[i] * probabilities[self.model_order[i]]
            for i in range(len(self.model_order))
        )

        ensemble_prediction = 1 if ensemble_score_normal >= self.decision_threshold else 0
        label = "Normal" if ensemble_prediction == 1 else "Abnormal"

        confidence = max(ensemble_score_normal, 1 - ensemble_score_normal)
        decision_margin = abs(ensemble_score_normal - self.decision_threshold)

        if decision_margin > 0.2:
            strength = "Strong"
        elif decision_margin > 0.1:
            strength = "Moderate"
        else:
            strength = "Weak"

        explanation = (
            f"Ensemble score: {ensemble_score_normal:.2%} | "
            f"Threshold: {self.decision_threshold:.2%} | "
            f"Strength: {strength}"
        )

        return {
            "label": label,
            "probability_normal": ensemble_score_normal,
            "probability_abnormal": 1 - ensemble_score_normal,
            "confidence": confidence,
            "score": ensemble_score_normal,
            "decision_threshold": self.decision_threshold,
            "decision_margin": decision_margin,
            "decision_strength": strength,
            "explanation": explanation,
            "message": f"Prediction: {label} (confidence: {confidence:.2%})",
        }


service: RemoteHeartDSSService | None = None


def get_service() -> RemoteHeartDSSService:
    global service
    if service is None:
        service = RemoteHeartDSSService()
    return service


app = FastAPI(
    title="Heart Health DSS - Inference API",
    description="Ensemble prediction using Harris Hawks Optimization (HHO) feature selection",
    version="1.0.0",
)


@app.get("/health")
async def health_check():
    try:
        get_service()
        return {"status": "ok", "service": "Heart Health DSS", "build_tag": APP_BUILD_TAG}
    except Exception as exc:
        return JSONResponse(status_code=503, content={"status": "error", "message": str(exc)})


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    try:
        service = get_service()

        contents = await file.read()
        if not contents:
            raise ValueError("Uploaded file is empty")

        # Convert audio to chromagram
        chroma_rgb = RemoteHeartDSSService.render_chromagram_to_rgb(contents)

        # Extract features from chromagram
        features = service.extract_features_from_image(chroma_rgb)
        
        # Apply HHO feature selection
        selected_features = service.select_features_with_hho(features)
        
        # Predict
        result = service.predict(selected_features)

        return PredictionResponse(**result)

    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/model-info")
async def model_info():
    service = get_service()
    return {
        "ensemble_models": service.model_order,
        "ensemble_weights": {name: float(service.weights[i]) for i, name in enumerate(service.model_order)},
        "feature_count": service.expected_dim,
        "feature_selection_method": "Harris Hawks Optimization (HHO)",
        "decision_threshold": service.decision_threshold,
        "build_tag": APP_BUILD_TAG,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7860)

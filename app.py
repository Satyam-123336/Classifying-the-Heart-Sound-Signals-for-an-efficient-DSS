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
from sklearn.decomposition import IncrementalPCA
from tensorflow.keras.applications import MobileNetV2, VGG19
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as preprocess_mobilenet
from tensorflow.keras.applications.vgg19 import preprocess_input as preprocess_vgg
from tensorflow.keras.models import Model

matplotlib.use("Agg")
import matplotlib.pyplot as plt


IMAGE_SIZE = (224, 224)
PCA_BATCH_SIZE = 128
MODEL_FILES = {
    "svm": "svm.pkl",
    "gradient_boosting": "gradient_boosting.pkl",
    "histogram_gradient_boosting": "histogram_gradient_boosting.pkl",
    "random_forest": "random_forest.pkl",
    "adaboost": "adaboost.pkl",
}


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
        self.raw_feature_cache = self.artifacts_dir / "results" / "features_raw.npy"
        self.reduced_feature_cache = self.artifacts_dir / "results" / "features_reduced.npy"
        self.label_cache = self.artifacts_dir / "results" / "labels.npy"
        self.pca_cache_path = self.artifacts_dir / "results" / "pca_cached.joblib"
        self.pca_meta_path = self.artifacts_dir / "results" / "pca_cached_meta.json"
        self.threshold_cache_path = self.artifacts_dir / "results" / "decision_threshold_cached.json"

        self.models, self.model_order, self.weights = self._load_models_and_weights()

        self.expected_dim = getattr(self.models[self.model_order[0]], "n_features_in_", None)
        if self.expected_dim is None:
            raise ValueError("Loaded model does not expose n_features_in_.")

        for name in self.model_order[1:]:
            dim = getattr(self.models[name], "n_features_in_", None)
            if dim != self.expected_dim:
                raise ValueError(f"Model feature dimension mismatch: {name} has {dim}, expected {self.expected_dim}")

        if not self.raw_feature_cache.exists():
            raise FileNotFoundError(f"Missing raw feature cache: {self.raw_feature_cache}")

        vgg_base = VGG19(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
        self.vgg_model = Model(inputs=vgg_base.input, outputs=vgg_base.output)
        self.vgg_model.trainable = False

        mobilenet_base = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
        self.mobilenet_model = Model(inputs=mobilenet_base.input, outputs=mobilenet_base.output)
        self.mobilenet_model.trainable = False

        self.pca = self._load_cached_pca()
        if self.pca is None:
            self.pca = self._fit_pca_from_cached_raw_features()
            self._save_cached_pca(self.pca)

        self.decision_threshold = self._load_cached_threshold()
        if self.decision_threshold is None:
            self.decision_threshold = self._calibrate_decision_threshold()
            self._save_cached_threshold(self.decision_threshold)

    @staticmethod
    def _safe_mtime(path: Path) -> float:
        try:
            return float(path.stat().st_mtime)
        except Exception:
            return 0.0

    def _load_models_and_weights(self):
        models = {}
        for name, filename in MODEL_FILES.items():
            model_path = self.models_dir / filename
            if model_path.exists():
                models[name] = joblib.load(model_path)

        if not models:
            raise FileNotFoundError(f"No model files found in {self.models_dir}")

        model_order = list(models.keys())
        weights = np.ones(len(model_order), dtype=np.float64)

        if self.ensemble_weights_path.exists():
            try:
                payload = json.loads(self.ensemble_weights_path.read_text(encoding="utf-8"))
                model_weights = payload.get("models", {})
                if isinstance(model_weights, dict):
                    weights = np.array([float(model_weights.get(name, 0.0)) for name in model_order], dtype=np.float64)
            except Exception:
                weights = np.ones(len(model_order), dtype=np.float64)

        if np.sum(weights) <= 0:
            weights = np.ones(len(model_order), dtype=np.float64)

        weights = weights / np.sum(weights)
        return models, model_order, weights

    def _pca_meta(self) -> dict:
        return {
            "expected_dim": int(self.expected_dim),
            "raw_feature_cache_mtime": self._safe_mtime(self.raw_feature_cache),
            "models_dir_mtime": self._safe_mtime(self.models_dir),
            "weights_mtime": self._safe_mtime(self.ensemble_weights_path),
        }

    def _threshold_meta(self) -> dict:
        return {
            "expected_dim": int(self.expected_dim),
            "models_dir_mtime": self._safe_mtime(self.models_dir),
            "weights_mtime": self._safe_mtime(self.ensemble_weights_path),
            "reduced_feature_cache_mtime": self._safe_mtime(self.reduced_feature_cache),
            "label_cache_mtime": self._safe_mtime(self.label_cache),
        }

    def _load_cached_pca(self) -> IncrementalPCA | None:
        if not self.pca_cache_path.exists() or not self.pca_meta_path.exists():
            return None

        try:
            cached_meta = json.loads(self.pca_meta_path.read_text(encoding="utf-8"))
            if cached_meta != self._pca_meta():
                return None

            pca = joblib.load(self.pca_cache_path)
            n_components = int(getattr(pca, "n_components_", 0))
            if n_components != int(self.expected_dim):
                return None
            return pca
        except Exception:
            return None

    def _save_cached_pca(self, pca: IncrementalPCA) -> None:
        try:
            self.pca_cache_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(pca, self.pca_cache_path)
            self.pca_meta_path.write_text(json.dumps(self._pca_meta()), encoding="utf-8")
        except Exception:
            pass

    def _load_cached_threshold(self) -> float | None:
        if not self.threshold_cache_path.exists():
            return None

        try:
            payload = json.loads(self.threshold_cache_path.read_text(encoding="utf-8"))
            if payload.get("meta", {}) != self._threshold_meta():
                return None
            threshold = float(payload.get("threshold"))
            if not np.isfinite(threshold):
                return None
            return threshold
        except Exception:
            return None

    def _save_cached_threshold(self, threshold: float) -> None:
        try:
            self.threshold_cache_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {"threshold": float(threshold), "meta": self._threshold_meta()}
            self.threshold_cache_path.write_text(json.dumps(payload), encoding="utf-8")
        except Exception:
            pass

    def _fit_pca_from_cached_raw_features(self) -> IncrementalPCA:
        raw_features = np.load(self.raw_feature_cache, mmap_mode="r")

        n_samples = raw_features.shape[0]
        effective_batch = min(n_samples, max(PCA_BATCH_SIZE, self.expected_dim))

        if self.expected_dim > effective_batch:
            raise ValueError(
                f"Cannot fit IncrementalPCA with n_components={self.expected_dim} and batch={effective_batch}."
            )

        pca = IncrementalPCA(n_components=self.expected_dim)
        for i in range(0, n_samples, effective_batch):
            batch = raw_features[i : i + effective_batch].astype(np.float32, copy=False)
            pca.partial_fit(batch)
        return pca

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

    @staticmethod
    def _probability_scores(model, x: np.ndarray) -> np.ndarray:
        if hasattr(model, "predict_proba"):
            return model.predict_proba(x)[:, 1].astype(np.float64)

        if hasattr(model, "decision_function"):
            raw = model.decision_function(x).astype(np.float64)
            min_v = float(np.min(raw))
            max_v = float(np.max(raw))
            if max_v - min_v < 1e-8:
                return np.full_like(raw, 0.5, dtype=np.float64)
            return (raw - min_v) / (max_v - min_v)

        return model.predict(x).astype(np.float64)

    def _ensemble_score(self, x: np.ndarray) -> np.ndarray:
        per_model = [self._probability_scores(self.models[name], x) for name in self.model_order]
        proba_matrix = np.column_stack(per_model)
        return proba_matrix @ self.weights

    def _calibrate_decision_threshold(self) -> float:
        if not self.label_cache.exists():
            return 0.5

        y_true = np.load(self.label_cache)
        if y_true.ndim != 1 or y_true.size == 0 or len(np.unique(y_true)) < 2:
            return 0.5

        if self.reduced_feature_cache.exists():
            x = np.load(self.reduced_feature_cache)
        else:
            raw = np.load(self.raw_feature_cache, mmap_mode="r")
            x = self.pca.transform(np.asarray(raw, dtype=np.float32))

        if x.shape[0] != y_true.shape[0]:
            return 0.5

        scores = self._ensemble_score(x)

        best_thr = 0.5
        best_bal = -1.0

        candidates = np.unique(np.quantile(scores, np.linspace(0.05, 0.95, 181)))
        for thr in candidates:
            pred = (scores >= thr).astype(int)
            tn = int(np.sum((y_true == 0) & (pred == 0)))
            fp = int(np.sum((y_true == 0) & (pred == 1)))
            fn = int(np.sum((y_true == 1) & (pred == 0)))
            tp = int(np.sum((y_true == 1) & (pred == 1)))

            sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            bal = 0.5 * (sens + spec)

            if bal > best_bal:
                best_bal = bal
                best_thr = float(thr)

        return best_thr

    def predict_from_audio_bytes(self, audio_bytes: bytes) -> PredictionResponse:
        waveform, sr = librosa.load(io.BytesIO(audio_bytes), sr=None, mono=True)
        if waveform.size == 0:
            raise ValueError("Uploaded audio is empty or unreadable")

        chroma_rgb = self._render_chromagram_to_rgb(waveform, sr)
        fused = self._extract_fused_features(chroma_rgb)
        reduced = self.pca.transform(fused).astype(np.float32, copy=False)

        score = float(self._ensemble_score(reduced)[0])
        score = min(max(score, 0.0), 1.0)

        label = "Normal" if score >= self.decision_threshold else "Abnormal"
        probability_normal = score
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

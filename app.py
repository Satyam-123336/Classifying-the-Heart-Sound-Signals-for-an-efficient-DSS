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
        self.reduced_feature_cache = self.artifacts_dir / "results" / "features_reduced.npy"
        self.label_cache = self.artifacts_dir / "results" / "labels.npy"
        self.pca_cache_path = self.artifacts_dir / "results" / "pca_cached.joblib"
        self.pca_meta_path = self.artifacts_dir / "results" / "pca_cached_meta.json"
        self.threshold_cache_path = self.artifacts_dir / "results" / "decision_threshold_cached.json"
        self.invert_score = False
        self.positive_label = "normal"

        self.models, self.model_order, self.weights = self._load_models_and_weights()

        self.expected_dim = getattr(self.models[self.model_order[0]], "n_features_in_", None)
        if self.expected_dim is None:
            raise ValueError("Loaded model does not expose n_features_in_.")

        for name in self.model_order[1:]:
            dim = getattr(self.models[name], "n_features_in_", None)
            if dim != self.expected_dim:
                raise ValueError(f"Model feature dimension mismatch: {name} has {dim}, expected {self.expected_dim}")

        vgg_base = VGG19(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
        self.vgg_model = Model(inputs=vgg_base.input, outputs=vgg_base.output)
        self.vgg_model.trainable = False

        mobilenet_base = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
        self.mobilenet_model = Model(inputs=mobilenet_base.input, outputs=mobilenet_base.output)
        self.mobilenet_model.trainable = False

        self.pca = self._load_cached_pca()
        if self.pca is None:
            raise FileNotFoundError(
                f"Missing PCA cache. Provide {self.pca_cache_path} before starting the service."
            )

        cached_threshold, cached_invert = self._load_cached_threshold()
        self.decision_threshold = cached_threshold
        self.invert_score = cached_invert
        if self.decision_threshold is None:
            self.decision_threshold, self.invert_score = self._calibrate_decision_threshold()
            self._save_cached_threshold(self.decision_threshold, self.invert_score)

        self.positive_label = self._resolve_positive_label()

    def _resolve_positive_label(self) -> str:
        env_value = os.environ.get("HEARTDSS_POSITIVE_LABEL", "").strip().lower()
        if env_value in {"normal", "abnormal"}:
            return env_value

        # Heuristic fallback: very low threshold usually indicates class-1 is the opposite semantic.
        if float(self.decision_threshold) < 0.3:
            return "abnormal"
        return "normal"

    @staticmethod
    def _safe_mtime(path: Path) -> float:
        try:
            return float(path.stat().st_mtime)
        except Exception:
            return 0.0

    def _pca_meta_is_compatible(self, cached_meta: object) -> bool:
        if not isinstance(cached_meta, dict):
            return False

        expected_dim = cached_meta.get("expected_dim")
        if expected_dim is not None:
            try:
                if int(expected_dim) != int(self.expected_dim):
                    return False
            except Exception:
                return False

        return True

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
            "reduced_feature_cache_mtime": self._safe_mtime(self.reduced_feature_cache),
            "results_dir_mtime": self._safe_mtime(self.artifacts_dir / "results"),
        }

    def _threshold_meta(self) -> dict:
        return {
            "expected_dim": int(self.expected_dim),
            "build_tag": APP_BUILD_TAG,
            "models_dir_mtime": self._safe_mtime(self.models_dir),
            "weights_mtime": self._safe_mtime(self.ensemble_weights_path),
            "reduced_feature_cache_mtime": self._safe_mtime(self.reduced_feature_cache),
            "label_cache_mtime": self._safe_mtime(self.label_cache),
        }

    def _load_cached_pca(self) -> IncrementalPCA | None:
        if not self.pca_cache_path.exists():
            return None

        try:
            pca = joblib.load(self.pca_cache_path)
            n_components = int(getattr(pca, "n_components_", 0))
            if n_components != int(self.expected_dim):
                return None

            if self.pca_meta_path.exists():
                try:
                    cached_meta = json.loads(self.pca_meta_path.read_text(encoding="utf-8"))
                except Exception:
                    return None
                if not self._pca_meta_is_compatible(cached_meta):
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

    def _load_cached_threshold(self) -> tuple[float | None, bool]:
        if not self.threshold_cache_path.exists():
            return None, False

        try:
            payload = json.loads(self.threshold_cache_path.read_text(encoding="utf-8"))
            if payload.get("meta", {}) != self._threshold_meta():
                return None, False
            if "invert_score" not in payload:
                return None, False
            threshold = float(payload.get("threshold"))
            if not np.isfinite(threshold):
                return None, False
            invert_score = bool(payload.get("invert_score", False))
            return threshold, invert_score
        except Exception:
            return None, False

    def _save_cached_threshold(self, threshold: float, invert_score: bool) -> None:
        try:
            self.threshold_cache_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "threshold": float(threshold),
                "invert_score": bool(invert_score),
                "meta": self._threshold_meta(),
            }
            self.threshold_cache_path.write_text(json.dumps(payload), encoding="utf-8")
        except Exception:
            pass

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

    def _calibrate_decision_threshold(self) -> tuple[float, bool]:
        if not self.label_cache.exists():
            return 0.5, False

        y_true = np.load(self.label_cache)
        if y_true.ndim != 1 or y_true.size == 0 or len(np.unique(y_true)) < 2:
            return 0.5, False

        if not self.reduced_feature_cache.exists():
            return 0.5, False

        x = np.load(self.reduced_feature_cache)

        if x.shape[0] != y_true.shape[0]:
            return 0.5, False

        scores = self._ensemble_score(x)

        def select_threshold(for_scores: np.ndarray) -> tuple[float, float, float]:
            local_best_thr = 0.5
            local_best_bal = -1.0
            local_best_rate = 1.0
            prevalence_target = float(np.mean(y_true == 1))

            best_prev_thr = 0.5
            best_prev_gap = 1e9
            best_prev_bal = -1.0
            candidates = np.unique(np.quantile(for_scores, np.linspace(0.05, 0.95, 181)))
            for thr in candidates:
                pred = (for_scores >= thr).astype(int)
                tn = int(np.sum((y_true == 0) & (pred == 0)))
                fp = int(np.sum((y_true == 0) & (pred == 1)))
                fn = int(np.sum((y_true == 1) & (pred == 0)))
                tp = int(np.sum((y_true == 1) & (pred == 1)))

                sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                bal = 0.5 * (sens + spec)
                pred_rate = float(np.mean(pred == 1))
                prev_gap = abs(pred_rate - prevalence_target)

                if bal > local_best_bal:
                    local_best_bal = bal
                    local_best_thr = float(thr)
                    local_best_rate = pred_rate

                if (prev_gap < best_prev_gap) or (abs(prev_gap - best_prev_gap) < 1e-12 and bal > best_prev_bal):
                    best_prev_gap = prev_gap
                    best_prev_thr = float(thr)
                    best_prev_bal = bal

            # Prevent degenerate all-one/all-zero behavior from dominating calibration.
            if local_best_rate < 0.2 or local_best_rate > 0.8:
                return best_prev_thr, best_prev_bal, best_prev_gap
            return local_best_thr, local_best_bal, abs(local_best_rate - prevalence_target)

        direct_thr, direct_bal, direct_gap = select_threshold(scores)
        inv_scores = 1.0 - scores
        inv_thr, inv_bal, inv_gap = select_threshold(inv_scores)

        if (inv_bal > direct_bal) or (abs(inv_bal - direct_bal) < 1e-12 and inv_gap < direct_gap):
            return inv_thr, True
        return direct_thr, False

    def predict_from_audio_bytes(self, audio_bytes: bytes) -> PredictionResponse:
        waveform, sr = librosa.load(io.BytesIO(audio_bytes), sr=None, mono=True)
        if waveform.size == 0:
            raise ValueError("Uploaded audio is empty or unreadable")

        chroma_rgb = self._render_chromagram_to_rgb(waveform, sr)
        fused = self._extract_fused_features(chroma_rgb)
        reduced = self.pca.transform(fused).astype(np.float32, copy=False)

        raw_score = float(self._ensemble_score(reduced)[0])
        raw_score = min(max(raw_score, 0.0), 1.0)
        score = 1.0 - raw_score if self.invert_score else raw_score

        if self.positive_label == "abnormal":
            probability_abnormal = score
            probability_normal = 1.0 - probability_abnormal
            label = "Abnormal" if probability_abnormal >= self.decision_threshold else "Normal"
        else:
            probability_normal = score
            probability_abnormal = 1.0 - probability_normal
            label = "Normal" if probability_normal >= self.decision_threshold else "Abnormal"

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
APP_BUILD_TAG = "2026-04-26-e3b6132-threshold-52435974"


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
    runtime = {
        "build": APP_BUILD_TAG,
        "decision_threshold": None,
        "invert_score": None,
        "positive_label": None,
    }
    if service is not None:
        runtime["decision_threshold"] = float(service.decision_threshold)
        runtime["invert_score"] = bool(getattr(service, "invert_score", False))
        runtime["positive_label"] = str(getattr(service, "positive_label", "normal"))

    return {
        "status": "ok" if service is not None else "error",
        "model_loaded": service is not None,
        "error": startup_error,
        "runtime": runtime,
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

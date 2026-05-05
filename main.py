import json
import os
from datetime import datetime

import cv2
import joblib
import numpy as np
from sklearn.decomposition import IncrementalPCA
from tensorflow.keras.applications import MobileNetV2, VGG19
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as preprocess_mobilenet
from tensorflow.keras.applications.vgg19 import preprocess_input as preprocess_vgg
from tensorflow.keras.models import Model
from tqdm import tqdm


IMAGE_FOLDER = "chroma-2022"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
PCA_BATCH_SIZE = 128
MODELS_DIR = "saved_models"
RESULTS_DIR = "results"
RAW_FEATURE_CACHE_PATH = os.path.join(RESULTS_DIR, "features_raw.npy")
REDUCED_FEATURE_CACHE_PATH = os.path.join(RESULTS_DIR, "features_reduced.npy")
WEIGHTS_PATH = os.path.join(RESULTS_DIR, "ensemble_weights.json")

MODEL_FILES = {
    "svm": "svm.pkl",
    "gradient_boosting": "gradient_boosting.pkl",
    "histogram_gradient_boosting": "histogram_gradient_boosting.pkl",
    "random_forest": "random_forest.pkl",
    "adaboost": "adaboost.pkl",
}


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def get_image_paths(folder: str) -> list[str]:
    valid_ext = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    if not os.path.isdir(folder):
        raise FileNotFoundError(
            f"Image folder not found: {folder}. "
            "Generate chromagrams first using generate_cgram_modified.py"
        )

    paths = []
    for name in sorted(os.listdir(folder)):
        if name.lower().endswith(valid_ext):
            paths.append(os.path.join(folder, name))

    if not paths:
        raise ValueError(f"No image files found in {folder}")
    return paths


def extract_features(model: Model, preprocess_func, image_paths: list[str], batch_size: int = BATCH_SIZE) -> np.ndarray:
    features = []
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Feature batches", mininterval=1):
        batch_paths = image_paths[i : i + batch_size]
        batch = []

        for img_path in batch_paths:
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: {os.path.basename(img_path)} could not be read. Skipping.")
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, IMAGE_SIZE)
            batch.append(img)

        if not batch:
            continue

        batch = np.asarray(batch, dtype=np.float32)
        batch = preprocess_func(batch)
        feat = model.predict(batch, verbose=0)
        features.append(feat.reshape(feat.shape[0], -1))

    if not features:
        raise ValueError("No readable images found during feature extraction")

    return np.vstack(features).astype(np.float32)


def load_models_and_weights() -> tuple[dict[str, object], np.ndarray, list[str], int]:
    models: dict[str, object] = {}

    for key, filename in MODEL_FILES.items():
        path = os.path.join(MODELS_DIR, filename)
        if os.path.exists(path):
            models[key] = joblib.load(path)

    if not models:
        raise FileNotFoundError(
            f"No trained model files found in {MODELS_DIR}. Run heart_health_dss.py first."
        )

    expected_dims = {
        name: getattr(model, "n_features_in_", None)
        for name, model in models.items()
    }
    if any(v is None for v in expected_dims.values()):
        raise ValueError("One or more models do not expose n_features_in_.")

    unique_dims = {int(v) for v in expected_dims.values()}
    if len(unique_dims) != 1:
        raise ValueError(f"Model feature dimension mismatch: {expected_dims}")
    expected_dim = unique_dims.pop()

    model_names = list(models.keys())
    weights = np.ones(len(model_names), dtype=np.float64)

    if os.path.exists(WEIGHTS_PATH):
        with open(WEIGHTS_PATH, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, dict) and isinstance(payload.get("models"), dict):
            raw = payload["models"]
            weights = np.array([float(raw.get(name, 0.0)) for name in model_names], dtype=np.float64)

    if np.sum(weights) <= 0:
        weights = np.ones(len(model_names), dtype=np.float64)
    weights = weights / np.sum(weights)

    return models, weights, model_names, expected_dim


def fit_pca_from_raw_cache(expected_dim: int) -> IncrementalPCA:
    if not os.path.exists(RAW_FEATURE_CACHE_PATH):
        raise FileNotFoundError(
            f"Missing raw feature cache: {RAW_FEATURE_CACHE_PATH}. "
            "Run heart_health_dss.py first."
        )

    raw = np.load(RAW_FEATURE_CACHE_PATH, mmap_mode="r")
    n_samples = raw.shape[0]
    effective_batch = min(n_samples, max(PCA_BATCH_SIZE, expected_dim))

    if expected_dim > effective_batch:
        raise ValueError(
            f"Cannot fit IncrementalPCA with n_components={expected_dim} and batch={effective_batch}."
        )

    pca = IncrementalPCA(n_components=expected_dim)
    for i in tqdm(range(0, n_samples, effective_batch), desc="PCA fit", mininterval=1):
        batch = raw[i : i + effective_batch].astype(np.float32, copy=False)
        pca.partial_fit(batch)

    return pca


def probability_scores(model, x: np.ndarray) -> np.ndarray:
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


def main() -> None:
    image_paths = get_image_paths(IMAGE_FOLDER)
    log(f"Total image files detected: {len(image_paths)}")

    log("Loading saved models and ensemble weights...")
    models, weights, model_names, expected_dim = load_models_and_weights()
    for name, w in zip(model_names, weights):
        log(f"  model={name:<28} weight={w:.4f}")

    log("Loading VGG19 and MobileNetV2 models...")
    vgg_base = VGG19(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    vgg_model = Model(inputs=vgg_base.input, outputs=vgg_base.output)
    vgg_model.trainable = False

    mobilenet_base = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    mobilenet_model = Model(inputs=mobilenet_base.input, outputs=mobilenet_base.output)
    mobilenet_model.trainable = False

    log("Extracting features from VGG19...")
    vgg_features = extract_features(vgg_model, preprocess_vgg, image_paths)
    log(f"VGG19 features shape: {vgg_features.shape}")

    log("Extracting features from MobileNetV2...")
    mobilenet_features = extract_features(mobilenet_model, preprocess_mobilenet, image_paths)
    log(f"MobileNetV2 features shape: {mobilenet_features.shape}")

    fused_features = np.hstack([vgg_features, mobilenet_features]).astype(np.float32, copy=False)
    log(f"Fused features shape: {fused_features.shape}")

    if os.path.exists(REDUCED_FEATURE_CACHE_PATH):
        cached = np.load(REDUCED_FEATURE_CACHE_PATH, mmap_mode="r")
        if cached.shape[0] == len(image_paths) and cached.shape[1] == expected_dim:
            log(f"Using cached reduced features from {REDUCED_FEATURE_CACHE_PATH} with shape {cached.shape}")
            model_input = np.asarray(cached, dtype=np.float32)
        else:
            log("Reduced feature cache does not match current run. Recomputing PCA projection...")
            pca = fit_pca_from_raw_cache(expected_dim)
            model_input = pca.transform(fused_features).astype(np.float32, copy=False)
    else:
        pca = fit_pca_from_raw_cache(expected_dim)
        model_input = pca.transform(fused_features).astype(np.float32, copy=False)

    log("Computing weighted soft-voting ensemble score...")
    per_model_scores = [probability_scores(models[name], model_input) for name in model_names]
    proba_matrix = np.column_stack(per_model_scores)
    ensemble_scores = proba_matrix @ weights
    final_pred = (ensemble_scores >= 0.5).astype(int)
    final_decision = np.where(final_pred == 1, "Normal", "Abnormal").tolist()

    num_normal = final_decision.count("Normal")
    num_abnormal = final_decision.count("Abnormal")

    print("\n===== DSS RESULTS =====")
    print("Total images:", len(final_decision))
    print("Predicted Normal:", num_normal)
    print("Predicted Abnormal:", num_abnormal)
    print("========================")

    np.save("vgg_features.npy", vgg_features)
    np.save("mobilenet_features.npy", mobilenet_features)
    np.save("dss_decisions.npy", np.array(final_decision))

    log("Feature arrays and DSS decisions saved successfully.")


if __name__ == "__main__":
    main()

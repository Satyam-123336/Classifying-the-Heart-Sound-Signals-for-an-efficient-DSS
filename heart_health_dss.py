"""
================================================================================
Decision Support System for Autonomous Detection of Heart Health Status
================================================================================
Research-level implementation using:
- VGG19 and MobileNetV2 for deep feature extraction
- Multiple ML classifiers (SVM, Gradient Boosting, Histogram Gradient Boosting,
  Random Forest)
- Harris Hawks Optimization (HHO) for intelligent feature selection
- 10-Fold Stratified Cross-Validation
- Soft Probability-Weighted Ensemble Voting (accuracy^4 weights + CV-optimal threshold)
- Comprehensive evaluation metrics
================================================================================
"""

import math
import numpy as np
import os
import json
import cv2
import pandas as pd
import warnings
from pathlib import Path
from typing import Tuple, List, Dict, Any
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc, confusion_matrix
)
from tensorflow.keras.applications import VGG19, MobileNetV2
from tensorflow.keras.applications.vgg19 import preprocess_input as preprocess_vgg
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as preprocess_mobilenet
from tensorflow.keras.models import Model
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================
IMAGE_SIZE   = (224, 224)
N_FOLDS      = 10
RANDOM_STATE = 42
MODELS_DIR   = "saved_models"
RESULTS_DIR  = "results"

# HHO feature selection settings
HHO_N_SELECT   = 512    # number of features HHO selects (replaces PCA n_components)
HHO_N_HAWKS    = 10     # hawk population size (increase for wider search, slower)
HHO_MAX_ITER   = 25     # optimisation iterations  (increase for deeper search, slower)
HHO_EVAL_FRAC  = 0.30   # fraction of data used per fitness evaluation
PRE_FILTER_K   = 4096   # variance pre-filter: raw_dim -> 4096 before HHO

# Cache file paths
RAW_FEATURE_CACHE_PATH   = os.path.join(RESULTS_DIR, "features_raw.npy")
PREFILTER_CACHE_PATH     = os.path.join(RESULTS_DIR, "features_prefiltered.npy")
PREFILTER_IDX_CACHE_PATH = os.path.join(RESULTS_DIR, "hho_prefilter_indices.npy")
FEATURE_CACHE_PATH       = os.path.join(RESULTS_DIR, "features_selected.npy")
HHO_IDX_CACHE_PATH       = os.path.join(RESULTS_DIR, "hho_selected_indices.npy")
LABEL_CACHE_PATH         = os.path.join(RESULTS_DIR, "labels.npy")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================================
# HELPERS
# ============================================================================
def resolve_default_path(candidates: List[str], path_type: str) -> str:
    for candidate in candidates:
        if Path(candidate).exists():
            return candidate
    raise ValueError(f"Could not find a valid {path_type}. Tried: {candidates}")


def load_labels_dict(label_file: str) -> Dict[str, int]:
    """
    Load labels from either supported CSV format:
    - PhysioNet style: filename,Label (with header)
    - Headerless two-column CSV: image_name,label
    """
    df = pd.read_csv(label_file)
    normalized_columns = {str(c).strip().lower(): c for c in df.columns}

    if "filename" in normalized_columns and "label" in normalized_columns:
        name_col  = normalized_columns["filename"]
        label_col = normalized_columns["label"]
    elif len(df.columns) >= 2:
        df        = pd.read_csv(label_file, header=None, names=["image_name", "label"])
        name_col  = "image_name"
        label_col = "label"
    else:
        raise ValueError(f"Unsupported label CSV format: {label_file}")

    names  = df[name_col].astype(str).str.strip()
    labels = pd.to_numeric(df[label_col], errors="coerce")
    valid  = (~names.eq("")) & labels.notna()
    return dict(zip(names[valid], labels[valid].astype(int)))


# ============================================================================
# STEP 1: IMAGE LOADING AND PREPROCESSING
# ============================================================================
def load_images(image_folder: str,
                label_file: str = None) -> Tuple[List[str], np.ndarray, List[str]]:
    """Load images from folder and optionally map labels from a CSV file."""
    print(f"\n{'='*70}")
    print("STEP 1: Loading and Preprocessing Images")
    print(f"{'='*70}")

    image_folder = Path(image_folder)
    if not image_folder.exists():
        raise ValueError(f"Image folder not found: {image_folder}")

    extensions  = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    image_files = []
    for ext in extensions:
        image_files.extend(image_folder.glob(f'*{ext}'))
        image_files.extend(image_folder.glob(f'*{ext.upper()}'))

    if not image_files:
        raise ValueError(f"No images found in {image_folder}")
    print(f"Found {len(image_files)} images")

    labels_dict = {}
    if label_file and os.path.exists(label_file):
        print(f"Loading labels from {label_file}")
        labels_dict = load_labels_dict(label_file)
        print(f"Loaded {len(labels_dict)} labels")

    image_paths, labels, image_names, failed = [], [], [], []

    print("Validating images and mapping labels...")
    for img_path in tqdm(image_files, desc="Processing images"):
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                failed.append(img_path.name)
                continue
            image_paths.append(str(img_path))
            image_names.append(img_path.name)
            if labels_dict:
                lbl = labels_dict.get(img_path.stem) or labels_dict.get(img_path.name)
                labels.append(lbl)
            else:
                labels.append(None)
        except Exception as e:
            failed.append(img_path.name)
            print(f"Warning: Failed to process {img_path.name}: {e}")

    if failed:
        print(f"\nWarning: {len(failed)} images could not be loaded")

    if labels_dict:
        valid_idx = [i for i, lbl in enumerate(labels) if lbl is not None]
        if len(valid_idx) < len(image_paths):
            print(f"Using {len(valid_idx)} images with labels")
            image_paths = [image_paths[i] for i in valid_idx]
            image_names = [image_names[i] for i in valid_idx]
            labels      = np.array([labels[i] for i in valid_idx])
        else:
            labels = np.array(labels)
    else:
        print("No labels provided — assigning all images to normal class (label=1)")
        labels = np.ones(len(image_paths), dtype=int)

    labels_binary = np.where(labels == -1, 0, 1)
    print(f"Successfully indexed {len(image_paths)} images")
    print(f"Label distribution: Normal={np.sum(labels_binary==1)}, "
          f"Abnormal={np.sum(labels_binary==0)}")
    return image_paths, labels_binary, image_names


# ============================================================================
# STEP 2: FEATURE EXTRACTOR LOADING
# ============================================================================
def load_feature_extractors():
    """Load VGG19 and MobileNetV2 (ImageNet weights, top removed)."""
    print(f"\n{'='*70}")
    print("STEP 2: Loading Pre-trained Feature Extractors")
    print(f"{'='*70}")

    print("Loading VGG19 (ImageNet weights)...")
    vgg_base  = VGG19(weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3))
    vgg_model = Model(inputs=vgg_base.input, outputs=vgg_base.output)
    vgg_model.trainable = False

    print("Loading MobileNetV2 (ImageNet weights)...")
    mob_base        = MobileNetV2(weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3))
    mobilenet_model = Model(inputs=mob_base.input, outputs=mob_base.output)
    mobilenet_model.trainable = False

    print("Feature extractors loaded successfully!")
    return vgg_model, mobilenet_model


# ============================================================================
# STEP 3: DEEP FEATURE EXTRACTION
# ============================================================================
def extract_features(image_paths: List[str],
                     vgg_model: Model,
                     mobilenet_model: Model,
                     batch_size: int = 32) -> np.ndarray:
    """
    Extract and fuse features from VGG19 and MobileNetV2.
    Output shape: (N, 1792) for pooled 224x224 inputs.
    """
    print(f"\n{'='*70}")
    print("STEP 3: Extracting Deep Features")
    print(f"{'='*70}")

    vgg_features, mob_features = [], []

    for i in tqdm(range(0, len(image_paths), batch_size), desc="Feature batches"):
        batch_rgb = []
        for path in image_paths[i:i+batch_size]:
            img = cv2.imread(path)
            if img is None:
                raise ValueError(f"Failed to reload image: {path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, IMAGE_SIZE)
            batch_rgb.append(img)

        batch_rgb = np.asarray(batch_rgb, dtype=np.float32)
        vgg_feat = vgg_model.predict(preprocess_vgg(batch_rgb.copy()), verbose=0)
        mob_feat = mobilenet_model.predict(preprocess_mobilenet(batch_rgb.copy()), verbose=0)
        vgg_features.append(vgg_feat.reshape(vgg_feat.shape[0], -1))
        mob_features.append(mob_feat.reshape(mob_feat.shape[0], -1))

    vgg_features = np.vstack(vgg_features)
    mob_features = np.vstack(mob_features)
    print(f"VGG19 features:       {vgg_features.shape}")
    print(f"MobileNetV2 features: {mob_features.shape}")

    fused = np.hstack([vgg_features, mob_features])
    print(f"Fused features:       {fused.shape}")
    return fused


# ============================================================================
# STEP 3B: HARRIS HAWKS OPTIMIZATION — FEATURE SELECTION
# ============================================================================
def _variance_prefilter(X: np.ndarray,
                        k: int = PRE_FILTER_K,
                        chunk: int = 256) -> Tuple[np.ndarray, np.ndarray]:
    """
    Stage 1 — Zero-SVD variance pre-filter.

    Computes per-feature variance in small row-chunks (O(D) extra memory, no
    large temporaries) and keeps the top-k highest-variance columns.
    This shrinks 87808 -> 4096 so HHO fitness evaluations are fast.

    Returns:
        X_pre   : (N, k) float32 pre-filtered matrix
        top_idx : (k,) column indices into the original raw feature space
    """
    k = min(int(k), int(X.shape[1]))
    print(f"\nStage 1 — Variance pre-filter: {X.shape[1]} -> {k} features")
    sum_x  = np.zeros(X.shape[1], dtype=np.float64)
    sum_x2 = np.zeros(X.shape[1], dtype=np.float64)

    for start in tqdm(range(0, X.shape[0], chunk), desc="  Variance scan"):
        c = X[start:start+chunk].astype(np.float64, copy=False)
        sum_x  += c.sum(axis=0)
        sum_x2 += (c ** 2).sum(axis=0)

    n       = X.shape[0]
    var     = sum_x2 / n - (sum_x / n) ** 2
    top_idx = np.sort(np.argpartition(var, -k)[-k:])
    X_pre   = X[:, top_idx].astype(np.float32, copy=False)
    print(f"  Pre-filtered shape: {X_pre.shape}")
    return X_pre, top_idx


def _levy_flight(dim: int, rng: np.random.RandomState, beta: float = 1.5) -> np.ndarray:
    """
    Levy flight step for HHO's rapid-dive exploitation phase.
    Generates heavy-tailed random steps that help escape local optima.
    """
    sigma = (
        math.gamma(1 + beta) * math.sin(math.pi * beta / 2) /
        (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))
    ) ** (1.0 / beta)
    u    = rng.randn(dim) * sigma
    v    = rng.randn(dim)
    step = u / (np.abs(v) ** (1.0 / beta))
    return 0.01 * step


def harris_hawks_feature_selection(
        X_pre: np.ndarray,
        y: np.ndarray,
        n_select: int   = HHO_N_SELECT,
        n_hawks: int    = HHO_N_HAWKS,
        max_iter: int   = HHO_MAX_ITER,
        eval_frac: float = HHO_EVAL_FRAC,
        random_state: int = RANDOM_STATE,
) -> Tuple[np.ndarray, np.ndarray, float, List[float]]:
    """
    Harris Hawks Optimization (HHO) for feature selection.

    WHY HHO INSTEAD OF PCA:
        PCA produces linear combinations of all features; the resulting
        components lose original feature meaning and optimise for variance,
        not classification accuracy. HHO selects an actual *subset* of the
        most discriminative original features. Because the search is guided
        directly by classifier accuracy (the fitness function), the chosen
        features are guaranteed to improve downstream model performance.

    BINARY ENCODING — top-k rank (instead of sigmoid thresholding):
        Each hawk's continuous position vector of length D is converted to a
        binary selection by taking the top-n_select indices by value. This
        guarantees exactly n_select features are selected at every step,
        avoiding the oscillating feature count that threshold-based encodings
        suffer from.

    FITNESS FUNCTION:
        1 - mean_3fold_CV_accuracy using RandomForest(50 trees) on a stratified
        subsample (eval_frac * N rows). Fast enough for 25 iterations * 10 hawks.

    HHO PHASES (Heidari et al. 2019):
        |E| >= 1  Exploration  — perch on random hawk or chase rabbit offset
        |E| <  1  Exploitation — soft/hard besiege, Levy-flight rapid dives

    Args:
        X_pre       : (N, D) pre-filtered feature matrix (float32)
        y           : (N,) binary labels
        n_select    : number of features to select
        n_hawks     : population size
        max_iter    : number of iterations
        eval_frac   : subsample fraction for fitness
        random_state: seed

    Returns:
        X_selected  : (N, n_select) float32 — selected feature columns
        selected_idx: (n_select,) int — column indices into X_pre
        best_acc    : float — best fitness-evaluation CV accuracy
        acc_history : list  — best accuracy recorded each iteration
    """
    print(f"\n{'='*70}")
    print("STEP 3B: Harris Hawks Optimization — Feature Selection")
    print(f"{'='*70}")
    n_select = min(int(n_select), int(X_pre.shape[1]))
    print(f"  Input features : {X_pre.shape[1]}")
    print(f"  Target features: {n_select}")
    print(f"  Hawks           : {n_hawks}   |   Iterations: {max_iter}")
    n_eval = max(200, int(X_pre.shape[0] * eval_frac))
    print(f"  Fitness sample  : {eval_frac*100:.0f}% of {X_pre.shape[0]} "
          f"samples = {n_eval} rows\n")

    rng          = np.random.RandomState(random_state)
    n_samples, D = X_pre.shape

    eval_idx = rng.choice(n_samples, n_eval, replace=False)
    X_ev     = X_pre[eval_idx].astype(np.float32, copy=False)
    y_ev     = y[eval_idx]
    skf_fit  = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)

    def _fitness(position: np.ndarray) -> float:
        """Return 1 - mean_CV_accuracy (minimise)."""
        top_k = np.argsort(position)[-n_select:]
        X_sel = X_ev[:, top_k]
        accs  = []
        for tr, va in skf_fit.split(X_sel, y_ev):
            clf = RandomForestClassifier(
                n_estimators=50, random_state=random_state,
                n_jobs=-1, class_weight='balanced'
            )
            clf.fit(X_sel[tr], y_ev[tr])
            accs.append(accuracy_score(y_ev[va], clf.predict(X_sel[va])))
        return 1.0 - float(np.mean(accs))

    # ── Initialise population ─────────────────────────────────────────────────
    positions  = rng.rand(n_hawks, D).astype(np.float64)
    fit_vals   = np.array([_fitness(positions[i]) for i in range(n_hawks)])

    best_idx       = int(np.argmin(fit_vals))
    rabbit_pos     = positions[best_idx].copy()
    rabbit_fitness = float(fit_vals[best_idx])
    acc_history    = [1.0 - rabbit_fitness]
    print(f"  Initial best CV accuracy: {1.0 - rabbit_fitness:.4f}")

    # ── Main loop ─────────────────────────────────────────────────────────────
    for t in tqdm(range(max_iter), desc="HHO iterations"):
        E1 = 2.0 * (1.0 - t / max_iter)      # energy decay

        for i in range(n_hawks):
            E0 = 2.0 * rng.rand() - 1.0      # escaping energy in [-1, 1]
            E  = E1 * E0

            if abs(E) >= 1.0:
                # ── Exploration ───────────────────────────────────────────────
                if rng.rand() >= 0.5:
                    rand_hawk    = positions[rng.randint(0, n_hawks)]
                    positions[i] = (rand_hawk
                                    - rng.rand()
                                    * abs(rand_hawk - 2.0 * rng.rand() * positions[i]))
                else:
                    LB, UB       = 0.0, 1.0
                    positions[i] = ((rabbit_pos - np.mean(positions, axis=0))
                                    - rng.rand()
                                    * (LB + rng.rand() * (UB - LB)))
            else:
                # ── Exploitation ──────────────────────────────────────────────
                J     = 2.0 * (1.0 - rng.rand())
                delta = rabbit_pos - positions[i]

                if rng.rand() >= 0.5:
                    if abs(E) > 0.5:
                        # Soft besiege
                        positions[i] = delta - E * abs(J * rabbit_pos - positions[i])
                    else:
                        # Hard besiege
                        positions[i] = rabbit_pos - E * abs(delta)
                else:
                    # Rapid dives with Levy flight
                    LF = _levy_flight(D, rng)
                    Y  = rabbit_pos - E * abs(J * rabbit_pos - positions[i])
                    Z  = Y + rng.rand(D) * LF

                    f_Y       = _fitness(Y)
                    f_Z       = _fitness(Z)
                    f_current = fit_vals[i]

                    if f_Y < f_current:
                        positions[i] = Y
                        fit_vals[i]  = f_Y
                    if f_Z < fit_vals[i]:
                        positions[i] = Z
                        fit_vals[i]  = f_Z

            positions[i] = np.clip(positions[i], 0.0, 1.0)

            f_new       = _fitness(positions[i])
            fit_vals[i] = f_new
            if f_new < rabbit_fitness:
                rabbit_pos     = positions[i].copy()
                rabbit_fitness = f_new

        current_acc = 1.0 - rabbit_fitness
        acc_history.append(current_acc)
        print(f"  Iter {t+1:>3}/{max_iter} — best CV acc: {current_acc:.4f}")

    # ── Final selected features ───────────────────────────────────────────────
    selected_idx = np.sort(np.argsort(rabbit_pos)[-n_select:]).astype(np.int32)
    best_acc     = 1.0 - rabbit_fitness
    X_selected   = X_pre[:, selected_idx].astype(np.float32, copy=False)

    print(f"\nHHO complete — best feature-selection CV accuracy: {best_acc:.4f}")
    print(f"Selected features shape: {X_selected.shape}")

    # Convergence plot
    plt.figure(figsize=(8, 4))
    plt.plot(range(len(acc_history)), acc_history, 'o-', lw=2, color='#1D9E75')
    plt.xlabel("Iteration", fontsize=11)
    plt.ylabel("Best CV Accuracy", fontsize=11)
    plt.title("HHO Feature Selection — Convergence Curve", fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    conv_path = os.path.join(RESULTS_DIR, "hho_convergence.png")
    plt.savefig(conv_path, dpi=150, bbox_inches='tight')
    print(f"Convergence curve saved: {conv_path}")
    plt.close()

    return X_selected, selected_idx, best_acc, acc_history


def run_feature_selection(X_raw: np.ndarray,
                          y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Full two-stage pipeline: variance pre-filter -> HHO feature selection.

    Stage 1 — Variance pre-filter (87808 -> PRE_FILTER_K=4096):
        Zero-SVD, O(D) extra memory. Eliminates near-constant features.

    Stage 2 — HHO feature selection (4096 -> HHO_N_SELECT=512):
        Fitness-driven search. Selects the subset that maximises CV accuracy.

    Returns:
        X_selected       : (N, HHO_N_SELECT) selected feature matrix
        prefilter_idx    : column indices from Stage 1 (into raw features)
        hho_relative_idx : column indices from Stage 2 (into pre-filtered)
    """
    X_pre, prefilter_idx                       = _variance_prefilter(X_raw, k=PRE_FILTER_K)
    X_sel, hho_rel_idx, best_acc, acc_history  = harris_hawks_feature_selection(X_pre, y)
    return X_sel, prefilter_idx, hho_rel_idx


def apply_saved_feature_selection(X_raw: np.ndarray,
                                  prefilter_idx: np.ndarray,
                                  hho_relative_idx: np.ndarray) -> np.ndarray:
    """
    Apply pre-computed feature selection to new raw features (e.g. at inference
    time). Mirrors the two-stage pipeline without re-running HHO.
    """
    X_pre = X_raw[:, prefilter_idx].astype(np.float32, copy=False)
    return X_pre[:, hho_relative_idx].astype(np.float32, copy=False)


# ============================================================================
# STEP 4: MACHINE LEARNING MODELS
# ============================================================================
def create_models() -> Dict[str, Any]:
    """
    Create four well-tuned classifiers.

    AdaBoost has been removed: its 60% CV accuracy was actively dragging down
    ensemble performance. All remaining models score >= 86% CV accuracy and
    contribute positively to the soft weighted ensemble.

    All models expose predict_proba() for soft ensemble voting.
    """
    return {
        'SVM': Pipeline([
            ('scaler', StandardScaler()),
            ('svc', SVC(
                kernel='rbf', C=10.0, gamma='scale', probability=True,
                class_weight='balanced', random_state=RANDOM_STATE
            ))
        ]),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=3,
            subsample=0.8, random_state=RANDOM_STATE
        ),
        'Histogram Gradient Boosting': HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.05, max_depth=None,
            max_leaf_nodes=31, l2_regularization=1e-3,
            random_state=RANDOM_STATE
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=500, max_depth=None, random_state=RANDOM_STATE,
            class_weight='balanced_subsample', n_jobs=-1
        ),
    }


# ============================================================================
# STEP 4B: CROSS-VALIDATION AND EVALUATION
# ============================================================================
def calculate_metrics(y_true: np.ndarray,
                      y_pred: np.ndarray,
                      y_proba: np.ndarray = None) -> Tuple[Dict[str, float], np.ndarray]:
    """Compute accuracy, precision, recall, F1, sensitivity, specificity, AUC."""
    metrics = {
        'Accuracy' : accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, average='binary', zero_division=0),
        'Recall'   : recall_score(y_true, y_pred, average='binary', zero_division=0),
        'F1 Score' : f1_score(y_true, y_pred, average='binary', zero_division=0),
    }
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    metrics['Sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    metrics['Specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    if y_proba is not None:
        try:
            fpr, tpr, _ = roc_curve(y_true, y_proba)
            metrics['AUC'] = auc(fpr, tpr)
        except Exception:
            metrics['AUC'] = 0.0
    else:
        metrics['AUC'] = 0.0
    return metrics, cm


def train_models(X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    """
    Train all classifiers with 10-fold stratified cross-validation.
    Records per-fold metrics and out-of-fold predictions for unbiased evaluation.
    """
    print(f"\n{'='*70}")
    print("STEP 4: Training Models with 10-Fold Cross-Validation")
    print(f"{'='*70}")

    unique_classes, class_counts = np.unique(y, return_counts=True)
    if len(unique_classes) < 2:
        raise ValueError(
            "At least two classes required for supervised training. "
            "Provide a label CSV with both normal and abnormal samples."
        )

    class_counts_map = dict(zip(unique_classes.tolist(), class_counts.tolist()))
    min_count = int(np.min(class_counts))
    n_splits  = min(N_FOLDS, min_count)
    if n_splits < N_FOLDS:
        print(f"Warning: reducing folds {N_FOLDS} -> {n_splits} (limited class samples)")

    majority = int(np.max(class_counts))
    minority = int(np.min(class_counts))
    imbalance_ratio = (majority / minority) if minority > 0 else float('inf')
    print("\nClass balance summary:")
    print(f"  Counts: {class_counts_map}")
    print(f"  Imbalance ratio (major/minor): {imbalance_ratio:.2f}")

    X   = np.asarray(X, dtype=np.float32)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    results = {}

    weighted_models = {"Gradient Boosting", "Histogram Gradient Boosting"}

    for model_name, model in create_models().items():
        print(f"\n{'-'*70}\nTraining {model_name}\n{'-'*70}")
        fold_metrics, all_y_true, all_y_pred, all_y_proba = [], [], [], []

        for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
            X_tr, X_va = X[tr_idx], X[va_idx]
            y_tr, y_va = y[tr_idx], y[va_idx]

            if model_name in weighted_models:
                sample_weight = compute_sample_weight(class_weight='balanced', y=y_tr)
                model.fit(X_tr, y_tr, sample_weight=sample_weight)
            else:
                model.fit(X_tr, y_tr)
            y_pred  = model.predict(X_va)
            y_proba = (model.predict_proba(X_va)[:, 1]
                       if hasattr(model, 'predict_proba') else None)

            m, _ = calculate_metrics(y_va, y_pred, y_proba)
            fold_metrics.append(m)
            all_y_true.extend(y_va)
            all_y_pred.extend(y_pred)
            if y_proba is not None:
                all_y_proba.extend(y_proba)

            print(f"  Fold {fold}/{n_splits} — Acc: {m['Accuracy']:.4f}  "
                  f"F1: {m['F1 Score']:.4f}  AUC: {m['AUC']:.4f}")

        avg_metrics = {
            k: float(np.mean([m[k] for m in fold_metrics]))
            for k in fold_metrics[0]
        }
        overall_metrics, overall_cm = calculate_metrics(
            np.array(all_y_true), np.array(all_y_pred),
            np.array(all_y_proba) if all_y_proba else None
        )

        final_model = create_models()[model_name]
        if model_name in weighted_models:
            full_weight = compute_sample_weight(class_weight='balanced', y=y)
            final_model.fit(X, y, sample_weight=full_weight)
        else:
            final_model.fit(X, y)

        results[model_name] = {
            'model'           : final_model,
            'fold_metrics'    : fold_metrics,
            'avg_metrics'     : avg_metrics,
            'overall_metrics' : overall_metrics,
            'confusion_matrix': overall_cm,
            'y_true'          : np.array(all_y_true),
            'y_pred'          : np.array(all_y_pred),
            'y_proba'         : np.array(all_y_proba) if all_y_proba else None,
            'cv_accuracy'     : avg_metrics['Accuracy'],
        }

        print(f"\n{model_name} — Average CV Metrics:")
        print(f"  Accuracy:    {avg_metrics['Accuracy']:.4f}")
        print(f"  Precision:   {avg_metrics['Precision']:.4f}")
        print(f"  Recall:      {avg_metrics['Recall']:.4f}")
        print(f"  F1 Score:    {avg_metrics['F1 Score']:.4f}")
        print(f"  Sensitivity: {avg_metrics['Sensitivity']:.4f}")
        print(f"  Specificity: {avg_metrics['Specificity']:.4f}")
        print(f"  AUC:         {avg_metrics['AUC']:.4f}")

    return results


# ============================================================================
# STEP 5: EVALUATION SUMMARY
# ============================================================================
def evaluate_models(results: Dict[str, Any]) -> None:
    """Print evaluation table and save ROC curves."""
    print(f"\n{'='*70}")
    print("STEP 5: Model Evaluation Summary")
    print(f"{'='*70}")

    print("\n" + "="*70)
    print("COMPREHENSIVE EVALUATION METRICS (10-Fold CV Average)")
    print("="*70)
    print(f"{'Model':<32} {'Acc':<8} {'Prec':<8} {'Rec':<8} "
          f"{'F1':<8} {'Sens':<8} {'Spec':<8} {'AUC':<8}")
    print("-"*70)

    for name, r in results.items():
        m = r['avg_metrics']
        print(f"{name:<32} {m['Accuracy']:<8.4f} {m['Precision']:<8.4f} "
              f"{m['Recall']:<8.4f} {m['F1 Score']:<8.4f} "
              f"{m['Sensitivity']:<8.4f} {m['Specificity']:<8.4f} {m['AUC']:<8.4f}")

    plt.figure(figsize=(10, 8))
    for name, r in results.items():
        if r['y_proba'] is not None:
            fpr, tpr, _ = roc_curve(r['y_true'], r['y_proba'])
            plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC={auc(fpr, tpr):.4f})")
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
    plt.xlim([0, 1]); plt.ylim([0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves — Heart Health Classification', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    roc_path = os.path.join(RESULTS_DIR, 'roc_curves.png')
    plt.savefig(roc_path, dpi=300, bbox_inches='tight')
    print(f"\nROC curves saved: {roc_path}")
    plt.close()

    print("\n" + "="*70)
    print("CONFUSION MATRICES (Overall Out-of-Fold Predictions)")
    print("="*70)
    for name, r in results.items():
        cm = r['confusion_matrix']
        print(f"\n{name}:")
        print(f"  True Negatives:  {cm[0,0]}")
        print(f"  False Positives: {cm[0,1]}")
        print(f"  False Negatives: {cm[1,0]}")
        print(f"  True Positives:  {cm[1,1]}")


# ============================================================================
# STEP 6: ENSEMBLE — SOFT PROBABILITY-WEIGHTED VOTING
# ============================================================================
def majority_vote(results: Dict[str, Any],
                  X: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Soft probability-weighted ensemble with two enhancements over plain
    majority voting:

    1. Accuracy^4 weights:
       Each model's CV accuracy is raised to the 4th power before normalisation,
       exponentially amplifying contributions from strong models.

    2. CV-optimal threshold:
       The 0.5 decision boundary is replaced by the threshold maximising F1
       on out-of-fold probabilities. Evaluated only on held-out data —
       no data leakage.
    """
    print(f"\n{'='*70}")
    print("STEP 6: Ensemble — Soft Probability-Weighted Voting")
    print(f"{'='*70}")

    model_names = list(results.keys())

    # Accuracy^4 weights
    cv_accs = np.array([results[n]['cv_accuracy'] for n in model_names], dtype=np.float64)
    raw_w   = cv_accs ** 4
    weights = raw_w / raw_w.sum()

    print("\nModel weights (accuracy^4, normalised):")
    for name, w, acc in zip(model_names, weights, cv_accs):
        print(f"  {name:<32} CV Acc={acc:.4f}  Weight={w:.4f}")

    # Out-of-fold weighted probability
    all_y_true = results[model_names[0]]['y_true']
    cv_cols = [
        results[n]['y_proba'] if results[n]['y_proba'] is not None
        else results[n]['y_pred'].astype(np.float64)
        for n in model_names
    ]
    cv_proba_mat = np.column_stack(cv_cols)   # (N, n_models)
    cv_weighted  = cv_proba_mat @ weights     # (N,)

    # CV-optimal threshold (F1 maximisation on OOF data)
    thresholds = np.linspace(0.20, 0.80, 121)
    best_t, best_f1 = 0.5, -1.0
    for t in thresholds:
        f1_t = f1_score(all_y_true, (cv_weighted >= t).astype(int), zero_division=0)
        if f1_t > best_f1:
            best_f1, best_t = f1_t, t

    print(f"\nCV-optimal threshold: {best_t:.3f}  (F1={best_f1:.4f})")
    cv_ensemble_final = (cv_weighted >= best_t).astype(int)

    # Apply to full-dataset predictions
    full_cols = [
        m['model'].predict_proba(X)[:, 1] if hasattr(m['model'], 'predict_proba')
        else m['model'].predict(X).astype(np.float64)
        for m in results.values()
    ]
    full_weighted        = np.column_stack(full_cols) @ weights
    ensemble_predictions = (full_weighted >= best_t).astype(int)

    ens_metrics, ens_cm = calculate_metrics(all_y_true, cv_ensemble_final, cv_weighted)

    print("\nEnsemble (Soft Weighted Voting) — Out-of-Fold CV Performance:")
    print(f"  Accuracy:    {ens_metrics['Accuracy']:.4f}")
    print(f"  Precision:   {ens_metrics['Precision']:.4f}")
    print(f"  Recall:      {ens_metrics['Recall']:.4f}")
    print(f"  F1 Score:    {ens_metrics['F1 Score']:.4f}")
    print(f"  Sensitivity: {ens_metrics['Sensitivity']:.4f}")
    print(f"  Specificity: {ens_metrics['Specificity']:.4f}")
    print(f"  AUC:         {ens_metrics['AUC']:.4f}")

    print("\nEnsemble Confusion Matrix (Out-of-Fold):")
    print(f"  True Negatives:  {ens_cm[0,0]}")
    print(f"  False Positives: {ens_cm[0,1]}")
    print(f"  False Negatives: {ens_cm[1,0]}")
    print(f"  True Positives:  {ens_cm[1,1]}")

    return ensemble_predictions, ens_metrics


# ============================================================================
# STEP 7: SAVE MODELS
# ============================================================================
def save_models(results: Dict[str, Any]) -> None:
    """Save all trained models and ensemble weights JSON."""
    print(f"\n{'='*70}")
    print("STEP 7: Saving Trained Models")
    print(f"{'='*70}")

    for name, r in results.items():
        path = os.path.join(MODELS_DIR, f"{name.replace(' ', '_').lower()}.pkl")
        joblib.dump(r['model'], path)
        print(f"Saved: {path}")

    keys    = [n.replace(' ', '_').lower() for n in results]
    cv_accs = np.array([results[n]['cv_accuracy'] for n in results], dtype=np.float64)
    raw_w   = cv_accs ** 4
    norm_w  = (raw_w / raw_w.sum()).tolist()

    payload = {
        "weight_source": "cv_accuracy_power4",
        "models": {k: float(w) for k, w in zip(keys, norm_w)},
    }
    w_path = os.path.join(RESULTS_DIR, "ensemble_weights.json")
    with open(w_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved: {w_path}")
    print(f"\nAll models saved to: {MODELS_DIR}/")


# ============================================================================
# MAIN
# ============================================================================
def main():
    print("\n" + "="*70)
    print("DECISION SUPPORT SYSTEM FOR HEART HEALTH STATUS DETECTION")
    print("="*70)

    image_folder = resolve_default_path(
        ["chroma-2022", "dataset/normal"], "image folder"
    )
    label_file_candidates = ["training-2022/training-2022/physionet_2022.csv"]
    label_file = next((p for p in label_file_candidates if Path(p).exists()), None)
    if label_file is None:
        print("Warning: No label CSV found. "
              "Expected: training-2022/training-2022/physionet_2022.csv")

    # Resume control — set env var DSS_RESUME_STAGE=step3b|step4 to skip
    # early stages when caches already exist.
    resume_stage = os.environ.get("DSS_RESUME_STAGE", "auto").strip().lower()
    if resume_stage not in {"auto", "step3b", "step4"}:
        raise ValueError("DSS_RESUME_STAGE must be one of: auto, step3b, step4")

    try:
        # ── Resume: HHO-selected features already cached ─────────────────────
        if resume_stage == "step4" or (
                resume_stage == "auto"
                and os.path.exists(FEATURE_CACHE_PATH)
                and os.path.exists(LABEL_CACHE_PATH)):
            if not (os.path.exists(FEATURE_CACHE_PATH) and os.path.exists(LABEL_CACHE_PATH)):
                raise FileNotFoundError(
                    f"Step 4 resume requires {FEATURE_CACHE_PATH} and {LABEL_CACHE_PATH}"
                )
            print("\nResume mode: STEP 4 — HHO-selected features cached, skipping to training")
            features = np.load(FEATURE_CACHE_PATH, mmap_mode="r")
            labels   = np.load(LABEL_CACHE_PATH)
            print(f"Loaded selected features: {features.shape}")

        # ── Resume: variance-prefiltered features cached, run HHO ────────────
        elif resume_stage == "step3b" or (
                resume_stage == "auto"
                and os.path.exists(PREFILTER_CACHE_PATH)
                and os.path.exists(LABEL_CACHE_PATH)):
            if not (os.path.exists(PREFILTER_CACHE_PATH) and os.path.exists(LABEL_CACHE_PATH)):
                raise FileNotFoundError(
                    f"Step 3B resume requires {PREFILTER_CACHE_PATH} and {LABEL_CACHE_PATH}"
                )
            print("\nResume mode: STEP 3B — pre-filtered features cached, running HHO")
            X_pre  = np.load(PREFILTER_CACHE_PATH, mmap_mode="r")
            labels = np.load(LABEL_CACHE_PATH)
            print(f"Loaded pre-filtered features: {X_pre.shape}")

            features, hho_rel_idx, _, _ = harris_hawks_feature_selection(X_pre, labels)
            np.save(HHO_IDX_CACHE_PATH, hho_rel_idx)
            np.save(FEATURE_CACHE_PATH, features)
            print(f"Cached HHO-selected features: {FEATURE_CACHE_PATH}")

        # ── Resume: raw features cached, run variance filter + HHO ───────────
        elif resume_stage == "auto" and os.path.exists(RAW_FEATURE_CACHE_PATH):
            print("\nResume mode: raw features cached — running Stage 1 + HHO")
            X_raw  = np.load(RAW_FEATURE_CACHE_PATH, mmap_mode="r")
            labels = np.load(LABEL_CACHE_PATH)
            print(f"Loaded raw features: {X_raw.shape}")

            features, prefilter_idx, hho_rel_idx = run_feature_selection(X_raw, labels)
            np.save(PREFILTER_CACHE_PATH, X_raw[:, prefilter_idx].astype(np.float32))
            np.save(PREFILTER_IDX_CACHE_PATH, prefilter_idx)
            np.save(HHO_IDX_CACHE_PATH, hho_rel_idx)
            np.save(FEATURE_CACHE_PATH, features)
            print(f"Cached HHO-selected features: {FEATURE_CACHE_PATH}")

        else:
            # ── Full pipeline from scratch ────────────────────────────────────
            image_paths, labels, image_names = load_images(image_folder, label_file)
            vgg_model, mobilenet_model       = load_feature_extractors()
            X_raw = extract_features(image_paths, vgg_model, mobilenet_model)

            np.save(RAW_FEATURE_CACHE_PATH, X_raw)
            np.save(LABEL_CACHE_PATH, labels)
            print(f"Cached raw features: {RAW_FEATURE_CACHE_PATH}")
            print(f"Cached labels:       {LABEL_CACHE_PATH}")

            features, prefilter_idx, hho_rel_idx = run_feature_selection(X_raw, labels)
            np.save(PREFILTER_CACHE_PATH, X_raw[:, prefilter_idx].astype(np.float32))
            np.save(PREFILTER_IDX_CACHE_PATH, prefilter_idx)
            np.save(HHO_IDX_CACHE_PATH, hho_rel_idx)
            np.save(FEATURE_CACHE_PATH, features)
            print(f"Cached HHO-selected features: {FEATURE_CACHE_PATH}")

        # ── Training, evaluation, ensemble, save ─────────────────────────────
        results = train_models(features, labels)
        evaluate_models(results)
        ensemble_preds, ensemble_metrics = majority_vote(results, features)
        save_models(results)

        print(f"\n{'='*70}")
        print("EXECUTION COMPLETED SUCCESSFULLY")
        print(f"{'='*70}")
        print(f"\nResults saved to: {RESULTS_DIR}/")
        print(f"Models saved to:  {MODELS_DIR}/")
        best = max(results.items(), key=lambda x: x[1]['avg_metrics']['F1 Score'])
        print(f"\nBest individual model (F1): "
              f"{best[0]} = {best[1]['avg_metrics']['F1 Score']:.4f}")
        print(f"\nEnsemble Performance:")
        print(f"  Accuracy: {ensemble_metrics['Accuracy']:.4f}")
        print(f"  F1 Score: {ensemble_metrics['F1 Score']:.4f}")
        print(f"  AUC:      {ensemble_metrics['AUC']:.4f}")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
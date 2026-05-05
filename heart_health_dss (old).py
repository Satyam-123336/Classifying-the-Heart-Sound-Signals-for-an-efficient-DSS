"""
================================================================================
Decision Support System for Autonomous Detection of Heart Health Status
================================================================================
Research-level implementation using:
- VGG19 and MobileNetV2 for deep feature extraction
- Multiple ML classifiers (SVM, Gradient Boosting, Histogram Gradient Boosting,
  Random Forest, AdaBoost)
- 10-Fold Cross-Validation
- Ensemble Majority Voting
- Comprehensive evaluation metrics
================================================================================
"""

import numpy as np
import os
import cv2
import pandas as pd
import warnings
from pathlib import Path
from typing import Tuple, List, Dict, Any
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    AdaBoostClassifier
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc, confusion_matrix, classification_report
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
IMAGE_SIZE = (224, 224)
N_FOLDS = 10
RANDOM_STATE = 42
MODELS_DIR = "saved_models-20260411T165838Z-3-001/saved_models"
RESULTS_DIR = "results"

# Create directories if they don't exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================================
# 1. IMAGE LOADING AND PREPROCESSING
# ============================================================================
def load_images(image_folder: str, label_file: str = None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load images from folder and optionally load labels from CSV file.
    
    Args:
        image_folder: Path to folder containing images
        label_file: Optional path to CSV file with labels (format: image_name,label)
    
    Returns:
        images: Array of preprocessed images (N, 224, 224, 3)
        labels: Array of labels (N,)
        image_names: List of image filenames
    """
    print(f"\n{'='*70}")
    print("STEP 1: Loading and Preprocessing Images")
    print(f"{'='*70}")
    
    image_folder = Path(image_folder)
    if not image_folder.exists():
        raise ValueError(f"Image folder not found: {image_folder}")
    
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(image_folder.glob(f'*{ext}')))
        image_files.extend(list(image_folder.glob(f'*{ext.upper()}')))
    
    if len(image_files) == 0:
        raise ValueError(f"No images found in {image_folder}")
    
    print(f"Found {len(image_files)} images")
    
    # Load labels if provided
    labels_dict = {}
    if label_file and os.path.exists(label_file):
        print(f"Loading labels from {label_file}")
        df = pd.read_csv(label_file, header=None, names=['image_name', 'label'])
        labels_dict = dict(zip(df['image_name'], df['label']))
        print(f"Loaded {len(labels_dict)} labels")
    
    # Load and preprocess images
    images = []
    labels = []
    image_names = []
    failed_images = []
    
    print("Loading and preprocessing images...")
    for img_path in tqdm(image_files, desc="Processing images"):
        try:
            # Read image
            img = cv2.imread(str(img_path))
            if img is None:
                failed_images.append(img_path.name)
                continue
            
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize to 224x224
            img = cv2.resize(img, IMAGE_SIZE)
            
            # Normalize to [0, 1] range
            img = img.astype(np.float32) / 255.0
            
            images.append(img)
            image_names.append(img_path.name)
            
            # Get label if available
            if labels_dict:
                # Try to match image name (with or without extension)
                img_name_no_ext = img_path.stem
                label = labels_dict.get(img_name_no_ext, None)
                if label is None:
                    # Try with full name
                    label = labels_dict.get(img_path.name, None)
                if label is not None:
                    labels.append(label)
                else:
                    labels.append(None)
            else:
                labels.append(None)
                
        except Exception as e:
            failed_images.append(img_path.name)
            print(f"Warning: Failed to process {img_path.name}: {str(e)}")
            continue
    
    if failed_images:
        print(f"\nWarning: {len(failed_images)} images could not be loaded")
    
    images = np.array(images)
    
    # Handle labels
    if labels_dict:
        # Filter out images without labels
        valid_indices = [i for i, lbl in enumerate(labels) if lbl is not None]
        if len(valid_indices) < len(images):
            print(f"Using {len(valid_indices)} images with labels")
            images = images[valid_indices]
            labels = np.array([labels[i] for i in valid_indices])
            image_names = [image_names[i] for i in valid_indices]
        else:
            labels = np.array(labels)
    else:
        # If no labels provided, create dummy labels (all normal/1)
        print("No labels provided. Using all images as normal class (label=1)")
        labels = np.ones(len(images), dtype=int)
    
    # Convert labels to binary: -1 -> 0 (abnormal), 1 -> 1 (normal)
    # For binary classification: 0 = abnormal, 1 = normal
    labels_binary = np.where(labels == -1, 0, 1)
    
    print(f"Successfully loaded {len(images)} images")
    print(f"Image shape: {images.shape}")
    print(f"Label distribution: Normal={np.sum(labels_binary==1)}, Abnormal={np.sum(labels_binary==0)}")
    
    return images, labels_binary, image_names


# ============================================================================
# 2. FEATURE EXTRACTION
# ============================================================================
def load_feature_extractors():
    """
    Load VGG19 and MobileNetV2 models for feature extraction.
    
    Returns:
        vgg_model: VGG19 model without top layer
        mobilenet_model: MobileNetV2 model without top layer
    """
    print(f"\n{'='*70}")
    print("STEP 2: Loading Pre-trained Feature Extractors")
    print(f"{'='*70}")
    
    print("Loading VGG19 (ImageNet weights)...")
    vgg_base = VGG19(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    vgg_model = Model(inputs=vgg_base.input, outputs=vgg_base.output)
    vgg_model.trainable = False
    
    print("Loading MobileNetV2 (ImageNet weights)...")
    mobilenet_base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    mobilenet_model = Model(inputs=mobilenet_base.input, outputs=mobilenet_base.output)
    mobilenet_model.trainable = False
    
    print("Feature extractors loaded successfully!")
    return vgg_model, mobilenet_model


def extract_features(images: np.ndarray, vgg_model: Model, mobilenet_model: Model,
                     batch_size: int = 32) -> np.ndarray:
    """
    Extract deep features using VGG19 and MobileNetV2, then fuse them.
    
    Args:
        images: Preprocessed images (N, 224, 224, 3)
        vgg_model: VGG19 model
        mobilenet_model: MobileNetV2 model
        batch_size: Batch size for feature extraction
    
    Returns:
        fused_features: Concatenated feature vectors (N, feature_dim)
    """
    print(f"\n{'='*70}")
    print("STEP 3: Extracting Deep Features")
    print(f"{'='*70}")
    
    n_images = len(images)
    
    # Prepare images for VGG19 (BGR format expected)
    images_vgg = images.copy()
    images_vgg = (images_vgg * 255.0).astype(np.uint8)
    images_vgg = np.array([cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in images_vgg])
    images_vgg = preprocess_vgg(images_vgg)
    
    # Prepare images for MobileNetV2
    images_mobilenet = images.copy()
    images_mobilenet = (images_mobilenet * 255.0).astype(np.uint8)
    images_mobilenet = preprocess_mobilenet(images_mobilenet)
    
    # Extract VGG19 features
    print("Extracting VGG19 features...")
    vgg_features = []
    for i in tqdm(range(0, n_images, batch_size), desc="VGG19"):
        batch = images_vgg[i:i+batch_size]
        feat = vgg_model.predict(batch, verbose=0)
        feat_flat = feat.reshape(feat.shape[0], -1)
        vgg_features.append(feat_flat)
    vgg_features = np.vstack(vgg_features)
    
    # Extract MobileNetV2 features
    print("Extracting MobileNetV2 features...")
    mobilenet_features = []
    for i in tqdm(range(0, n_images, batch_size), desc="MobileNetV2"):
        batch = images_mobilenet[i:i+batch_size]
        feat = mobilenet_model.predict(batch, verbose=0)
        feat_flat = feat.reshape(feat.shape[0], -1)
        mobilenet_features.append(feat_flat)
    mobilenet_features = np.vstack(mobilenet_features)
    
    print(f"VGG19 features shape: {vgg_features.shape}")
    print(f"MobileNetV2 features shape: {mobilenet_features.shape}")
    
    # Feature fusion: Concatenate features
    print("Fusing features...")
    fused_features = np.hstack([vgg_features, mobilenet_features])
    print(f"Fused features shape: {fused_features.shape}")
    
    return fused_features


# ============================================================================
# 3. MACHINE LEARNING MODELS
# ============================================================================
def create_models() -> Dict[str, Any]:
    """
    Create and return dictionary of ML models.
    
    Returns:
        Dictionary of model name -> model instance
    """
    models = {
        'SVM': SVC(kernel='rbf', probability=True, random_state=RANDOM_STATE),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=5,
            random_state=RANDOM_STATE
        ),
        'Histogram Gradient Boosting': HistGradientBoostingClassifier(
            max_iter=100, learning_rate=0.1, max_depth=5,
            random_state=RANDOM_STATE
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        'AdaBoost': AdaBoostClassifier(
            n_estimators=50, learning_rate=1.0, random_state=RANDOM_STATE
        )
    }
    return models


# ============================================================================
# 4. CROSS-VALIDATION AND EVALUATION
# ============================================================================
def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                     y_proba: np.ndarray = None) -> Dict[str, float]:
    """
    Calculate comprehensive evaluation metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (for ROC)
    
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Basic metrics
    metrics['Accuracy'] = accuracy_score(y_true, y_pred)
    metrics['Precision'] = precision_score(y_true, y_pred, average='binary', zero_division=0)
    metrics['Recall'] = recall_score(y_true, y_pred, average='binary', zero_division=0)
    metrics['F1 Score'] = f1_score(y_true, y_pred, average='binary', zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Sensitivity (Recall) and Specificity
    metrics['Sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    metrics['Specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    # ROC AUC if probabilities available
    if y_proba is not None:
        try:
            fpr, tpr, _ = roc_curve(y_true, y_proba)
            metrics['AUC'] = auc(fpr, tpr)
        except:
            metrics['AUC'] = 0.0
    else:
        metrics['AUC'] = 0.0
    
    return metrics, cm


def train_models(X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    """
    Train all models using 10-fold cross-validation.
    
    Args:
        X: Feature matrix
        y: Labels
    
    Returns:
        Dictionary containing trained models, metrics, and predictions
    """
    print(f"\n{'='*70}")
    print("STEP 4: Training Models with 10-Fold Cross-Validation")
    print(f"{'='*70}")
    
    models = create_models()
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    
    results = {}
    
    for model_name, model in models.items():
        print(f"\n{'-'*70}")
        print(f"Training {model_name}")
        print(f"{'-'*70}")
        
        fold_metrics = []
        all_y_true = []
        all_y_pred = []
        all_y_proba = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Train model
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_val)
            y_proba = model.predict_proba(X_val)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics
            metrics, cm = calculate_metrics(y_val, y_pred, y_proba)
            fold_metrics.append(metrics)
            
            all_y_true.extend(y_val)
            all_y_pred.extend(y_pred)
            if y_proba is not None:
                all_y_proba.extend(y_proba)
            
            print(f"Fold {fold}/{N_FOLDS} - "
                  f"Accuracy: {metrics['Accuracy']:.4f}, "
                  f"F1: {metrics['F1 Score']:.4f}, "
                  f"AUC: {metrics['AUC']:.4f}")
        
        # Calculate average metrics
        avg_metrics = {}
        for metric_name in fold_metrics[0].keys():
            avg_metrics[metric_name] = np.mean([m[metric_name] for m in fold_metrics])
        
        # Overall metrics on all predictions
        overall_metrics, overall_cm = calculate_metrics(
            np.array(all_y_true), np.array(all_y_pred),
            np.array(all_y_proba) if all_y_proba else None
        )
        
        # Train final model on all data
        final_model = create_models()[model_name]
        final_model.fit(X, y)
        
        results[model_name] = {
            'model': final_model,
            'fold_metrics': fold_metrics,
            'avg_metrics': avg_metrics,
            'overall_metrics': overall_metrics,
            'confusion_matrix': overall_cm,
            'y_true': np.array(all_y_true),
            'y_pred': np.array(all_y_pred),
            'y_proba': np.array(all_y_proba) if all_y_proba else None
        }
        
        # Print summary
        print(f"\n{model_name} - Average Metrics (10-Fold CV):")
        print(f"  Accuracy:  {avg_metrics['Accuracy']:.4f}")
        print(f"  Precision: {avg_metrics['Precision']:.4f}")
        print(f"  Recall:    {avg_metrics['Recall']:.4f}")
        print(f"  F1 Score:  {avg_metrics['F1 Score']:.4f}")
        print(f"  Sensitivity: {avg_metrics['Sensitivity']:.4f}")
        print(f"  Specificity: {avg_metrics['Specificity']:.4f}")
        print(f"  AUC:       {avg_metrics['AUC']:.4f}")
    
    return results


def evaluate_models(results: Dict[str, Any]) -> None:
    """
    Print comprehensive evaluation results and plot ROC curves.
    
    Args:
        results: Dictionary containing model results from train_models()
    """
    print(f"\n{'='*70}")
    print("STEP 5: Model Evaluation Summary")
    print(f"{'='*70}")
    
    # Print detailed results table
    print("\n" + "="*70)
    print("COMPREHENSIVE EVALUATION METRICS (10-Fold Cross-Validation Average)")
    print("="*70)
    print(f"{'Model':<30} {'Acc':<8} {'Prec':<8} {'Rec':<8} {'F1':<8} {'Sens':<8} {'Spec':<8} {'AUC':<8}")
    print("-"*70)
    
    for model_name, result in results.items():
        m = result['avg_metrics']
        print(f"{model_name:<30} {m['Accuracy']:<8.4f} {m['Precision']:<8.4f} "
              f"{m['Recall']:<8.4f} {m['F1 Score']:<8.4f} {m['Sensitivity']:<8.4f} "
              f"{m['Specificity']:<8.4f} {m['AUC']:<8.4f}")
    
    # Plot ROC curves
    print("\nGenerating ROC curves...")
    plt.figure(figsize=(10, 8))
    
    for model_name, result in results.items():
        if result['y_proba'] is not None:
            fpr, tpr, _ = roc_curve(result['y_true'], result['y_proba'])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {roc_auc:.4f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - Heart Health Classification', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    roc_path = os.path.join(RESULTS_DIR, 'roc_curves.png')
    plt.savefig(roc_path, dpi=300, bbox_inches='tight')
    print(f"ROC curves saved to: {roc_path}")
    plt.close()
    
    # Print confusion matrices
    print("\n" + "="*70)
    print("CONFUSION MATRICES (Overall Predictions)")
    print("="*70)
    for model_name, result in results.items():
        cm = result['confusion_matrix']
        print(f"\n{model_name}:")
        print(f"  True Negatives:  {cm[0,0]}")
        print(f"  False Positives: {cm[0,1]}")
        print(f"  False Negatives: {cm[1,0]}")
        print(f"  True Positives:  {cm[1,1]}")


# ============================================================================
# 5. ENSEMBLE - MAJORITY VOTING
# ============================================================================
def majority_vote(results: Dict[str, Any], X: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Implement majority voting ensemble across all classifiers.
    
    Args:
        results: Dictionary containing trained models
        X: Feature matrix for prediction
    
    Returns:
        ensemble_predictions: Final predictions from majority voting
        ensemble_metrics: Metrics for ensemble predictions
    """
    print(f"\n{'='*70}")
    print("STEP 6: Ensemble Majority Voting")
    print(f"{'='*70}")
    
    # Get predictions from all models
    all_predictions = []
    model_names = []
    
    for model_name, result in results.items():
        model = result['model']
        predictions = model.predict(X)
        all_predictions.append(predictions)
        model_names.append(model_name)
    
    all_predictions = np.array(all_predictions).T  # Shape: (n_samples, n_models)
    
    # Majority voting
    ensemble_predictions = []
    for pred_row in all_predictions:
        # Count votes for each class
        votes = np.bincount(pred_row.astype(int))
        # Get class with most votes
        majority_class = np.argmax(votes)
        ensemble_predictions.append(majority_class)
    
    ensemble_predictions = np.array(ensemble_predictions)
    
    # Calculate metrics (using true labels from cross-validation)
    # For ensemble, we'll use the overall predictions from cross-validation
    # by combining predictions from all folds
    all_y_true = results[list(results.keys())[0]]['y_true']
    
    # Re-create ensemble predictions from CV results
    cv_ensemble_preds = []
    for model_name in model_names:
        cv_ensemble_preds.append(results[model_name]['y_pred'])
    cv_ensemble_preds = np.array(cv_ensemble_preds).T
    
    cv_ensemble_final = []
    for pred_row in cv_ensemble_preds:
        votes = np.bincount(pred_row.astype(int))
        majority_class = np.argmax(votes)
        cv_ensemble_final.append(majority_class)
    cv_ensemble_final = np.array(cv_ensemble_final)
    
    ensemble_metrics, ensemble_cm = calculate_metrics(all_y_true, cv_ensemble_final)
    
    print("\nEnsemble (Majority Voting) Performance:")
    print(f"  Accuracy:  {ensemble_metrics['Accuracy']:.4f}")
    print(f"  Precision: {ensemble_metrics['Precision']:.4f}")
    print(f"  Recall:    {ensemble_metrics['Recall']:.4f}")
    print(f"  F1 Score:  {ensemble_metrics['F1 Score']:.4f}")
    print(f"  Sensitivity: {ensemble_metrics['Sensitivity']:.4f}")
    print(f"  Specificity: {ensemble_metrics['Specificity']:.4f}")
    
    return ensemble_predictions, ensemble_metrics


def save_models(results: Dict[str, Any]) -> None:
    """
    Save all trained models using joblib.
    
    Args:
        results: Dictionary containing trained models
    """
    print(f"\n{'='*70}")
    print("STEP 7: Saving Trained Models")
    print(f"{'='*70}")
    
    for model_name, result in results.items():
        model = result['model']
        filename = os.path.join(MODELS_DIR, f"{model_name.replace(' ', '_').lower()}.pkl")
        joblib.dump(model, filename)
        print(f"Saved: {filename}")
    
    print(f"\nAll models saved to: {MODELS_DIR}/")


# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    """
    Main execution function.
    """
    print("\n" + "="*70)
    print("DECISION SUPPORT SYSTEM FOR HEART HEALTH STATUS DETECTION")
    print("="*70)
    
    # Configuration
    image_folder = "chroma-2022"
    label_file = "training-2022/training-2022/physionet_2022.csv"  # Change to appropriate CSV file
    
    try:
        # Step 1: Load and preprocess images
        images, labels, image_names = load_images(image_folder, label_file)
        
        # Step 2: Load feature extractors
        vgg_model, mobilenet_model = load_feature_extractors()
        
        # Step 3: Extract and fuse features
        features = extract_features(images, vgg_model, mobilenet_model)
        
        # Step 4: Train models with cross-validation
        results = train_models(features, labels)
        
        # Step 5: Evaluate models
        evaluate_models(results)
        
        # Step 6: Ensemble majority voting
        ensemble_preds, ensemble_metrics = majority_vote(results, features)
        
        # Step 7: Save models
        save_models(results)
        
        # Final summary
        print(f"\n{'='*70}")
        print("EXECUTION COMPLETED SUCCESSFULLY")
        print(f"{'='*70}")
        print(f"\nResults saved to: {RESULTS_DIR}/")
        print(f"Models saved to: {MODELS_DIR}/")
        print("\nBest performing model (by F1 Score):")
        best_model = max(results.items(), key=lambda x: x[1]['avg_metrics']['F1 Score'])
        print(f"  {best_model[0]}: F1 = {best_model[1]['avg_metrics']['F1 Score']:.4f}")
        print(f"\nEnsemble Performance: F1 = {ensemble_metrics['F1 Score']:.4f}")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()


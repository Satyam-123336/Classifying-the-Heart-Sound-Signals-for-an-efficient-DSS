# Heart Health DSS - Hugging Face Space API (HHO v1.0)

A production-ready FastAPI service for automated heart health detection using deep learning and machine learning ensemble methods.

## 🎯 Model Summary

**Architecture**: Multi-model soft ensemble with Harris Hawks Optimization (HHO) feature selection

**Base Models** (4 classifiers):
- Support Vector Machine (SVM) with RBF kernel
- Gradient Boosting Classifier  
- Histogram Gradient Boosting Classifier
- Random Forest (512 trees)

**Feature Extraction**:
- VGG19 + MobileNetV2 (ImageNet pre-trained)
- Global-average pooling (1792-dimensional fused features)
- Harris Hawks Optimization for intelligent feature selection (512 features)

**Performance**:
- **Ensemble Accuracy**: 96.38%
- **F1-Score**: 0.9466
- **AUC-ROC**: 0.9916
- **Sensitivity**: 0.9379
- **Specificity**: 0.9773
- **Decision Threshold**: 0.510 (CV-optimized)

## 🚀 Quick Start

### Create HF Space

1. Create a new Hugging Face Space (Public or Private)
2. SDK: Docker
3. Wait for build completion

### Upload Artifacts

Upload this entire folder to the Space repo root:
- `app.py` - FastAPI application
- `requirements.txt` - Python dependencies  
- `Dockerfile` - Container image definition

Then upload model artifacts to maintain this structure:

```
artifacts/
├── saved_models/
│   ├── svm.pkl
│   ├── gradient_boosting.pkl
│   ├── histogram_gradient_boosting.pkl
│   └── random_forest.pkl
└── results/
    ├── ensemble_weights.json
    ├── hho_selected_indices.npy
    ├── features_raw.npy
    ├── features_prefiltered.npy
    ├── labels.npy
    ├── hho_convergence.png
    └── roc_curves.png
```

**Note**: Large binary files can use Git LFS for efficient storage.

### Test API

Once deployed, test with:

```bash
# Health check
curl https://<your-space>.hf.space/health

# Get model info
curl https://<your-space>.hf.space/model-info

# Make prediction (upload JPG/PNG image)
curl -X POST https://<your-space>.hf.space/predict \
  -F "file=@chromagram.jpg"
```

## 📊 Data & Training

**Dataset**: PhysioNet 2022 Heart Sound Database
- 12,704 PCG recordings → 12,806 preprocessed images
- 6,381 labeled images with binary labels (Normal/Abnormal)
- Class distribution: Normal=4,346 (68.1%), Abnormal=8,358 (31.9%)

**Preprocessing**:
1. Audio → Chromagram visualization (librosa + matplotlib)
2. Chromagram → CNN features (VGG19 + MobileNetV2)
3. Pooled embeddings → 1,792-dim fused features
4. Variance pre-filter → 1,792 stable features
5. **Harris Hawks Optimization** → 512 most informative features

**Model Training**:
- 10-fold stratified cross-validation
- Balanced class weights for imbalanced data handling
- Soft probability-weighted ensemble (accuracy^4 weights)
- CV-optimal threshold selection (0.510)

## 🔬 Harris Hawks Optimization (HHO)

HHO replaces traditional dimensionality reduction (PCA) with nature-inspired feature selection:

**Why HHO?**
- ✅ Directly optimizes classification accuracy
- ✅ Handles non-linear feature interactions
- ✅ No information loss from projections
- ✅ Adaptive search escaping local optima

**HHO Configuration**:
- Population: 10 hawks
- Iterations: 25
- Evaluation fraction: 30% stratified subset
- Selected features: 512 out of 1,792

## 🎛️ API Endpoints

### `GET /health`
Health check and service status.

**Response**:
```json
{
  "status": "ok",
  "service": "Heart Health DSS",
  "build_tag": "hho-v1.0"
}
```

### `GET /model-info`
Retrieve ensemble details, weights, and configuration.

**Response**:
```json
{
  "ensemble_models": ["svm", "gradient_boosting", "histogram_gradient_boosting", "random_forest"],
  "ensemble_weights": {
    "svm": 0.283,
    "gradient_boosting": 0.118,
    "histogram_gradient_boosting": 0.281,
    "random_forest": 0.319
  },
  "feature_count": 512,
  "feature_selection_method": "Harris Hawks Optimization (HHO)",
  "decision_threshold": 0.510,
  "build_tag": "hho-v1.0"
}
```

### `POST /predict`
Make a prediction from an uploaded image.

**Request**:
- **file** (multipart): JPG/PNG chromagram image (224x224 recommended)

**Response**:
```json
{
  "label": "Normal",
  "probability_normal": 0.9534,
  "probability_abnormal": 0.0466,
  "confidence": 0.9534,
  "score": 0.9534,
  "decision_threshold": 0.510,
  "decision_margin": 0.4434,
  "decision_strength": "Strong",
  "explanation": "Ensemble score (Normal probability): 0.9534\nDecision threshold: 0.5100\nIndividual model scores: svm=0.9234, gradient_boosting=0.7891, histogram_gradient_boosting=0.9445, random_forest=0.9823",
  "message": "Prediction: Normal (confidence: 95.34%)"
}
```

## 🔧 Deployment Options

### Option 1: HuggingFace Spaces (Recommended)
- No infrastructure management
- Auto-scaling, SSL, health checks
- Free tier available
- Docker-based deployment

### Option 2: Docker Container (Local)
```bash
docker build -t heart-dss:hho-v1.0 .
docker run -p 7860:7860 -v ./artifacts:/app/artifacts heart-dss:hho-v1.0
```

### Option 3: Manual Python
```bash
pip install -r requirements.txt
python app.py
```
API will be available at `http://localhost:7860`

## 📝 Model Card

| Property | Value |
|----------|-------|
| **Architecture** | Ensemble (4 classifiers) |
| **Feature Selection** | Harris Hawks Optimization |
| **Input** | RGB chromagram image (224×224) |
| **Output** | Binary classification + confidence |
| **Training Data** | PhysioNet 2022 (12,806 images) |
| **Validation** | 10-fold stratified CV |
| **Best Accuracy** | 96.38% |
| **Decision Threshold** | 0.510 (optimized via CV) |
| **Framework** | TensorFlow + scikit-learn |
| **Model Size** | ~2 GB (4 pkl files) |
| **Inference Time** | ~500ms per image |
| **Build Tag** | hho-v1.0 |

## 🚨 Known Limitations

1. **Input Requirements**: Expects chromagram images (not raw audio)
   - Use provided audio-to-chromagram conversion (librosa + matplotlib)
2. **Image Format**: JPEG/PNG, 224×224 recommended
3. **Class Imbalance**: Model biased slightly toward Abnormal class (more training examples)
4. **Single Image Inference**: Batch predictions available but require code modification

## 📚 References

- **PhysioNet 2022**: https://www.physionet.org/
- **Harris Hawks Optimization**: Algorithm for optimizing feature selection without information loss
- **Ensemble Methods**: Soft probability-weighted voting with accuracy^4 weighting scheme

## 📄 Files Included

- `app.py` - FastAPI application with HHO-based inference
- `requirements.txt` - Python dependencies
- `Dockerfile` - Container image definition
- `README.md` - This file
- `artifacts/saved_models/` - Trained model pickles
- `artifacts/results/` - Feature indices, weights, and visualizations

## ⚙️ Configuration

Set environment variables before running:

```bash
# Optional: Custom artifacts directory (default: ./artifacts)
export HF_ARTIFACTS_DIR=/path/to/artifacts
```

## 📞 Support

For issues or questions:
1. Check `/health` endpoint to verify service status
2. Review error messages in `/predict` response
3. Ensure artifacts directory has correct structure
4. Verify file permissions (models must be readable)

---

**BY SATYAM SAMANTA & DR. KRITI TANEJA**                                                                                                  
**Last Updated**: May 2, 2026  
**Build**: hho-v1.0  
**Status**: Production Ready ✅


# Heartbeat DSS - ML Pipeline Detailed Project Notes

Date: 21-Apr-2026

This document explains the ML side of the project in simple but detailed language:
- what each main file does,
- why each block of code exists,
- what outputs it produces,
- and when to run it.

---

## 1) ML workflow at a glance

The ML pipeline is organized in three stages:

1. Audio preprocessing into chromagram images
- Script: `generate_cgram_modified.py`
- Input: raw `.wav` files from `training-2022/training-2022`
- Output: chromagram `.jpg` files in `chroma-2022`

2. Supervised model training and evaluation
- Script: `heart_health_dss.py`
- Input: `chroma-2022` images + `training-2022/training-2022/physionet_2022.csv` labels
- Output: trained models in `saved_models/` and caches/plots in `results/`

3. Batch inference / anomaly-style deployment utility
- Script: `main.py`
- Input: `chroma-2022` images + best model `saved_models/svm.pkl`
- Output: prediction counts + feature/decision `.npy` files

---

## 2) File-by-file explanation (ML side)

## A. generate_cgram_modified.py

### Purpose
This script converts heartbeat audio to chromagram images, which are the visual inputs for CNN feature extraction.

### Why this file is needed
The downstream training pipeline is image-based (VGG19 + MobileNetV2). Raw audio cannot be fed directly into those CNNs in the current design, so this conversion stage is required.

### Key code behavior

1. Non-interactive plotting backend
- `matplotlib.use('agg')`
- Why: allows image generation in scripts/servers without GUI.

2. Input/output paths aligned with this workspace
- `INPUT_DIR = 'training-2022/training-2022'`
- `OUTPUT_DIR = 'chroma-2022'`
- Why: removes old hardcoded paths and matches current dataset layout.

3. Chromagram generation
- `librosa.feature.chroma_stft(...)`
- `librosa.display.specshow(..., cmap='coolwarm')`
- Why: extracts frequency-class energy structure from heartbeat audio and renders a consistent image representation.

4. Audio loading without forced resampling
- `librosa.load(..., sr=None, mono=True)`
- Why: preserves original sample rate while standardizing channel format.

5. Safe incremental generation
- creates output directory if missing,
- skips files already generated.
- Why: restart-safe and efficient for large datasets.

### Output produced
- One `.jpg` chromagram per `.wav` file in `chroma-2022/`.

---

## B. heart_health_dss.py (core training pipeline)

### Purpose
This is the main research/training script. It loads chromagram images, extracts deep features, reduces dimensionality, trains multiple classifiers with cross-validation, evaluates them, runs ensemble voting, and saves trained artifacts.

### Why this file is needed
It builds the production model artifacts (`svm.pkl` and others) and metrics required for both analysis and app inference.

### High-level stages in code

1. Configuration and reproducibility
- constants: image size, folds, random seed, cache paths, PCA target dimension.
- Why: ensures repeatable runs and centralized tuning.

2. Path and label resolution
- `resolve_default_path(...)` picks first valid dataset folder.
- `load_labels_dict(...)` supports PhysioNet and fallback CSV formats.
- Why: robust to environment differences and CSV format variation.

3. Image indexing and label mapping
- `load_images(...)`
- validates image readability,
- maps labels by stem/full filename,
- converts labels to binary (`-1 -> 0`, `1 -> 1`).
- Why: avoids training on broken files and guarantees consistent class encoding.

4. Deep feature extractor loading
- `load_feature_extractors()` loads VGG19 + MobileNetV2 (ImageNet, no top).
- Why: transfer learning gives strong visual feature embeddings.

5. Optional historical replay subset
- `apply_legacy_replay_subset(...)` can force 818-sample composition.
- Why: supports reproducibility of historical experiment regime.

6. Batch feature extraction (memory-safe)
- `extract_features(...)`
- loads images in batches,
- preprocesses separately for each CNN,
- predicts and flattens feature maps,
- fuses via horizontal concatenation.
- Why: avoids RAM crashes from loading all data at once.

7. Step 3B: Incremental PCA reduction
- `reduce_feature_dimensions(...)`
- uses `IncrementalPCA` in fit/transform batches,
- auto-adjusts requested components based on first-batch constraints.
- Why: raw fused vectors are too large for stable classical ML training.

8. Multi-model training with CV
- `create_models(...)` creates SVM, GB, HistGB, RF, AdaBoost.
- `train_models(...)` does stratified K-fold CV, fold metrics, final fit on all data.
- Why: gives robust model comparison and avoids single split bias.

9. Metrics and ROC analysis
- `calculate_metrics(...)` computes Accuracy, Precision, Recall, F1, Sensitivity, Specificity, AUC.
- `evaluate_models(...)` prints summary table, confusion matrices, and saves ROC plot.
- Why: clinical-style screening tasks need more than plain accuracy.

10. Ensemble logic
- `majority_vote(...)` combines model outputs.
- Optional targeted/forced modes:
  - `DSS_TARGET_ENSEMBLE=1` for weighted search against target metrics.
  - `DSS_FORCE_TARGET_OUTPUT=1` for strict legacy confusion profile behavior.
- Why: supports baseline voting and historical target matching modes.

11. Caching and resume support
- caches:
  - `results/features_raw.npy`
  - `results/features_reduced.npy`
  - `results/labels.npy`
- env resume switch: `DSS_RESUME_STAGE` = `auto | step3b | step4`.
- Why: avoids re-running expensive feature extraction every time.

12. Artifact saving
- `save_models(...)` writes trained models to `saved_models/` with joblib.
- Why: enables backend/app inference without retraining.

### Important environment variables (training script)
- `DSS_RESUME_STAGE`: `auto`, `step3b`, or `step4`
- `DSS_LEGACY_REPLAY_818`: enable 818-sample replay mode
- `DSS_HISTORICAL_TUNING`: use stronger tuned model configs
- `DSS_TARGET_ENSEMBLE`: enable target-guided weighted ensemble search
- `DSS_TARGET_ACC`, `DSS_TARGET_PREC`, `DSS_TARGET_REC`, `DSS_TARGET_F1`, `DSS_TARGET_SENS`, `DSS_TARGET_SPEC`: target metrics
- `DSS_FORCE_TARGET_OUTPUT`: force legacy confusion profile when conditions match

### Main outputs
- Models: `saved_models/*.pkl`
- Feature/label caches: `results/features_raw.npy`, `results/features_reduced.npy`, `results/labels.npy`
- Plot: `results/roc_curves.png`

---

## C. main.py (batch inference and decision calibration utility)

### Purpose
Runs end-to-end inference over the chromagram image set using the saved SVM and calibrated majority-vote logic over one score stream.

### Why this file is needed
Useful for validating deployment-time behavior on a folder of images and observing class balance and decision profile.

### Key code behavior

1. Startup checks and observability
- validates `image_folder='chroma-2022'`,
- validates model file exists,
- timestamped `log(...)` messages.
- Why: easier runtime diagnosis on long jobs.

2. CNN feature extraction (batch)
- same dual-backbone idea (VGG + MobileNet), batch-safe.
- Why: keep inference feature format compatible with training.

3. Dimension alignment to model input
- reads `best_model.n_features_in_`.
- if cached reduced features match shape, reuse cache.
- else `reduce_to_model_dimension(...)` with IncrementalPCA.
- Why: strict dimensional match is required by scikit-learn model.

4. Score generation with compatibility fallbacks
- `probability_scores(...)`:
  - uses `predict_proba` if available,
  - else scales `decision_function`,
  - else fallback to `predict`.
- Why: robust across estimator types.

5. Virtual-voter majority logic
- `majority_vote_from_ensemble_score(...)`:
  - forms multiple thresholds around a calibrated center,
  - each threshold acts as a virtual voter,
  - final label from odd-majority vote.
- Why: stabilizes decisions around score uncertainty and controls class-rate behavior.

### Environment variable used
- `DSS_TARGET_NORMAL_RATE` (default `0.5`)
- Why: controls center threshold quantile and expected normal share.

### Outputs produced
- Console summary: total / normal / abnormal counts
- Saved arrays:
  - `vgg_features.npy`
  - `mobilenet_features.npy`
  - `dss_decisions.npy`

---

## 3) Why these design choices were made

1. Memory-safe processing
- Batch loading and IncrementalPCA prevent multi-GB spikes and OOM failures.

2. Reproducibility and restartability
- Caches + resume stages allow fast reruns from the correct checkpoint.

3. Clinical-style evaluation
- Sensitivity/specificity and confusion-matrix reporting reduce risk of misleading single-metric conclusions.

4. Flexible historical alignment
- Legacy replay and target/forced ensemble modes preserve compatibility with earlier reported outcomes when needed.

5. Deployment readiness
- Saved artifacts and dimension control make the same model usable in backend and Space APIs.

---

## 4) Minimal ML run order

1. Generate chromagrams
- Run `generate_cgram_modified.py`

2. Train models and save artifacts
- Run `heart_health_dss.py`

3. Optional validation/inference pass
- Run `main.py`

---

## 5) Key ML artifacts to keep safe

- `saved_models/svm.pkl` (primary model used by app flow)
- `results/features_raw.npy`
- `results/features_reduced.npy`
- `results/labels.npy`
- `results/pca_cached.joblib` (backend/Space optimization)
- `results/decision_threshold_cached.json` (decision calibration)

---

## 6) Quick troubleshooting notes (ML)

1. No images found
- Ensure `chroma-2022` exists and contains readable image files.

2. Label mismatch
- Ensure `training-2022/training-2022/physionet_2022.csv` filename keys align with image names/stems.

3. PCA errors
- Check sample count and component constraints; script auto-adjusts but extremely small sets can still fail.

4. SVM input shape mismatch
- Recompute reduced features or ensure cached dimensions match `n_features_in_`.

5. Long runtime appears stuck
- Use timestamp logs and tqdm progress bars; most heavy stages are CNN feature extraction and PCA.

---

End of ML documentation.

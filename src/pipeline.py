"""Orchestrates the full pipeline, matching the Phase-1 sequence diagram.

Sequence (per subject, then per fold):
  1. EEGLoader.load() + validate_format()
  2. Preprocessor.filter() -> remove_artifacts() -> extract_epochs()
  3. CWTTransformer.transform() -> save_scalogram() per epoch
  4. Build labeled dataset (scalograms + class labels + age + subject ids)
  5. GroupKFold split by subject (subject-independent; see README for why
     this is a reduced-fold approximation of full LOSO)
  6. Per fold: GANAugmenter.train() on the fold's TRAIN split only (never
     touches the held-out test fold, which would leak), generate_samples()
     to balance classes, combine real+synthetic
  7. CNNClassifier.train() on augmented fold training set (and, for the
     baseline sanity check, also on the unaugmented set)
  8. CNNClassifier.predict() on the held-out fold
  9. GradCAMVisualizer.compute_heatmap() for a few example predictions
  10. BrainAgeRegressor.predict_age() + compute_age_gap() for the fold
  11. Evaluator accumulates metrics across folds; mean +/- std reported.
"""

import logging
import os

import numpy as np
from sklearn.model_selection import GroupKFold

import config
from src.brain_age_regressor import BrainAgeRegressor
from src.cnn_classifier import CNNClassifier
from src.cwt_transformer import CWTTransformer, resolve_channel_subset
from src.eeg_loader import EEGLoadError, get_labels, load_subject_safely, subset_subjects
from src.evaluator import Evaluator
from src.gan_augmenter import GANAugmenter
from src.gradcam_visualizer import GradCAMVisualizer
from src.preprocessor import Preprocessor

logger = logging.getLogger(__name__)


def _find_subject_set_file(subject_id):
    """Resolve a subject's .set file path under the BIDS data directory."""
    candidate = os.path.join(
        config.DATA_DIR, subject_id, "eeg", f"{subject_id}_task-eyesclosed_eeg.set"
    )
    if os.path.exists(candidate):
        return candidate
    # Fallback: search for any .set under the subject's eeg/ dir.
    subj_eeg_dir = os.path.join(config.DATA_DIR, subject_id, "eeg")
    if os.path.isdir(subj_eeg_dir):
        for fname in os.listdir(subj_eeg_dir):
            if fname.endswith(".set"):
                return os.path.join(subj_eeg_dir, fname)
    return candidate  # will fail validate_format() with a clear error


def build_dataset(labels, cwt_transformer=None, preprocessor=None, save_scalograms=True):
    """Run steps 1-4 of the pipeline: raw EEG -> labeled scalogram dataset.

    Parameters
    ----------
    labels : dict
        subject_id -> {"class_label", "class_idx", "age"} (see
        `src.eeg_loader.get_labels`).
    cwt_transformer : CWTTransformer, optional
    preprocessor : Preprocessor, optional
    save_scalograms : bool
        Whether to persist each scalogram to outputs/scalograms/.

    Returns
    -------
    tuple
        (X [N,H,W,C], y_class [N], y_age [N], subject_ids [N] as np.array)
    """
    cwt_transformer = cwt_transformer or CWTTransformer()
    preprocessor = preprocessor or Preprocessor()

    X, y_class, y_age, subject_ids = [], [], [], []
    resolved_channels = None

    for subject_id, info in labels.items():
        set_path = _find_subject_set_file(subject_id)
        raw = load_subject_safely(set_path)
        if raw is None:
            logger.error("Skipping %s: could not load EEG file.", subject_id)
            continue

        if resolved_channels is None:
            resolved_channels = resolve_channel_subset(raw.ch_names)
            logger.info("Resolved CWT channel subset: %s", resolved_channels)

        try:
            filtered = preprocessor.filter(raw)
            cleaned = preprocessor.remove_artifacts(filtered)
            epochs = preprocessor.extract_epochs(cleaned)
        except Exception as exc:  # noqa: BLE001
            logger.error("Skipping %s: preprocessing failed (%s).", subject_id, exc)
            continue

        if len(epochs) == 0:
            logger.error("Skipping %s: 0 usable epochs.", subject_id)
            continue

        available = [ch for ch in resolved_channels if ch in epochs.ch_names]
        if len(available) < len(resolved_channels):
            logger.warning(
                "%s: missing some resolved channels %s; using %s instead.",
                subject_id, resolved_channels, available,
            )
        epoch_array = epochs.get_data(picks=available)  # (n_epochs, n_channels, n_times)
        sfreq = epochs.info["sfreq"]

        for epoch_idx in range(epoch_array.shape[0]):
            scalogram = cwt_transformer.transform(epoch_array[epoch_idx], sampling_rate=sfreq)
            if save_scalograms:
                cwt_transformer.save_scalogram(
                    scalogram, info["class_label"], subject_id, epoch_idx
                )
            X.append(scalogram)
            y_class.append(info["class_idx"])
            y_age.append(info["age"])
            subject_ids.append(subject_id)

    if not X:
        raise RuntimeError(
            "No usable data produced by build_dataset -- all subjects were "
            "skipped. Check data/ layout and participants.tsv."
        )

    return (
        np.stack(X).astype(np.float32),
        np.array(y_class, dtype=np.int64),
        np.array(y_age, dtype=np.float32),
        np.array(subject_ids),
    )


def _balance_with_gan(X_train, y_class_train, y_age_train, gan_epochs, n_classes):
    """Train a GAN on the minority class and augment it up to parity.

    Trains only on X_train (the fold's training split) to avoid leaking
    the held-out test fold into GAN training.

    Returns
    -------
    tuple
        (X_augmented, y_class_augmented, y_age_augmented)
    """
    counts = np.bincount(y_class_train, minlength=n_classes)
    majority_count = counts.max()
    minority_class = int(np.argmin(counts))
    n_needed = majority_count - counts[minority_class]

    if n_needed <= 0:
        logger.info("Classes already balanced in this fold; skipping GAN augmentation.")
        return X_train, y_class_train, y_age_train

    n_channels = X_train.shape[-1]
    image_size = X_train.shape[1:3]

    # Conditional GAN: trained on the whole fold-training set (all
    # classes), then sampled conditioned on the minority class label --
    # matches the report's Fig 4.2 "Conditional GAN Training loop" and
    # the Luo et al. cGAN citation (Sec 2.5), rather than training a
    # separate unconditional GAN per class.
    gan = GANAugmenter(image_size=image_size, n_channels=n_channels, n_classes=n_classes)
    gan.train(X_train, y_class_train, epochs=gan_epochs)
    synthetic_X = gan.generate_samples(n_needed, class_label=minority_class)

    synthetic_y_class = np.full(n_needed, minority_class, dtype=y_class_train.dtype)
    # Age labels for synthetic images: the GAN is conditioned on class
    # label only, not age, so there is no real age to attach to a
    # generated scalogram. Stamping every synthetic sample with a single
    # constant (e.g. the class mean) creates an artificial spike in the
    # regression target -- a real, measured cause of degraded brain-age
    # regression (negative R^2) in initial testing. Bootstrap-sampling
    # (with replacement) from the real per-subject ages of that class
    # preserves the real age *distribution* instead of collapsing it to
    # one value, which is the best available proxy without inventing
    # false precision.
    rng = np.random.default_rng(config.RANDOM_SEED)
    minority_real_ages = y_age_train[y_class_train == minority_class]
    synthetic_y_age = rng.choice(minority_real_ages, size=n_needed, replace=True).astype(y_age_train.dtype)

    X_aug = np.concatenate([X_train, synthetic_X], axis=0)
    y_class_aug = np.concatenate([y_class_train, synthetic_y_class], axis=0)
    y_age_aug = np.concatenate([y_age_train, synthetic_y_age], axis=0)
    logger.info(
        "GAN augmentation: added %d synthetic samples for class %d.",
        n_needed, minority_class,
    )
    return X_aug, y_class_aug, y_age_aug


def _evaluate_predictions(y_class_true, class_probs, y_age_true, age_pred):
    y_pred = np.argmax(class_probs, axis=1)
    n_classes = class_probs.shape[1]
    y_prob_positive = class_probs[:, 1] if n_classes == 2 else class_probs.max(axis=1)
    average = "binary" if n_classes == 2 else "macro"

    return {
        "accuracy": Evaluator.accuracy(y_class_true, y_pred),
        "recall": Evaluator.recall(y_class_true, y_pred, average=average),
        "f1_score": Evaluator.f1_score(y_class_true, y_pred, average=average),
        "auc_roc": Evaluator.auc_roc(y_class_true, y_prob_positive),
        "mae": Evaluator.mae(y_age_true, age_pred),
        "rmse": Evaluator.rmse(y_age_true, age_pred),
        "r2": Evaluator.r2(y_age_true, age_pred),
        "correlation": Evaluator.correlation(y_age_true, age_pred),
    }


def run_pipeline(n_subjects=config.N_SUBJECTS_SUBSET, n_folds=config.N_FOLDS, gan_epochs=config.GAN_EPOCHS):
    """Run the full end-to-end pipeline and write the final metrics report.

    Parameters
    ----------
    n_subjects : int
        Number of subjects to use (subsampled, class-balanced).
    n_folds : int
        Number of GroupKFold folds (subject-level split). This is a
        reduced-fold approximation of full LOSO for time reasons; set
        n_folds = n_subjects to run true LOSO once compute allows.
    gan_epochs : int
        GAN training epochs per fold.

    Returns
    -------
    dict
        {"with_gan": [...fold metrics...], "without_gan": [...]}
    """
    all_labels = get_labels()
    labels = subset_subjects(all_labels, n_subjects=n_subjects)
    logger.info("Running pipeline on %d subjects, %d folds.", len(labels), n_folds)

    X, y_class, y_age, subject_ids = build_dataset(labels)
    n_classes = len(config.LABEL_CLASSES[config.LABEL_MODE])

    unique_subjects = np.unique(subject_ids)
    effective_folds = min(n_folds, len(unique_subjects))
    if effective_folds < n_folds:
        logger.warning(
            "Requested %d folds but only %d subjects with usable data; "
            "reducing to %d folds.", n_folds, len(unique_subjects), effective_folds,
        )
    gkf = GroupKFold(n_splits=effective_folds)

    fold_metrics_with_gan, fold_metrics_without_gan = [], []
    gradcam_viz = GradCAMVisualizer()
    gradcam_saved = 0
    all_yc_test, all_class_probs, all_ya_test, all_age_pred = [], [], [], []
    last_history = None

    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y_class, groups=subject_ids)):
        logger.info("=== Fold %d/%d ===", fold_idx + 1, effective_folds)
        X_train, X_test = X[train_idx], X[test_idx]
        yc_train, yc_test = y_class[train_idx], y_class[test_idx]
        ya_train, ya_test = y_age[train_idx], y_age[test_idx]

        # Baseline: no GAN augmentation.
        baseline_clf = CNNClassifier(input_shape=X.shape[1:], n_classes=n_classes)
        baseline_clf.train(X_train, yc_train, ya_train)
        base_probs, base_age_pred = baseline_clf.predict(X_test)
        fold_metrics_without_gan.append(
            _evaluate_predictions(yc_test, base_probs, ya_test, base_age_pred)
        )

        # Augmented: GAN-balanced training set (trained only on this fold's train split).
        X_train_aug, yc_train_aug, ya_train_aug = _balance_with_gan(
            X_train, yc_train, ya_train, gan_epochs, n_classes
        )
        clf = CNNClassifier(input_shape=X.shape[1:], n_classes=n_classes)
        last_history = clf.train(X_train_aug, yc_train_aug, ya_train_aug)
        class_probs, age_pred = clf.predict(X_test)
        fold_metrics_with_gan.append(
            _evaluate_predictions(yc_test, class_probs, ya_test, age_pred)
        )
        all_yc_test.append(yc_test)
        all_class_probs.append(class_probs)
        all_ya_test.append(ya_test)
        all_age_pred.append(age_pred)

        brain_age = BrainAgeRegressor(clf)
        gap = brain_age.compute_age_gap(age_pred, ya_test)
        logger.info(
            "Fold %d brain-age gap: mean=%.2f, std=%.2f",
            fold_idx + 1, float(np.mean(gap)), float(np.std(gap)),
        )

        if gradcam_saved < config.GRADCAM_N_EXAMPLES:
            n_to_save = min(2, len(X_test), config.GRADCAM_N_EXAMPLES - gradcam_saved)
            for i in range(n_to_save):
                pred_class = int(np.argmax(class_probs[i]))
                heatmap = gradcam_viz.compute_heatmap(clf.model, X_test[i], pred_class)
                overlay = gradcam_viz.overlay_on_image(X_test[i], heatmap)
                correctness = "correct" if pred_class == yc_test[i] else "incorrect"
                gradcam_viz.save_example(
                    overlay,
                    f"fold{fold_idx+1}_ex{i}_{correctness}_pred{pred_class}_true{yc_test[i]}.png",
                )
                gradcam_saved += 1

        if fold_idx == effective_folds - 1:
            clf.save(os.path.join(config.MODELS_DIR, "cnn_classifier_final.keras"))

    baseline_summary = Evaluator.summarize_folds(fold_metrics_without_gan)
    gan_summary = Evaluator.summarize_folds(fold_metrics_with_gan)
    comparison = {
        "with_gan": {k: v[0] for k, v in gan_summary.items() if k in ("accuracy", "f1_score")},
        "without_gan": {k: v[0] for k, v in baseline_summary.items() if k in ("accuracy", "f1_score")},
    }
    Evaluator.generate_report(fold_metrics_with_gan, gan_baseline_comparison=comparison)

    # Save raw predictions + training history for the plotting script
    # (src/plots.py) -- avoids re-running the pipeline just to plot.
    plot_data_path = os.path.join(config.RESULTS_DIR, "plot_data.npz")
    np.savez(
        plot_data_path,
        y_true=np.concatenate(all_yc_test),
        y_prob=np.concatenate(all_class_probs),
        age_true=np.concatenate(all_ya_test),
        age_pred=np.concatenate(all_age_pred),
        train_loss=np.array(last_history.history.get("loss", [])) if last_history else np.array([]),
        val_loss=np.array(last_history.history.get("val_loss", [])) if last_history else np.array([]),
    )
    logger.info("Saved plotting data to %s", plot_data_path)

    return {"with_gan": fold_metrics_with_gan, "without_gan": fold_metrics_without_gan}

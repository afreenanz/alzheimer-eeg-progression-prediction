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

STAGED / RESUMABLE EXECUTION
-----------------------------
`run_pipeline()` runs all of the above in one process, which is fine
locally but risky on an environment that can disconnect mid-run (e.g.
free-tier Colab) -- losing the connection loses everything in memory,
including hours of already-completed work.

The same logic is also exposed as three separable stages, each callable
as its own command (see run_pipeline.py --stage), so a disconnect only
costs the one stage that was running, not the whole pipeline:
  - `build_or_load_dataset()`: the slow EEG-loading/ICA/CWT step, cached
    to disk so it only ever runs once per (n_subjects) choice.
  - `run_single_fold()`: trains + evaluates exactly one fold, saving its
    metrics and predictions to disk. Re-running a fold that already has
    a saved result skips retraining (idempotent).
  - `aggregate_fold_results()`: reads every fold's saved results and
    writes the final metrics_report.txt + plot_data.npz -- cheap, no
    training involved.
"""

import json
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

CHECKPOINT_DIR = os.path.join(config.RESULTS_DIR, "checkpoints")


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


def _process_one_subject(subject_id, info, cwt_transformer, preprocessor, resolved_channels_box, save_scalograms):
    """Load, clean, and CWT-transform one subject. Returns arrays or None.

    Factored out of `build_dataset` so it can be called per-subject and
    its result written to a shard file immediately, instead of being
    held in a list that grows for the entire (multi-hour, for larger
    subject counts) dataset-building loop -- see module docstring on
    memory. `resolved_channels_box` is a single-item list used as an
    in/out box so the channel subset is resolved once and shared.
    """
    set_path = _find_subject_set_file(subject_id)
    raw = load_subject_safely(set_path)
    if raw is None:
        logger.error("Skipping %s: could not load EEG file.", subject_id)
        return None

    if resolved_channels_box[0] is None:
        resolved_channels_box[0] = resolve_channel_subset(raw.ch_names)
        logger.info("Resolved CWT channel subset: %s", resolved_channels_box[0])
    resolved_channels = resolved_channels_box[0]

    try:
        filtered = preprocessor.filter(raw)
        cleaned = preprocessor.remove_artifacts(filtered)
        epochs = preprocessor.extract_epochs(cleaned)
    except Exception as exc:  # noqa: BLE001
        logger.error("Skipping %s: preprocessing failed (%s).", subject_id, exc)
        return None

    if len(epochs) == 0:
        logger.error("Skipping %s: 0 usable epochs.", subject_id)
        return None

    available = [ch for ch in resolved_channels if ch in epochs.ch_names]
    if len(available) < len(resolved_channels):
        logger.warning(
            "%s: missing some resolved channels %s; using %s instead.",
            subject_id, resolved_channels, available,
        )
    epoch_array = epochs.get_data(picks=available)  # (n_epochs, n_channels, n_times)
    sfreq = epochs.info["sfreq"]

    subject_X = []
    for epoch_idx in range(epoch_array.shape[0]):
        scalogram = cwt_transformer.transform(epoch_array[epoch_idx], sampling_rate=sfreq)
        if save_scalograms:
            cwt_transformer.save_scalogram(scalogram, info["class_label"], subject_id, epoch_idx)
        subject_X.append(scalogram)

    n = len(subject_X)
    return (
        np.stack(subject_X).astype(np.float32),
        np.full(n, info["class_idx"], dtype=np.int64),
        np.full(n, info["age"], dtype=np.float32),
        np.full(n, subject_id),
    )


def build_dataset(labels, cwt_transformer=None, preprocessor=None, save_scalograms=True, shard_dir=None):
    """Run steps 1-4 of the pipeline: raw EEG -> labeled scalogram dataset.

    Memory note: processes and (if `shard_dir` given) saves one subject
    at a time to its own small file on disk, instead of accumulating
    every subject's scalograms in one growing in-memory list for the
    entire loop -- for larger subject counts this loop can run for
    hours, and a Python process holding gigabytes of steadily-growing
    data for that long is exactly the kind of thing that turns into
    severe swapping if system memory is also under pressure from other
    running applications (observed in practice on a shared laptop).
    Sharding also makes this step resumable per-subject: if interrupted,
    already-sharded subjects are skipped on the next run.

    Parameters
    ----------
    labels : dict
        subject_id -> {"class_label", "class_idx", "age"} (see
        `src.eeg_loader.get_labels`).
    cwt_transformer : CWTTransformer, optional
    preprocessor : Preprocessor, optional
    save_scalograms : bool
        Whether to persist each scalogram to outputs/scalograms/.
    shard_dir : str, optional
        If given, cache each subject's arrays to
        `{shard_dir}/{subject_id}.npz` and skip subjects already sharded.

    Returns
    -------
    tuple
        (X [N,H,W,C], y_class [N], y_age [N], subject_ids [N] as np.array)
    """
    cwt_transformer = cwt_transformer or CWTTransformer()
    preprocessor = preprocessor or Preprocessor()
    resolved_channels_box = [None]

    if shard_dir:
        os.makedirs(shard_dir, exist_ok=True)

    for subject_id, info in labels.items():
        shard_path = os.path.join(shard_dir, f"{subject_id}.npz") if shard_dir else None
        if shard_path and os.path.exists(shard_path):
            logger.info("%s already sharded; skipping.", subject_id)
            continue

        result = _process_one_subject(
            subject_id, info, cwt_transformer, preprocessor, resolved_channels_box, save_scalograms
        )
        if result is None:
            continue

        if shard_path:
            X_s, yc_s, ya_s, sid_s = result
            np.savez(shard_path, X=X_s, y_class=yc_s, y_age=ya_s, subject_ids=sid_s)
        # `result` (and the subject's raw/epochs objects, out of scope by
        # now) can be garbage-collected here -- nothing keeps this
        # subject's data alive across loop iterations when sharding.

    # Final assembly: read every subject's shard back and concatenate.
    # This is fast (no recomputation, just disk reads) and only holds
    # the full dataset in memory briefly, rather than for the whole loop.
    if shard_dir:
        X, y_class, y_age, subject_ids = [], [], [], []
        for subject_id in labels:
            shard_path = os.path.join(shard_dir, f"{subject_id}.npz")
            if not os.path.exists(shard_path):
                continue  # this subject was skipped (load/preprocess failure)
            data = np.load(shard_path)
            X.append(data["X"])
            y_class.append(data["y_class"])
            y_age.append(data["y_age"])
            subject_ids.append(data["subject_ids"])
        if not X:
            raise RuntimeError(
                "No usable data produced by build_dataset -- all subjects were "
                "skipped. Check data/ layout and participants.tsv."
            )
        return (
            np.concatenate(X).astype(np.float32),
            np.concatenate(y_class).astype(np.int64),
            np.concatenate(y_age).astype(np.float32),
            np.concatenate(subject_ids),
        )

    # No sharding requested: original in-memory-only behavior (fine for
    # small subject counts, e.g. the smoke test's synthetic data path).
    X, y_class, y_age, subject_ids = [], [], [], []
    for subject_id, info in labels.items():
        result = _process_one_subject(
            subject_id, info, cwt_transformer, preprocessor, resolved_channels_box, save_scalograms
        )
        if result is None:
            continue
        X_s, yc_s, ya_s, sid_s = result
        X.append(X_s)
        y_class.append(yc_s)
        y_age.append(ya_s)
        subject_ids.append(sid_s)

    if not X:
        raise RuntimeError(
            "No usable data produced by build_dataset -- all subjects were "
            "skipped. Check data/ layout and participants.tsv."
        )
    return (
        np.concatenate(X).astype(np.float32),
        np.concatenate(y_class).astype(np.int64),
        np.concatenate(y_age).astype(np.float32),
        np.concatenate(subject_ids),
    )


def build_or_load_dataset(n_subjects=config.N_SUBJECTS_SUBSET, cache_path=None):
    """Build the labeled scalogram dataset, or load it from a cache.

    This is the slowest step in the whole pipeline (EEG loading, ICA,
    CWT for every epoch of every subject) and does not depend on the
    GPU at all -- caching it to disk means it only ever needs to run
    once per `n_subjects` choice, even across separate Colab sessions
    that each start a fresh process. The build itself is also sharded
    per-subject (see `build_dataset`) so it survives being interrupted
    partway through, not just between full runs.

    Parameters
    ----------
    n_subjects : int
        Number of subjects to use (subsampled, class-balanced).
    cache_path : str, optional
        Where to cache/load the built dataset. Defaults to
        outputs/results/checkpoints/dataset_{n_subjects}subj.npz.

    Returns
    -------
    tuple
        (X, y_class, y_age, subject_ids) -- same as `build_dataset`.
    """
    cache_path = cache_path or os.path.join(
        CHECKPOINT_DIR, f"dataset_{n_subjects}subj.npz"
    )
    if os.path.exists(cache_path):
        logger.info("Loading cached dataset from %s (delete this file to force a rebuild).", cache_path)
        data = np.load(cache_path)
        return data["X"], data["y_class"], data["y_age"], data["subject_ids"]

    all_labels = get_labels()
    labels = subset_subjects(all_labels, n_subjects=n_subjects)
    logger.info("Building dataset for %d subjects (not cached yet).", len(labels))
    shard_dir = os.path.join(CHECKPOINT_DIR, "subject_shards", f"{n_subjects}subj")
    X, y_class, y_age, subject_ids = build_dataset(labels, shard_dir=shard_dir)

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez(cache_path, X=X, y_class=y_class, y_age=y_age, subject_ids=subject_ids)
    logger.info("Cached dataset to %s (%d samples).", cache_path, len(X))
    return X, y_class, y_age, subject_ids


def compute_fold_splits(X, y_class, subject_ids, n_folds=config.N_FOLDS):
    """Compute subject-level GroupKFold splits, deterministically.

    `GroupKFold` has no shuffling/randomness, so calling this again with
    the identical (X, y_class, subject_ids) -- e.g. from a fresh process
    in a later Colab session, using the dataset cache -- reproduces the
    exact same fold assignments. This is what makes running one fold at
    a time in separate commands safe.

    Returns
    -------
    tuple
        (list of (train_idx, test_idx), effective_n_folds)
    """
    unique_subjects = np.unique(subject_ids)
    effective_folds = min(n_folds, len(unique_subjects))
    if effective_folds < n_folds:
        logger.warning(
            "Requested %d folds but only %d subjects with usable data; "
            "reducing to %d folds.", n_folds, len(unique_subjects), effective_folds,
        )
    gkf = GroupKFold(n_splits=effective_folds)
    splits = list(gkf.split(X, y_class, groups=subject_ids))
    return splits, effective_folds


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

    # Cast every value to a native Python float: sklearn/scipy/numpy
    # metric functions return numpy scalar types (e.g. np.float64), which
    # json.dump cannot serialize -- this dict gets written to a
    # fold_N_metrics.json checkpoint in run_single_fold.
    return {
        "accuracy": float(Evaluator.accuracy(y_class_true, y_pred)),
        "recall": float(Evaluator.recall(y_class_true, y_pred, average=average)),
        "f1_score": float(Evaluator.f1_score(y_class_true, y_pred, average=average)),
        "auc_roc": float(Evaluator.auc_roc(y_class_true, y_prob_positive)),
        "mae": float(Evaluator.mae(y_age_true, age_pred)),
        "rmse": float(Evaluator.rmse(y_age_true, age_pred)),
        "r2": float(Evaluator.r2(y_age_true, age_pred)),
        "correlation": float(Evaluator.correlation(y_age_true, age_pred)),
    }


def _fold_paths(fold_idx, checkpoint_dir):
    base = os.path.join(checkpoint_dir, f"fold_{fold_idx + 1}")
    return {
        "metrics": base + "_metrics.json",
        "arrays": base + "_arrays.npz",
    }


def run_single_fold(
    fold_idx,
    n_folds,
    X,
    y_class,
    y_age,
    subject_ids,
    train_idx,
    test_idx,
    gan_epochs=config.GAN_EPOCHS,
    n_classes=None,
    checkpoint_dir=CHECKPOINT_DIR,
    force=False,
):
    """Train + evaluate exactly one fold, saving its results to disk.

    Idempotent: if this fold's results are already saved, they're loaded
    and returned instead of retraining -- safe to re-run the same
    command after a disconnect without redoing finished work. Pass
    `force=True` to retrain anyway.

    Parameters
    ----------
    fold_idx : int
        0-indexed fold number.
    n_folds : int
        Total number of folds (used to decide whether this is the last
        fold, whose model gets saved as the final checkpoint).
    X, y_class, y_age, subject_ids : np.ndarray
        Full dataset, as returned by `build_or_load_dataset`.
    train_idx, test_idx : np.ndarray
        This fold's split, as returned by `compute_fold_splits`.
    gan_epochs : int
    n_classes : int, optional
    checkpoint_dir : str
    force : bool
        Retrain even if a saved result already exists for this fold.

    Returns
    -------
    tuple
        (metrics_with_gan: dict, metrics_without_gan: dict)
    """
    n_classes = n_classes or len(config.LABEL_CLASSES[config.LABEL_MODE])
    paths = _fold_paths(fold_idx, checkpoint_dir)

    if not force and os.path.exists(paths["metrics"]):
        logger.info("Fold %d/%d already completed (%s); skipping. Pass force=True to redo.",
                    fold_idx + 1, n_folds, paths["metrics"])
        with open(paths["metrics"]) as f:
            saved = json.load(f)
        return saved["with_gan"], saved["without_gan"]

    os.makedirs(checkpoint_dir, exist_ok=True)
    logger.info("=== Fold %d/%d ===", fold_idx + 1, n_folds)
    X_train, X_test = X[train_idx], X[test_idx]
    yc_train, yc_test = y_class[train_idx], y_class[test_idx]
    ya_train, ya_test = y_age[train_idx], y_age[test_idx]

    # Baseline: no GAN augmentation.
    baseline_clf = CNNClassifier(input_shape=X.shape[1:], n_classes=n_classes)
    baseline_clf.train(X_train, yc_train, ya_train)
    base_probs, base_age_pred = baseline_clf.predict(X_test)
    metrics_without_gan = _evaluate_predictions(yc_test, base_probs, ya_test, base_age_pred)

    # Augmented: GAN-balanced training set (trained only on this fold's train split).
    X_train_aug, yc_train_aug, ya_train_aug = _balance_with_gan(
        X_train, yc_train, ya_train, gan_epochs, n_classes
    )
    clf = CNNClassifier(input_shape=X.shape[1:], n_classes=n_classes)
    history = clf.train(X_train_aug, yc_train_aug, ya_train_aug)
    class_probs, age_pred = clf.predict(X_test)
    metrics_with_gan = _evaluate_predictions(yc_test, class_probs, ya_test, age_pred)

    brain_age = BrainAgeRegressor(clf)
    gap = brain_age.compute_age_gap(age_pred, ya_test)
    logger.info(
        "Fold %d brain-age gap: mean=%.2f, std=%.2f",
        fold_idx + 1, float(np.mean(gap)), float(np.std(gap)),
    )

    gradcam_viz = GradCAMVisualizer()
    n_to_save = min(2, len(X_test))
    for i in range(n_to_save):
        pred_class = int(np.argmax(class_probs[i]))
        heatmap = gradcam_viz.compute_heatmap(clf.model, X_test[i], pred_class)
        overlay = gradcam_viz.overlay_on_image(X_test[i], heatmap)
        correctness = "correct" if pred_class == yc_test[i] else "incorrect"
        gradcam_viz.save_example(
            overlay,
            f"fold{fold_idx + 1}_ex{i}_{correctness}_pred{pred_class}_true{yc_test[i]}.png",
        )

    if fold_idx == n_folds - 1:
        clf.save(os.path.join(config.MODELS_DIR, "cnn_classifier_final.keras"))

    with open(paths["metrics"], "w") as f:
        json.dump({"with_gan": metrics_with_gan, "without_gan": metrics_without_gan}, f, indent=2)
    np.savez(
        paths["arrays"],
        y_true=yc_test,
        y_prob=class_probs,
        age_true=ya_test,
        age_pred=age_pred,
        train_loss=np.array(history.history.get("loss", [])),
        val_loss=np.array(history.history.get("val_loss", [])),
    )
    logger.info("Fold %d/%d complete, saved to %s", fold_idx + 1, n_folds, checkpoint_dir)

    return metrics_with_gan, metrics_without_gan


def aggregate_fold_results(n_folds, checkpoint_dir=CHECKPOINT_DIR):
    """Combine every completed fold's saved results into the final report.

    Reads each fold_{i}_metrics.json / fold_{i}_arrays.npz written by
    `run_single_fold`. Raises a clear error naming any fold that hasn't
    completed yet, rather than silently aggregating a partial result.

    Parameters
    ----------
    n_folds : int
        Total number of folds expected.
    checkpoint_dir : str

    Returns
    -------
    dict
        {"with_gan": [...fold metrics...], "without_gan": [...]}
    """
    fold_metrics_with_gan, fold_metrics_without_gan = [], []
    all_yc_test, all_class_probs, all_ya_test, all_age_pred = [], [], [], []
    last_train_loss, last_val_loss = np.array([]), np.array([])

    missing = []
    for fold_idx in range(n_folds):
        paths = _fold_paths(fold_idx, checkpoint_dir)
        if not os.path.exists(paths["metrics"]) or not os.path.exists(paths["arrays"]):
            missing.append(fold_idx + 1)
            continue
        with open(paths["metrics"]) as f:
            saved = json.load(f)
        fold_metrics_with_gan.append(saved["with_gan"])
        fold_metrics_without_gan.append(saved["without_gan"])

        arrays = np.load(paths["arrays"])
        all_yc_test.append(arrays["y_true"])
        all_class_probs.append(arrays["y_prob"])
        all_ya_test.append(arrays["age_true"])
        all_age_pred.append(arrays["age_pred"])
        if fold_idx == n_folds - 1:
            last_train_loss = arrays["train_loss"]
            last_val_loss = arrays["val_loss"]

    if missing:
        raise RuntimeError(
            f"Cannot aggregate: fold(s) {missing} have not completed yet. "
            f"Run them first, e.g. `python run_pipeline.py --stage fold --fold {missing[0]} ...`."
        )

    baseline_summary = Evaluator.summarize_folds(fold_metrics_without_gan)
    gan_summary = Evaluator.summarize_folds(fold_metrics_with_gan)
    comparison = {
        "with_gan": {k: v[0] for k, v in gan_summary.items() if k in ("accuracy", "f1_score")},
        "without_gan": {k: v[0] for k, v in baseline_summary.items() if k in ("accuracy", "f1_score")},
    }
    Evaluator.generate_report(fold_metrics_with_gan, gan_baseline_comparison=comparison)

    plot_data_path = os.path.join(config.RESULTS_DIR, "plot_data.npz")
    np.savez(
        plot_data_path,
        y_true=np.concatenate(all_yc_test),
        y_prob=np.concatenate(all_class_probs),
        age_true=np.concatenate(all_ya_test),
        age_pred=np.concatenate(all_age_pred),
        train_loss=last_train_loss,
        val_loss=last_val_loss,
    )
    logger.info("Saved plotting data to %s", plot_data_path)

    return {"with_gan": fold_metrics_with_gan, "without_gan": fold_metrics_without_gan}


def run_pipeline(n_subjects=config.N_SUBJECTS_SUBSET, n_folds=config.N_FOLDS, gan_epochs=config.GAN_EPOCHS):
    """Run the full end-to-end pipeline in one process and write the report.

    Convenience wrapper around `build_or_load_dataset` + `run_single_fold`
    (x n_folds) + `aggregate_fold_results` for when running everything in
    a single command is fine (e.g. a short local run). On an environment
    that can disconnect mid-run, prefer running the three stages as
    separate commands instead -- see run_pipeline.py --stage.

    Returns
    -------
    dict
        {"with_gan": [...fold metrics...], "without_gan": [...]}
    """
    X, y_class, y_age, subject_ids = build_or_load_dataset(n_subjects)
    n_classes = len(config.LABEL_CLASSES[config.LABEL_MODE])
    splits, effective_folds = compute_fold_splits(X, y_class, subject_ids, n_folds)

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        run_single_fold(
            fold_idx, effective_folds, X, y_class, y_age, subject_ids,
            train_idx, test_idx, gan_epochs=gan_epochs, n_classes=n_classes,
        )

    return aggregate_fold_results(effective_folds)

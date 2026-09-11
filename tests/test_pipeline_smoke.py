"""End-to-end smoke test on synthetic data.

Runs BEFORE touching the real (slow-to-download, ICA-heavy) dataset.
Preprocessing (EEGLoader/Preprocessor -> ICA) is skipped here since ICA on
random noise is meaningless; instead this generates synthetic
"EEG-shaped" epoch arrays directly and exercises CWT -> GAN -> CNN ->
Grad-CAM -> Evaluator, matching the real pipeline's data shapes at every
stage. Should run in well under a minute and catches integration bugs
(shape mismatches, Keras API misuse) before burning real time on the
actual dataset.

Run with:  pytest tests/test_pipeline_smoke.py  -v
"""

import os
import random
import sys

import numpy as np
import pytest
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.cnn_classifier import CNNClassifier
from src.cwt_transformer import CWTTransformer
from src.evaluator import Evaluator
from src.gan_augmenter import GANAugmenter
from src.gradcam_visualizer import GradCAMVisualizer

SEED = 42
N_SUBJECTS = 6
N_EPOCHS_PER_SUBJECT = 4
N_CHANNELS = config.N_CHANNELS_USED
SAMPLING_RATE = config.SAMPLING_RATE  # matches real ds004504 rate
EPOCH_SAMPLES = int(config.EPOCH_DURATION_SEC * SAMPLING_RATE)
IMAGE_SIZE = (32, 32)  # smaller than config.IMAGE_SIZE to keep the smoke test fast


@pytest.fixture(autouse=True)
def _set_seeds():
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)


def _make_synthetic_epochs():
    """Fake EEG-shaped epoch arrays + class/age labels, no real EEG I/O."""
    rng = np.random.default_rng(SEED)
    epochs, y_class, y_age, subject_ids = [], [], [], []
    for subj_idx in range(N_SUBJECTS):
        class_idx = subj_idx % 2  # balanced HC/AD
        age = float(rng.uniform(60, 85))
        for epoch_idx in range(N_EPOCHS_PER_SUBJECT):
            epoch = rng.normal(size=(N_CHANNELS, EPOCH_SAMPLES)).astype(np.float32)
            epochs.append(epoch)
            y_class.append(class_idx)
            y_age.append(age)
            subject_ids.append(f"sub-synthetic{subj_idx:03d}")
    return epochs, np.array(y_class), np.array(y_age, dtype=np.float32), np.array(subject_ids)


def test_cwt_transform_shapes():
    """CWTTransformer should produce correctly-shaped, [0,1]-normalized images."""
    epochs, *_ = _make_synthetic_epochs()
    transformer = CWTTransformer(image_size=IMAGE_SIZE)
    scalogram = transformer.transform(epochs[0], sampling_rate=SAMPLING_RATE)

    assert scalogram.shape == (IMAGE_SIZE[0], IMAGE_SIZE[1], N_CHANNELS)
    assert scalogram.min() >= 0.0
    assert scalogram.max() <= 1.0
    assert not np.isnan(scalogram).any()


def _build_synthetic_scalogram_dataset():
    epochs, y_class, y_age, subject_ids = _make_synthetic_epochs()
    transformer = CWTTransformer(image_size=IMAGE_SIZE)
    X = np.stack(
        [transformer.transform(e, sampling_rate=SAMPLING_RATE) for e in epochs]
    )
    return X.astype(np.float32), y_class, y_age, subject_ids


def test_gan_augmenter_runs_and_generates_correct_shape():
    """Conditional GANAugmenter should train without error and generate valid samples."""
    X, y_class, _, _ = _build_synthetic_scalogram_dataset()
    n_classes = len(config.LABEL_CLASSES[config.LABEL_MODE])

    gan = GANAugmenter(image_size=IMAGE_SIZE, n_channels=N_CHANNELS, n_classes=n_classes)
    history = gan.train(X, y_class, epochs=2, batch_size=4)
    assert len(history["g_loss"]) == 2
    assert len(history["d_loss"]) == 2
    assert all(np.isfinite(v) for v in history["g_loss"])
    assert all(np.isfinite(v) for v in history["d_loss"])

    synthetic = gan.generate_samples(5, class_label=0)
    assert synthetic.shape == (5, IMAGE_SIZE[0], IMAGE_SIZE[1], N_CHANNELS)
    assert not np.isnan(synthetic).any()

    per_sample_labels = np.array([0, 1, 0, 1, 0])
    synthetic_mixed = gan.generate_samples(5, class_label=per_sample_labels)
    assert synthetic_mixed.shape == (5, IMAGE_SIZE[0], IMAGE_SIZE[1], N_CHANNELS)


def _train_synthetic_cnn():
    """Helper: train a CNNClassifier on synthetic data, used by multiple tests."""
    X, y_class, y_age, _ = _build_synthetic_scalogram_dataset()
    n_classes = len(config.LABEL_CLASSES[config.LABEL_MODE])

    clf = CNNClassifier(input_shape=X.shape[1:], n_classes=n_classes, batch_size=4)
    history = clf.train(X, y_class, y_age, epochs=2)
    assert "loss" in history.history
    return clf, X, y_class, y_age


def test_cnn_classifier_train_predict_evaluate():
    """CNNClassifier dual-output train/predict/evaluate should run end-to-end."""
    clf, X, y_class, y_age = _train_synthetic_cnn()
    n_classes = len(config.LABEL_CLASSES[config.LABEL_MODE])

    class_probs, age_pred = clf.predict(X)
    assert class_probs.shape == (X.shape[0], n_classes)
    assert age_pred.shape == (X.shape[0],)
    assert np.allclose(class_probs.sum(axis=1), 1.0, atol=1e-4)

    results = clf.evaluate(X, y_class, y_age)
    assert "loss" in results


def test_gradcam_visualizer_runs():
    """Grad-CAM heatmap + overlay should run without error on a trained model."""
    clf, X, y_class, _ = _train_synthetic_cnn()
    viz = GradCAMVisualizer()

    heatmap = viz.compute_heatmap(clf.model, X[0], class_idx=int(y_class[0]))
    assert heatmap.ndim == 2
    assert heatmap.min() >= 0.0
    assert heatmap.max() <= 1.0 + 1e-6

    overlay = viz.overlay_on_image(X[0], heatmap)
    assert overlay.shape == (X.shape[1], X.shape[2], 3)
    assert overlay.dtype == np.uint8


def test_evaluator_metrics_in_valid_ranges():
    """Evaluator metrics should be computed and fall within valid ranges."""
    rng = np.random.default_rng(SEED)
    y_true = rng.integers(0, 2, size=20)
    y_pred = rng.integers(0, 2, size=20)
    y_prob = rng.uniform(size=20)
    age_true = rng.uniform(60, 85, size=20)
    age_pred = age_true + rng.normal(scale=2.0, size=20)

    acc = Evaluator.accuracy(y_true, y_pred)
    rec = Evaluator.recall(y_true, y_pred)
    f1 = Evaluator.f1_score(y_true, y_pred)
    auc = Evaluator.auc_roc(y_true, y_prob)
    mae = Evaluator.mae(age_true, age_pred)
    rmse = Evaluator.rmse(age_true, age_pred)
    r2 = Evaluator.r2(age_true, age_pred)
    corr = Evaluator.correlation(age_true, age_pred)

    assert 0.0 <= acc <= 1.0
    assert 0.0 <= rec <= 1.0
    assert 0.0 <= f1 <= 1.0
    assert 0.0 <= auc <= 1.0 or np.isnan(auc)
    assert mae >= 0.0
    assert rmse >= 0.0
    assert -1.0 <= corr <= 1.0
    assert isinstance(r2, float)

    report = Evaluator.generate_report(
        [
            {
                "accuracy": acc, "recall": rec, "f1_score": f1, "auc_roc": auc,
                "mae": mae, "rmse": rmse, "r2": r2, "correlation": corr,
            }
        ],
        out_path=os.path.join(config.RESULTS_DIR, "smoke_test_report.txt"),
    )
    assert "Metrics Report" in report


def test_full_synthetic_pipeline_end_to_end():
    """CWT -> GAN -> CNN -> Grad-CAM -> Evaluator, all chained, no exceptions."""
    X, y_class, y_age, subject_ids = _build_synthetic_scalogram_dataset()
    n_classes = len(config.LABEL_CLASSES[config.LABEL_MODE])

    gan = GANAugmenter(image_size=IMAGE_SIZE, n_channels=N_CHANNELS, n_classes=n_classes)
    gan.train(X, y_class, epochs=2, batch_size=4)
    synthetic = gan.generate_samples(4, class_label=0)
    X_aug = np.concatenate([X, synthetic], axis=0)
    yc_aug = np.concatenate([y_class, np.zeros(4, dtype=y_class.dtype)])
    ya_aug = np.concatenate([y_age, np.full(4, y_age[y_class == 0].mean(), dtype=y_age.dtype)])

    clf = CNNClassifier(input_shape=X_aug.shape[1:], n_classes=n_classes, batch_size=4)
    clf.train(X_aug, yc_aug, ya_aug, epochs=2)
    class_probs, age_pred = clf.predict(X)

    y_pred = np.argmax(class_probs, axis=1)
    acc = Evaluator.accuracy(y_class, y_pred)
    assert 0.0 <= acc <= 1.0

    viz = GradCAMVisualizer()
    heatmap = viz.compute_heatmap(clf.model, X[0], class_idx=int(y_pred[0]))
    overlay = viz.overlay_on_image(X[0], heatmap)
    assert overlay.shape[-1] == 3

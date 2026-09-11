"""Single-subject inference: mirrors the Phase 1 report's Sequence Diagram.

Fig 4.5 of the Phase 1 report describes a specific inference flow for the
"Clinician / Medical Professional" use case (Fig 4.3), distinct from the
k-fold training/evaluation flow in pipeline.py:

    User -> EEGLoader.load(path) -> Raw
         -> Preprocessor.filter() -> remove_artifacts() -> extract_epochs()
         -> CWTTransformer.transform() -> scalogram(s)
         -> CNNClassifier.predict() -> class probabilities + score
         -> GradCAMVisualizer.compute_heatmap() -> overlay
         -> BrainAgeRegressor.predict_age() + compute_age_gap()
         -> results returned to the user

This script is that flow, standalone, for one subject's EEG file and one
already-trained model checkpoint (see outputs/models/, written by
run_pipeline.py). It's meant as both a usable inference entry point and
the live demo script for a panel presentation.

Usage
-----
    python predict_subject.py --eeg-file data/sub-005/eeg/sub-005_task-eyesclosed_eeg.set \\
        --model outputs/models/cnn_classifier_final.keras --chronological-age 70
"""

import argparse
import json
import logging
import os

import numpy as np
import tensorflow as tf

import config
from src.brain_age_regressor import BrainAgeRegressor
from src.cnn_classifier import CNNClassifier
from src.cwt_transformer import CWTTransformer, resolve_channel_subset
from src.eeg_loader import EEGLoader
from src.gradcam_visualizer import GradCAMVisualizer
from src.preprocessor import Preprocessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def predict_subject(eeg_file, model_path, chronological_age=None, save_gradcam=True):
    """Run the full inference sequence diagram for one subject's EEG file.

    Parameters
    ----------
    eeg_file : str
        Path to the subject's EEGLAB .set file.
    model_path : str
        Path to a saved CNNClassifier checkpoint (.keras).
    chronological_age : float, optional
        If given, the brain-age gap is also reported (predicted - actual).
    save_gradcam : bool
        Whether to save an example Grad-CAM overlay per epoch classified.

    Returns
    -------
    dict
        Summary with per-epoch predictions and subject-level aggregates.
    """
    subject_id = os.path.basename(eeg_file).split("_")[0]
    class_names = config.LABEL_CLASSES[config.LABEL_MODE]

    logger.info("[1/5] EEGLoader.load(%s)", eeg_file)
    raw = EEGLoader(eeg_file).load()

    logger.info("[2/5] Preprocessor.filter() -> remove_artifacts() -> extract_epochs()")
    preprocessor = Preprocessor()
    filtered = preprocessor.filter(raw)
    cleaned = preprocessor.remove_artifacts(filtered)
    epochs = preprocessor.extract_epochs(cleaned)
    if len(epochs) == 0:
        raise RuntimeError(f"{subject_id}: 0 usable epochs, cannot run inference.")

    logger.info("[3/5] CWTTransformer.transform() -> scalograms")
    resolved_channels = resolve_channel_subset(epochs.ch_names)
    transformer = CWTTransformer()
    epoch_array = epochs.get_data(picks=resolved_channels)
    sfreq = epochs.info["sfreq"]
    scalograms = np.stack(
        [transformer.transform(epoch_array[i], sampling_rate=sfreq) for i in range(epoch_array.shape[0])]
    ).astype(np.float32)

    logger.info("[4/5] CNNClassifier.predict() -> class probabilities + brain age")
    clf = CNNClassifier(n_classes=len(class_names))
    clf.load(model_path)
    class_probs, age_pred = clf.predict(scalograms)
    mean_probs = class_probs.mean(axis=0)
    predicted_class_idx = int(np.argmax(mean_probs))
    predicted_class = class_names[predicted_class_idx]
    mean_age_pred = float(age_pred.mean())

    logger.info("[5/5] GradCAMVisualizer + BrainAgeRegressor.compute_age_gap()")
    gradcam_paths = []
    if save_gradcam:
        viz = GradCAMVisualizer()
        n_examples = min(3, scalograms.shape[0])
        for i in range(n_examples):
            pred_i = int(np.argmax(class_probs[i]))
            heatmap = viz.compute_heatmap(clf.model, scalograms[i], pred_i)
            overlay = viz.overlay_on_image(scalograms[i], heatmap)
            path = viz.save_example(overlay, f"inference_{subject_id}_epoch{i}_pred{class_names[pred_i]}.png")
            gradcam_paths.append(path)

    result = {
        "subject_id": subject_id,
        "n_epochs": int(scalograms.shape[0]),
        "predicted_class": predicted_class,
        "class_probabilities": {name: float(p) for name, p in zip(class_names, mean_probs)},
        "predicted_brain_age": mean_age_pred,
        "gradcam_examples": gradcam_paths,
    }

    if chronological_age is not None:
        gap = BrainAgeRegressor.compute_age_gap(mean_age_pred, chronological_age)
        result["chronological_age"] = float(chronological_age)
        result["brain_age_gap"] = float(gap)

    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eeg-file", required=True, help="Path to subject's .set file")
    parser.add_argument(
        "--model", default=os.path.join(config.MODELS_DIR, "cnn_classifier_final.keras"),
        help="Path to a trained CNNClassifier checkpoint",
    )
    parser.add_argument("--chronological-age", type=float, default=None, help="Known age, for brain-age gap")
    parser.add_argument("--no-gradcam", action="store_true", help="Skip saving Grad-CAM overlays")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        raise FileNotFoundError(
            f"No trained model at {args.model} -- run `python run_pipeline.py` first, "
            "which saves a checkpoint to outputs/models/cnn_classifier_final.keras."
        )

    result = predict_subject(
        args.eeg_file, args.model,
        chronological_age=args.chronological_age,
        save_gradcam=not args.no_gradcam,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

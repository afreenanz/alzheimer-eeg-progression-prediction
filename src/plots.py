"""Generates the standard results-section plots from a completed run.

Reads outputs/results/plot_data.npz (written by src.pipeline.run_pipeline)
and produces:
  - training loss curve
  - ROC curve
  - confusion matrix
  - predicted-vs-actual brain age scatter plot
all saved as PNGs under outputs/results/.

Usage
-----
    python -m src.plots
"""

import logging
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve

import config

logger = logging.getLogger(__name__)


def plot_training_loss(train_loss, val_loss, out_path):
    """Plot train/val loss curves from the last fold's training history."""
    plt.figure(figsize=(6, 4))
    plt.plot(train_loss, label="train loss")
    if len(val_loss):
        plt.plot(val_loss, label="val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss (last fold)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info("Saved %s", out_path)


def plot_roc_curve(y_true, y_prob, out_path):
    """Plot the ROC curve for the positive class."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, label="ROC curve")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (pooled across folds)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info("Saved %s", out_path)


def plot_confusion_matrix(y_true, y_pred, class_names, out_path):
    """Plot a confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(4.5, 4))
    plt.imshow(cm, cmap="Blues")
    plt.colorbar()
    plt.xticks(range(len(class_names)), class_names)
    plt.yticks(range(len(class_names)), class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix (pooled across folds)")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info("Saved %s", out_path)


def plot_age_scatter(age_true, age_pred, out_path):
    """Plot predicted vs. actual brain age with the y=x reference line."""
    plt.figure(figsize=(5, 5))
    plt.scatter(age_true, age_pred, alpha=0.6)
    lo = min(age_true.min(), age_pred.min())
    hi = max(age_true.max(), age_pred.max())
    plt.plot([lo, hi], [lo, hi], linestyle="--", color="gray", label="y = x")
    plt.xlabel("Chronological Age")
    plt.ylabel("Predicted (Brain) Age")
    plt.title("Predicted vs. Actual Brain Age (pooled across folds)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info("Saved %s", out_path)


def generate_all_plots(plot_data_path=None, out_dir=None):
    """Load saved run data and generate all standard result plots.

    Parameters
    ----------
    plot_data_path : str, optional
        Defaults to outputs/results/plot_data.npz.
    out_dir : str, optional
        Defaults to outputs/results/.
    """
    plot_data_path = plot_data_path or os.path.join(config.RESULTS_DIR, "plot_data.npz")
    out_dir = out_dir or config.RESULTS_DIR
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(plot_data_path):
        raise FileNotFoundError(
            f"{plot_data_path} not found -- run the pipeline first "
            "(python run_pipeline.py)."
        )

    data = np.load(plot_data_path)
    y_true, y_prob = data["y_true"], data["y_prob"]
    age_true, age_pred = data["age_true"], data["age_pred"]
    train_loss, val_loss = data["train_loss"], data["val_loss"]

    y_pred = np.argmax(y_prob, axis=1)
    y_prob_positive = y_prob[:, 1] if y_prob.shape[1] == 2 else y_prob.max(axis=1)
    class_names = config.LABEL_CLASSES[config.LABEL_MODE]

    plot_training_loss(train_loss, val_loss, os.path.join(out_dir, "training_loss.png"))
    plot_roc_curve(y_true, y_prob_positive, os.path.join(out_dir, "roc_curve.png"))
    plot_confusion_matrix(y_true, y_pred, class_names, os.path.join(out_dir, "confusion_matrix.png"))
    plot_age_scatter(age_true, age_pred, os.path.join(out_dir, "brain_age_scatter.png"))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    generate_all_plots()

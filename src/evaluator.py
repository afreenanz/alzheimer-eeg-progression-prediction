"""Classification and regression metrics, plus a consolidated report."""

import logging
import os

import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    recall_score,
    roc_auc_score,
)

import config

logger = logging.getLogger(__name__)


class Evaluator:
    """Thin wrappers around sklearn/scipy metrics for both CNN heads."""

    @staticmethod
    def accuracy(y_true, y_pred):
        """Classification accuracy."""
        return accuracy_score(y_true, y_pred)

    @staticmethod
    def recall(y_true, y_pred, average="binary"):
        """Classification recall."""
        return recall_score(y_true, y_pred, average=average, zero_division=0)

    @staticmethod
    def f1_score(y_true, y_pred, average="binary"):
        """Classification F1 score."""
        return f1_score(y_true, y_pred, average=average, zero_division=0)

    @staticmethod
    def auc_roc(y_true, y_prob):
        """AUC-ROC. `y_prob` is the predicted probability of the positive class."""
        try:
            return roc_auc_score(y_true, y_prob)
        except ValueError as exc:
            logger.warning("AUC-ROC undefined (%s); returning nan.", exc)
            return float("nan")

    @staticmethod
    def mae(y_true, y_pred):
        """Mean absolute error for brain-age regression."""
        return mean_absolute_error(y_true, y_pred)

    @staticmethod
    def rmse(y_true, y_pred):
        """Root mean squared error for brain-age regression."""
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))

    @staticmethod
    def r2(y_true, y_pred):
        """R-squared for brain-age regression."""
        return r2_score(y_true, y_pred)

    @staticmethod
    def correlation(y_true, y_pred):
        """Pearson correlation coefficient between predicted and true age."""
        if len(y_true) < 2:
            return float("nan")
        r, _ = pearsonr(y_true, y_pred)
        return r

    @staticmethod
    def summarize_folds(fold_metrics):
        """Aggregate a list of per-fold metric dicts into mean +/- std.

        Parameters
        ----------
        fold_metrics : list of dict
            Each dict maps metric name -> value for one fold.

        Returns
        -------
        dict
            metric name -> (mean, std).
        """
        keys = fold_metrics[0].keys() if fold_metrics else []
        summary = {}
        for k in keys:
            values = [m[k] for m in fold_metrics if not np.isnan(m.get(k, np.nan))]
            if values:
                summary[k] = (float(np.mean(values)), float(np.std(values)))
            else:
                summary[k] = (float("nan"), float("nan"))
        return summary

    @staticmethod
    def generate_report(fold_metrics, out_path=None, gan_baseline_comparison=None):
        """Write a consolidated metrics report to outputs/results/.

        Parameters
        ----------
        fold_metrics : list of dict
            Per-fold metrics (classification + regression combined).
        out_path : str, optional
            Destination path; defaults to
            outputs/results/metrics_report.txt.
        gan_baseline_comparison : dict, optional
            {"with_gan": {...}, "without_gan": {...}} accuracy/F1 summary
            for the GAN-vs-no-GAN sanity check.

        Returns
        -------
        str
            The report text (also written to disk).
        """
        out_path = out_path or os.path.join(config.RESULTS_DIR, "metrics_report.txt")
        summary = Evaluator.summarize_folds(fold_metrics)

        lines = []
        lines.append("=" * 70)
        lines.append("EEG-Based Alzheimer's Progression Prediction -- Metrics Report")
        lines.append("=" * 70)
        lines.append(f"Label mode: {config.LABEL_MODE} (see config.py / README for rationale)")
        lines.append(f"Folds: {len(fold_metrics)} (subject-level GroupKFold)")
        lines.append("")
        lines.append("-- Classification (mean +/- std across folds) --")
        for k in ("accuracy", "recall", "f1_score", "auc_roc"):
            if k in summary:
                mean, std = summary[k]
                lines.append(f"  {k:12s}: {mean:.4f} +/- {std:.4f}")
        lines.append("")
        lines.append("-- Brain Age Regression (mean +/- std across folds) --")
        for k in ("mae", "rmse", "r2", "correlation"):
            if k in summary:
                mean, std = summary[k]
                lines.append(f"  {k:12s}: {mean:.4f} +/- {std:.4f}")

        if gan_baseline_comparison:
            lines.append("")
            lines.append("-- GAN Augmentation vs. Baseline (no augmentation) --")
            for condition, metrics in gan_baseline_comparison.items():
                lines.append(f"  {condition}:")
                for k, v in metrics.items():
                    lines.append(f"    {k:12s}: {v:.4f}")

        lines.append("=" * 70)
        report_text = "\n".join(lines)

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            f.write(report_text + "\n")
        logger.info("Wrote metrics report to %s", out_path)
        print(report_text)
        return report_text

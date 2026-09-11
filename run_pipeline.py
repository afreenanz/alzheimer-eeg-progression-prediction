"""Main entry point: runs the full EEG Alzheimer's pipeline end-to-end.

    python run_pipeline.py --n_subjects 20 --folds 3 --gan_epochs 50

For long runs on an environment that can disconnect mid-run (e.g.
free-tier Colab), run it as separate stages instead -- each is its own
short-lived command, so a disconnect only costs the one stage that was
running:

    python run_pipeline.py --stage build --n_subjects 65
    python run_pipeline.py --stage fold --fold 1 --n_subjects 65 --folds 3 --gan_epochs 50
    python run_pipeline.py --stage fold --fold 2 --n_subjects 65 --folds 3 --gan_epochs 50
    python run_pipeline.py --stage fold --fold 3 --n_subjects 65 --folds 3 --gan_epochs 50
    python run_pipeline.py --stage aggregate --folds 3

Re-running the same `--stage fold --fold N` command after a disconnect
is safe -- a fold that already finished is loaded from disk instead of
retrained.
"""

import argparse
import logging
import random

import numpy as np
import tensorflow as tf

import config
from src.pipeline import (
    aggregate_fold_results,
    build_or_load_dataset,
    compute_fold_splits,
    run_pipeline,
    run_single_fold,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def set_seeds(seed):
    """Fix random seeds across numpy, python's random, and TensorFlow."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n_subjects", type=int, default=config.N_SUBJECTS_SUBSET)
    parser.add_argument("--folds", type=int, default=config.N_FOLDS)
    parser.add_argument("--gan_epochs", type=int, default=config.GAN_EPOCHS)
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    parser.add_argument(
        "--stage", choices=["all", "build", "fold", "aggregate"], default="all",
        help="'all' runs everything in one process (default, fine for short local runs). "
             "'build'/'fold'/'aggregate' run one resumable stage at a time (recommended on Colab).",
    )
    parser.add_argument("--fold", type=int, default=None, help="1-indexed fold number, required for --stage fold")
    parser.add_argument("--force", action="store_true", help="--stage fold: retrain even if already completed")
    args = parser.parse_args()

    set_seeds(args.seed)

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        logger.info("GPU(s) available: %s", gpus)
    else:
        logger.info("No GPU detected; running on CPU (will be slower).")

    if args.stage == "all":
        logger.info(
            "Starting pipeline (single process): n_subjects=%d folds=%d gan_epochs=%d seed=%d",
            args.n_subjects, args.folds, args.gan_epochs, args.seed,
        )
        run_pipeline(n_subjects=args.n_subjects, n_folds=args.folds, gan_epochs=args.gan_epochs)
        logger.info("Pipeline complete. See outputs/results/metrics_report.txt")

    elif args.stage == "build":
        logger.info("Building/caching dataset for %d subjects...", args.n_subjects)
        X, *_ = build_or_load_dataset(args.n_subjects)
        logger.info("Dataset ready: %d samples. Now run --stage fold for each fold.", len(X))

    elif args.stage == "fold":
        if args.fold is None:
            parser.error("--stage fold requires --fold N (1-indexed)")
        X, y_class, y_age, subject_ids = build_or_load_dataset(args.n_subjects)
        splits, effective_folds = compute_fold_splits(X, y_class, subject_ids, args.folds)
        if not (1 <= args.fold <= effective_folds):
            parser.error(f"--fold must be between 1 and {effective_folds} (got {args.fold})")
        train_idx, test_idx = splits[args.fold - 1]
        run_single_fold(
            args.fold - 1, effective_folds, X, y_class, y_age, subject_ids,
            train_idx, test_idx, gan_epochs=args.gan_epochs, force=args.force,
        )
        logger.info(
            "Fold %d/%d done. Once all folds are complete, run --stage aggregate.",
            args.fold, effective_folds,
        )

    elif args.stage == "aggregate":
        aggregate_fold_results(args.folds)
        logger.info("Pipeline complete. See outputs/results/metrics_report.txt")


if __name__ == "__main__":
    main()

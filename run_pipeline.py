"""Main entry point: runs the full EEG Alzheimer's pipeline end-to-end.

    python run_pipeline.py --n_subjects 20 --folds 3 --gan_epochs 50
"""

import argparse
import logging
import random

import numpy as np
import tensorflow as tf

import config
from src.pipeline import run_pipeline

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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n_subjects", type=int, default=config.N_SUBJECTS_SUBSET)
    parser.add_argument("--folds", type=int, default=config.N_FOLDS)
    parser.add_argument("--gan_epochs", type=int, default=config.GAN_EPOCHS)
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    args = parser.parse_args()

    set_seeds(args.seed)

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        logger.info("GPU(s) available: %s", gpus)
    else:
        logger.info("No GPU detected; running on CPU (will be slower).")

    logger.info(
        "Starting pipeline: n_subjects=%d folds=%d gan_epochs=%d seed=%d",
        args.n_subjects, args.folds, args.gan_epochs, args.seed,
    )
    run_pipeline(n_subjects=args.n_subjects, n_folds=args.folds, gan_epochs=args.gan_epochs)
    logger.info("Pipeline complete. See outputs/results/metrics_report.txt")


if __name__ == "__main__":
    main()

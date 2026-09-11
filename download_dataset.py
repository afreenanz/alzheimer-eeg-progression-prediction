"""Download OpenNeuro ds004504 (Miltiadous et al. AD/FTD/HC EEG dataset).

Tries, in order: `aws s3 sync` against the public OpenNeuro S3 mirror
(no credentials needed for public buckets), then the OpenNeuro CLI
(`openneuro-cli`) if installed. Prints manual-download instructions if
neither tool is available.

Usage
-----
    python download_dataset.py [--n-subjects N]

By default downloads participants.tsv, then uses it to pick a
class-balanced subset of N subjects (via src.eeg_loader.get_labels +
subset_subjects) before syncing their eeg/ directories. This matters
because ds004504's subjects are grouped sequentially by diagnosis
(sub-001..036 = AD, sub-037..065 = HC, sub-066..088 = FTD) -- naively
downloading "the first N subjects" would silently pull only AD subjects.
Pass --n-subjects -1 to download the full dataset (88 subjects).

Usage
-----
    python download_dataset.py [--n-subjects N]
"""

import argparse
import logging
import shutil
import subprocess
import sys

import config

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _run(cmd):
    logger.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0


def download_via_s3(n_subjects):
    if shutil.which("aws") is None:
        return False

    logger.info("Downloading participants.tsv via aws s3 sync...")
    ok = _run(
        [
            "aws", "s3", "cp", "--no-sign-request",
            f"{config.OPENNEURO_S3_BUCKET}/participants.tsv",
            config.PARTICIPANTS_TSV,
        ]
    )
    if not ok:
        return False

    if n_subjects == -1:
        return _run(
            [
                "aws", "s3", "sync", "--no-sign-request",
                config.OPENNEURO_S3_BUCKET, config.DATA_DIR,
            ]
        )

    # Pick a class-balanced subset using participants.tsv, not the first
    # N subjects by index -- see module docstring.
    from src.eeg_loader import get_labels, subset_subjects

    labels = get_labels()
    subset = subset_subjects(labels, n_subjects=n_subjects)
    logger.info("Selected %d class-balanced subjects to download: %s", len(subset), sorted(subset))

    for subject_id in sorted(subset):
        ok = _run(
            [
                "aws", "s3", "sync", "--no-sign-request",
                f"{config.OPENNEURO_S3_BUCKET}/{subject_id}",
                f"{config.DATA_DIR}/{subject_id}",
            ]
        )
        if not ok:
            logger.warning("Failed to sync %s; continuing with remaining subjects.", subject_id)
    return True


def download_via_openneuro_cli(n_subjects):
    if shutil.which("openneuro-cli") is None and shutil.which("openneuro") is None:
        return False
    cli = "openneuro-cli" if shutil.which("openneuro-cli") else "openneuro"
    return _run([cli, "download", "--snapshot", "1.0.0", config.OPENNEURO_DATASET_ID, config.DATA_DIR])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-subjects", type=int, default=config.N_SUBJECTS_SUBSET,
        help="Number of subjects to download (-1 for all 88).",
    )
    args = parser.parse_args()

    if download_via_s3(args.n_subjects):
        logger.info("Download complete via aws s3.")
        return
    if download_via_openneuro_cli(args.n_subjects):
        logger.info("Download complete via openneuro-cli.")
        return

    print(
        "\nCould not find `aws` or `openneuro-cli` on PATH.\n"
        "Install one of:\n"
        "  pip install awscli   # then: aws s3 sync --no-sign-request "
        f"{config.OPENNEURO_S3_BUCKET} {config.DATA_DIR}\n"
        "  npm install -g @openneuro/cli\n"
        "Or download manually from:\n"
        f"  {config.OPENNEURO_HTTPS_BASE}\n"
        f"and place the extracted BIDS dataset under {config.DATA_DIR}/\n"
    )
    sys.exit(1)


if __name__ == "__main__":
    main()

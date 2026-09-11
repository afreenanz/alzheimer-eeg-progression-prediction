"""Load raw EEG recordings and subject labels for ds004504.

Also hosts `get_labels`, the single swappable function that maps subject
IDs to diagnostic class + chronological age. See config.py's module
docstring for the AD-vs-HC label substitution rationale: the report's
sMCI/pMCI design cannot be built from ds004504, which instead provides
AD / FTD / HC groups. Only this function needs to change if real
sMCI/pMCI-labeled data becomes available -- the rest of the pipeline
consumes its output (class label + age) generically.
"""

import logging
import os

import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)


class EEGLoadError(Exception):
    """Raised when an EEG file cannot be loaded or fails validation."""


class EEGLoader:
    """Loads a single subject's EEG recording from EEGLAB .set/.fdt files.

    Attributes
    ----------
    file_path : str
        Path to the subject's .set file.
    sampling_rate : float
        Expected sampling rate in Hz, used only for validation logging.
    """

    def __init__(self, file_path, sampling_rate=config.SAMPLING_RATE):
        self.file_path = file_path
        self.sampling_rate = sampling_rate

    def validate_format(self):
        """Check the file exists, has a plausible channel count and rate.

        Returns
        -------
        bool
            True if the file passes basic sanity checks.

        Raises
        ------
        EEGLoadError
            If the file is missing or clearly malformed.
        """
        if not os.path.exists(self.file_path):
            raise EEGLoadError(f"EEG file not found: {self.file_path}")
        if not self.file_path.endswith(".set"):
            raise EEGLoadError(
                f"Expected an EEGLAB .set file, got: {self.file_path}"
            )
        fdt_path = self.file_path.replace(".set", ".fdt")
        if not os.path.exists(fdt_path):
            logger.warning(
                "No .fdt sidecar found next to %s -- data may be embedded "
                "in the .set file itself; continuing.",
                self.file_path,
            )
        return True

    def load(self):
        """Load the recording as an MNE Raw object.

        Returns
        -------
        mne.io.Raw
            The loaded raw EEG recording.

        Raises
        ------
        EEGLoadError
            If the file is missing, corrupt, or has an unexpected number
            of channels / sampling rate.
        """
        import mne

        self.validate_format()
        try:
            raw = mne.io.read_raw_eeglab(
                self.file_path, preload=True, verbose="ERROR"
            )
        except Exception as exc:  # noqa: BLE001 - surface as EEGLoadError
            raise EEGLoadError(
                f"Failed to read EEGLAB file {self.file_path}: {exc}"
            ) from exc

        n_channels = len(raw.ch_names)
        if n_channels != config.N_CHANNELS_EXPECTED:
            logger.warning(
                "%s: expected %d channels, found %d. Continuing, but "
                "channel-subset selection downstream will adapt to what "
                "is actually present.",
                self.file_path,
                config.N_CHANNELS_EXPECTED,
                n_channels,
            )
        actual_sfreq = raw.info["sfreq"]
        if actual_sfreq <= 0:
            raise EEGLoadError(
                f"{self.file_path}: invalid sampling rate {actual_sfreq}"
            )
        if abs(actual_sfreq - self.sampling_rate) > 1e-6:
            logger.info(
                "%s: sampling rate is %.1f Hz (config expected %.1f Hz); "
                "using the file's actual rate.",
                self.file_path,
                actual_sfreq,
                self.sampling_rate,
            )
            self.sampling_rate = actual_sfreq
        return raw


def load_subject_safely(file_path, sampling_rate=config.SAMPLING_RATE):
    """Load one subject, returning None (and logging) instead of raising.

    Used by the batch pipeline so a single corrupt/missing subject does
    not crash the whole run.

    Parameters
    ----------
    file_path : str
        Path to the subject's .set file.
    sampling_rate : float
        Expected sampling rate.

    Returns
    -------
    mne.io.Raw or None
        The loaded raw recording, or None if loading failed.
    """
    loader = EEGLoader(file_path, sampling_rate)
    try:
        return loader.load()
    except EEGLoadError as exc:
        logger.error("Skipping subject at %s: %s", file_path, exc)
        return None


def get_labels(participants_tsv=config.PARTICIPANTS_TSV, label_mode=config.LABEL_MODE):
    """Build a subject_id -> (class_label, age) map from participants.tsv.

    This is the single swappable label-loading step referenced in
    config.py. Today it implements the AD-vs-HC substitution decision
    (LABEL_MODE == "AD_HC"): FTD subjects are excluded so the task stays
    strictly binary, matching the report's binary sMCI/pMCI design intent.
    A LABEL_MODE == "SMCI_PMCI" branch is stubbed for when real
    progression-labeled data is available -- swap only this function.

    Parameters
    ----------
    participants_tsv : str
        Path to the BIDS participants.tsv file.
    label_mode : str
        Either "AD_HC" (implemented) or "SMCI_PMCI" (not yet available).

    Returns
    -------
    dict
        Mapping of subject_id (e.g. "sub-001") -> dict with keys
        "class_label" (str, one of config.LABEL_CLASSES[label_mode]),
        "class_idx" (int), and "age" (float).
    """
    if label_mode == "SMCI_PMCI":
        raise NotImplementedError(
            "SMCI_PMCI label mode is not available for ds004504 -- this "
            "dataset has no progression labels. See config.py docstring. "
            "Swap in a real sMCI/pMCI-labeled participants file and "
            "implement this branch when such data becomes available."
        )
    if label_mode != "AD_HC":
        raise ValueError(f"Unknown LABEL_MODE: {label_mode}")

    if not os.path.exists(participants_tsv):
        raise EEGLoadError(f"participants.tsv not found at {participants_tsv}")

    df = pd.read_csv(participants_tsv, sep="\t")
    df.columns = [c.strip() for c in df.columns]

    id_col = "participant_id" if "participant_id" in df.columns else df.columns[0]
    group_col = next(
        (c for c in df.columns if c.lower() in ("group", "diagnosis", "dx")), None
    )
    age_col = next((c for c in df.columns if c.lower() == "age"), None)
    if group_col is None or age_col is None:
        raise EEGLoadError(
            f"participants.tsv missing expected 'Group'/'Age' columns; "
            f"found columns: {list(df.columns)}"
        )

    classes = config.LABEL_CLASSES["AD_HC"]  # ["HC", "AD"]
    labels = {}
    for _, row in df.iterrows():
        group = str(row[group_col]).strip().upper()
        if group not in ("A", "AD", "C", "CN", "HC"):
            # Skips FTD ("F") and any unrecognized group -- see docstring:
            # this keeps the task strictly binary (AD vs HC).
            continue
        class_label = "AD" if group in ("A", "AD") else "HC"
        subject_id = str(row[id_col]).strip()
        age = row[age_col]
        if pd.isna(age):
            logger.warning("%s: missing age in participants.tsv, skipping.", subject_id)
            continue
        labels[subject_id] = {
            "class_label": class_label,
            "class_idx": classes.index(class_label),
            "age": float(age),
        }
    logger.info(
        "Loaded labels for %d subjects (AD=%d, HC=%d) from %s",
        len(labels),
        sum(1 for v in labels.values() if v["class_label"] == "AD"),
        sum(1 for v in labels.values() if v["class_label"] == "HC"),
        participants_tsv,
    )
    return labels


def subset_subjects(labels, n_subjects=config.N_SUBJECTS_SUBSET, seed=config.RANDOM_SEED):
    """Pick a roughly class-balanced subset of subjects for the scoped build.

    Parameters
    ----------
    labels : dict
        Output of `get_labels`.
    n_subjects : int
        Total number of subjects to keep.
    seed : int
        Random seed for reproducible subsampling.

    Returns
    -------
    dict
        A subset of `labels` with `n_subjects` entries, balanced as evenly
        as possible between classes.
    """
    rng = np.random.default_rng(seed)
    by_class = {}
    for sid, info in labels.items():
        by_class.setdefault(info["class_label"], []).append(sid)
    for sids in by_class.values():
        rng.shuffle(sids)

    n_classes = len(by_class)
    per_class = n_subjects // n_classes
    chosen = []
    for sids in by_class.values():
        chosen.extend(sids[:per_class])
    remaining = n_subjects - len(chosen)
    if remaining > 0:
        leftovers = [
            sid for sids in by_class.values() for sid in sids[per_class:]
        ]
        rng.shuffle(leftovers)
        chosen.extend(leftovers[:remaining])

    return {sid: labels[sid] for sid in chosen}

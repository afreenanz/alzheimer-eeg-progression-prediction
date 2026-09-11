"""Central configuration for the EEG-based Alzheimer's progression pipeline.

All hyperparameters, paths, and scale-reduction knobs live here so the
pipeline can be scaled up (more subjects, full LOSO, more GAN epochs,
224x224 images) by changing values in exactly one place.

LABEL SUBSTITUTION NOTE
------------------------
The Phase-1 design report specifies classifying sMCI vs pMCI (Stable vs
Progressive Mild Cognitive Impairment) using OpenNeuro ds004504. That
dataset does NOT contain sMCI/pMCI labels -- it contains three diagnostic
groups: Alzheimer's Disease (AD, 36 subjects), Frontotemporal Dementia
(FTD, 23 subjects), and Healthy Control (HC, 29 subjects).

This implementation substitutes AD vs HC as the primary binary
classification task, since it is the cleanest binary split available in
this dataset and matches the report's binary-classification design. The
label-loading step is isolated to a single function
(`src.eeg_loader.get_labels`) driven by the `LABEL_MODE` flag below -- if
real sMCI/pMCI-labeled data becomes available later, only that one
function needs to change.
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
PARTICIPANTS_TSV = os.path.join(DATA_DIR, "participants.tsv")

OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
SCALOGRAMS_DIR = os.path.join(OUTPUTS_DIR, "scalograms")
GRADCAM_DIR = os.path.join(OUTPUTS_DIR, "gradcam")
MODELS_DIR = os.path.join(OUTPUTS_DIR, "models")
RESULTS_DIR = os.path.join(OUTPUTS_DIR, "results")

for _d in (DATA_DIR, SCALOGRAMS_DIR, GRADCAM_DIR, MODELS_DIR, RESULTS_DIR):
    os.makedirs(_d, exist_ok=True)

# ---------------------------------------------------------------------------
# Label mode -- see module docstring above.
# "AD_HC" is the only mode implemented today. "SMCI_PMCI" is a documented
# placeholder for when real progression-labeled data becomes available.
# ---------------------------------------------------------------------------
LABEL_MODE = "AD_HC"
LABEL_CLASSES = {"AD_HC": ["HC", "AD"], "SMCI_PMCI": ["sMCI", "pMCI"]}

# ---------------------------------------------------------------------------
# Dataset scale (12-hour scoped build). Bump these up once more compute /
# time is available -- nothing else in the pipeline needs to change.
# ---------------------------------------------------------------------------
N_SUBJECTS_SUBSET = 20          # subset of the full 88 subjects
N_FOLDS = 3                     # reduced-fold approximation of full LOSO
                                 # (full LOSO == GroupKFold(n_splits=len(subjects)))
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# EEG acquisition (per ds004504 spec)
# ---------------------------------------------------------------------------
SAMPLING_RATE = 500              # Hz, ds004504 recordings are sampled at 500 Hz
N_CHANNELS_EXPECTED = 19         # 10-20 system, per dataset spec

# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
BANDPASS_RANGE = (0.5, 45.0)     # Hz
ICA_N_COMPONENTS = 15            # capped to available channels at runtime
EPOCH_DURATION_SEC = 4.0
EPOCH_OVERLAP_SEC = 2.0

# ---------------------------------------------------------------------------
# CWT / scalogram generation
# ---------------------------------------------------------------------------
WAVELET_TYPE = "morl"            # Morlet wavelet
CWT_FREQ_RANGE = (1, 40)         # Hz
# Full report spec is 224x224; reduced to 64x64 for the 12-hour build to
# keep CWT + CNN + GAN training tractable on CPU / shared Colab GPU.
IMAGE_SIZE = (64, 64)

# Channel-selection strategy (option (c) from the module spec): use a fixed
# subset of diagnostically-relevant frontal/temporal/occipital channels,
# stacked as image depth (channels) for the CNN input, instead of
# averaging across all 19 channels or using all of them. Names are
# resolved against the actual montage at load time (see
# src/cwt_transformer.py::resolve_channel_subset) since not every montage
# uses these exact labels.
CHANNEL_SUBSET_PREFERRED = ["Fp1", "Fp2", "F7", "F8", "T3", "T4"]
N_CHANNELS_USED = len(CHANNEL_SUBSET_PREFERRED)

# ---------------------------------------------------------------------------
# GAN augmentation (DCGAN-style, shallow since images are only 64x64)
# ---------------------------------------------------------------------------
GAN_EPOCHS = 50                  # capped for the 12-hour build; production
                                  # GANs need far more (500-1000+) to converge
GAN_BATCH_SIZE = 16
GAN_LATENT_DIM = 100
GAN_LEARNING_RATE = 2e-4

# ---------------------------------------------------------------------------
# CNN classifier + regressor (dual-output)
# ---------------------------------------------------------------------------
CNN_LEARNING_RATE = 1e-3
CNN_BATCH_SIZE = 16
CNN_EPOCHS = 30
CNN_LOSS_WEIGHTS = {"classification": 1.0, "regression": 0.5}

# ---------------------------------------------------------------------------
# Grad-CAM
# ---------------------------------------------------------------------------
GRADCAM_ALPHA = 0.4               # overlay transparency
GRADCAM_N_EXAMPLES = 6            # saved example overlays per run

# ---------------------------------------------------------------------------
# OpenNeuro dataset
# ---------------------------------------------------------------------------
OPENNEURO_DATASET_ID = "ds004504"
OPENNEURO_S3_BUCKET = "s3://openneuro.org/ds004504"
OPENNEURO_HTTPS_BASE = "https://openneuro.org/crn/datasets/ds004504/download"

# EEG-Based Alzheimer's Progression Prediction

A Python pipeline that classifies EEG scalograms into diagnostic groups and
estimates brain age, with GAN-based data augmentation and Grad-CAM
explainability. Implements the Phase-1 design report for a BE Major/Mini
Project (BMSCE, VTU), **at reduced scale** for a ~12-hour build budget with
no dedicated GPU beyond Google Colab.

The goal of this build was correctness and completeness of every module
(EEG loading, preprocessing, CWT scalograms, GAN augmentation, dual-output
CNN, Grad-CAM, brain-age regression, subject-independent evaluation) over
training a state-of-the-art model. Every reduction from the full report
spec is a config value, documented below and in-code, so scaling up is a
matter of changing numbers, not rewriting modules.

## What was built

- `src/eeg_loader.py` -- `EEGLoader` (load/validate EEGLAB `.set/.fdt`
  files via MNE) + `get_labels()`, the single swappable label-loading
  function (see **Label substitution** below).
- `src/preprocessor.py` -- `Preprocessor`: bandpass filter + average
  re-reference, ICA-based artifact removal (EOG-proxy detection with a
  kurtosis fallback), fixed 4s/2s-overlap epoching with per-epoch
  baseline correction and per-channel normalization (matches Fig 4.1
  Stage 2's four sub-steps: filtering, artifact removal, re-referencing,
  normalization).
- `src/cwt_transformer.py` -- `CWTTransformer`: per-channel Morlet CWT ->
  resized, normalized multi-channel scalogram images.
- `src/gan_augmenter.py` -- `GANAugmenter`: **conditional** shallow DCGAN
  (`Generator`, `Discriminator`), label-conditioned per Fig 4.2's
  "Conditional GAN Training loop" and the Luo et al. cGAN citation
  (Sec 2.5) -- a single GAN instance is trained across both classes and
  sampled on demand for whichever class needs balancing.
- `src/cnn_classifier.py` -- `CNNClassifier`: shared conv backbone with a
  classification head (AD/HC) and a regression head (brain age).
- `src/gradcam_visualizer.py` -- `GradCAMVisualizer`: standard Grad-CAM
  over the classification head, with heatmap overlay rendering.
- `src/brain_age_regressor.py` -- `BrainAgeRegressor`: thin wrapper over
  the CNN's regression head (see **Design choice** below) plus
  `compute_age_gap`.
- `src/evaluator.py` -- `Evaluator`: accuracy/recall/F1/AUC-ROC plus
  MAE/RMSE/R²/Pearson-r for brain age, and `generate_report()`.
- `src/pipeline.py` -- orchestrates the full sequence end-to-end,
  including subject-level `GroupKFold`, per-fold GAN augmentation, and a
  baseline-vs-augmented comparison.
- `src/plots.py` -- training loss curve, ROC curve, confusion matrix,
  predicted-vs-actual brain age scatter plot.
- `predict_subject.py` -- standalone single-subject inference script,
  mirroring the Phase 1 report's Sequence Diagram (Fig 4.5) end-to-end:
  `EEGLoader.load()` -> `Preprocessor` -> `CWTTransformer.transform()` ->
  `CNNClassifier.predict()` -> `GradCAMVisualizer.compute_heatmap()` ->
  `BrainAgeRegressor.compute_age_gap()`. This is the "Clinician / Medical
  Professional" use case from Fig 4.3, separate from the k-fold
  training/evaluation flow in `pipeline.py`, and the intended live demo.
- `tests/test_pipeline_smoke.py` -- end-to-end smoke test on synthetic
  data (see **Testing** below).

## Label substitution: AD vs HC instead of sMCI vs pMCI

The Phase-1 report specifies classifying **sMCI vs pMCI** (Stable vs
Progressive Mild Cognitive Impairment) using OpenNeuro **ds004504**. That
dataset (Miltiadous et al., *"A dataset of EEG recordings from
Alzheimer's disease, Frontotemporal dementia and Healthy subjects"*)
does **not** contain sMCI/pMCI labels. It contains three diagnostic
groups instead:

| Group | Subjects |
|---|---|
| Alzheimer's Disease (AD) | 36 |
| Frontotemporal Dementia (FTD) | 23 |
| Healthy Control (HC) | 29 |

**Decision:** build the classifier for **AD vs HC** (binary), the
cleanest binary split available in this dataset and a direct match for
the report's binary-classification design intent (sMCI/pMCI is also
binary). FTD subjects are excluded from the current label set so the
task stays strictly binary.

This is a **documented, deliberate adaptation**, not an oversight. The
label-loading step is isolated to one function --
`src.eeg_loader.get_labels()`, gated by `config.LABEL_MODE` (currently
`"AD_HC"`) -- so if real sMCI/pMCI-labeled data becomes available later,
only that one function needs to change; nothing else in the pipeline
(preprocessing, CWT, GAN, CNN, Grad-CAM, evaluation) depends on which
label scheme is in use.

## Reduced-fold cross-validation instead of full LOSO

The report specifies full Leave-One-Subject-Out (LOSO) cross-validation.
On the ~20-subject scoped build that means 20 full training runs -- too
slow for a 12-hour budget on CPU / shared Colab GPU.

**Decision:** use `sklearn.model_selection.GroupKFold` with
`n_splits=config.N_FOLDS` (default 3), grouped by `subject_ids`. This
enforces the same critical property as LOSO -- no subject's epochs
appear in both train and test for any fold -- just with fewer, larger
folds. Epochs from the same subject are highly correlated, so this
subject-independence property is what actually matters for a meaningful
estimate; the fold *count* is the only thing reduced.

**To scale up to full LOSO:** change `N_FOLDS` in `config.py` (or pass
`--folds`) to `len(subjects)`. No other code changes are required --
`GroupKFold(n_splits=...)` is already parameterized.

## Reduced GAN training

`GANAugmenter.train()` is capped at `config.GAN_EPOCHS` (default 50) via
`--gan_epochs`. This is far below convergence for a production GAN --
generated scalograms will be low-fidelity (blurry, possibly mode-
collapsed on such a small dataset). This is a documented limitation, not
a bug: the point of including GAN augmentation at reduced scale was to
(a) exercise the full augmentation architecture end-to-end and (b) let
the pipeline empirically show whether even a weak GAN helps or hurts
downstream classification, rather than assuming augmentation helps by
construction.

**This is why the pipeline always trains and evaluates a baseline
(no-augmentation) classifier alongside the GAN-augmented one for every
fold** -- see the `with_gan` / `without_gan` comparison in
`outputs/results/metrics_report.txt`. Discriminator/generator loss is
logged every epoch (and flagged if the discriminator loss collapses near
zero, a sign of mode collapse) so training instability is visible, not
silent.

**To scale up:** raise `GAN_EPOCHS` (config.py has a note that 500-1000+
is typical for real convergence) and/or deepen the DCGAN architecture in
`src/gan_augmenter.py`.

## Real-data results, a discovered bug, and the fix

The first full real-data run (20 real subjects: 10 AD, 10 HC, from
ds004504, 3-fold GroupKFold, 50 GAN epochs) produced:

| Metric | Result |
|---|---|
| Accuracy | 0.668 +/- 0.202 |
| Recall | 0.960 +/- 0.057 |
| F1 | 0.751 +/- 0.126 |
| **AUC-ROC** | **0.658 +/- 0.215 (report's NFR1 target is >=0.75 -- not met)** |
| Brain-age MAE | 6.14 +/- 0.47 years |
| **Brain-age R^2** | **-0.86 +/- 0.55 (worse than always predicting the mean age)** |
| Accuracy with GAN vs. without | 0.668 vs. 0.415 -- GAN augmentation clearly helped classification |

Full report archived at
`outputs/results/run1_20subj_meanage_labeling/metrics_report.txt`.

**The negative brain-age R^2 pointed to a real bug, not just "not enough
data":** `GANAugmenter` is conditioned on class label only, not age --
it has no notion of age to attach to a generated scalogram. The original
`_balance_with_gan` (src/pipeline.py) stamped *every* synthetic image
with a single constant value (the real mean age of that class in the
fold). In Fold 2 alone, this meant 1,299 training images all carried the
exact same age label, mixed in with the real, actually-varied ages --
actively teaching the regression head a distorted signal ("many
different-looking scans -> one identical age"), not just diluting it
with noise.

**Fix applied:** synthetic samples now get a **bootstrap-resampled** age
-- each synthetic image is assigned the age of a randomly-chosen real
subject from that same class in that fold's training data (with
replacement), instead of one constant value. This preserves the real age
*distribution* instead of collapsing it to a single point, without
inventing false per-image precision the GAN has no way to actually
provide. See `src/pipeline.py::_balance_with_gan`.

A second real-data run with this fix has not yet been completed (the
first run alone took ~2.5 hours on CPU) -- see **Switching to Colab**
below for the intended next run.

## Other scale reductions

| Report spec | This build | Config knob |
|---|---|---|
| Scalogram size 224x224 | 64x64 (32x32 in the smoke test) | `config.IMAGE_SIZE` |
| Full 88 subjects | 20 subjects, class-balanced subsample | `config.N_SUBJECTS_SUBSET` |
| Full LOSO (N folds) | 3-fold subject-level `GroupKFold` | `config.N_FOLDS` |
| GAN training to convergence | 50 epochs | `config.GAN_EPOCHS` |
| All 19 EEG channels | 6-channel frontal/temporal subset, stacked as image depth | `config.CHANNEL_SUBSET_PREFERRED` |

### Channel-selection strategy

Rather than averaging across all 19 channels (loses spatial/lateralization
information) or stacking all 19 as image depth (blows up CNN input size
and compute for a 12-hour budget), this build uses **option (c)**: a
fixed subset of diagnostically-relevant channels, stacked as image depth
for the CNN input. Preferred channels: `Fp1, Fp2, F7, F8, T3, T4`
(frontal + temporal -- frontal slowing and temporal-lobe changes are
well documented in AD/FTD EEG literature). Actual channel names are
resolved against the loaded montage at runtime
(`src.cwt_transformer.resolve_channel_subset`), since montage naming can
vary and shouldn't be hardcoded.

### Design choice: BrainAgeRegressor as a thin wrapper

The report's class diagram lists `BrainAgeRegressor` as its own class,
but architecturally brain-age regression shares the CNN's convolutional
features with the classification head -- the "Dual-Output CNN" in the
report's design diagram. Training a fully separate regression model would
duplicate the conv backbone and roughly double training time, a poor
trade-off given the time budget. `BrainAgeRegressor` is therefore a thin
wrapper delegating to `CNNClassifier`'s regression head, keeping the
class diagram's interface intact without a redundant model.

## Current limitations

- AD vs HC, not sMCI vs pMCI (see above) -- the clinical question
  answered is "does this EEG look like AD or a healthy control," not
  disease progression.
- 20 subjects is a small sample; per-fold test sets are small, so metric
  variance across folds (reported as mean ± std) will be non-trivial.
- GAN-generated scalograms are low-fidelity at 50 epochs; treat the
  with-GAN vs without-GAN comparison in the metrics report as the actual
  evidence for whether augmentation helped, not an assumption.
- 64x64 scalograms discard fine-grained time-frequency detail that
  224x224 would retain.
- ICA artifact rejection uses a kurtosis-threshold fallback when no
  EOG-proxy channel is available/usable; this is a coarser heuristic than
  a dedicated EOG channel would allow.

## Scaling up to the full report spec

1. **More subjects:** raise `config.N_SUBJECTS_SUBSET` (or `--n_subjects`)
   up to 88 (all available AD+HC+FTD subjects, if FTD is later included
   in a 3-class version) or however many AD+HC subjects exist (65).
2. **Full LOSO:** raise `config.N_FOLDS` (or `--folds`) to equal the
   subject count.
3. **Longer GAN training:** raise `config.GAN_EPOCHS` (or `--gan_epochs`)
   to 500-1000+; consider deepening `GANAugmenter`'s Conv2D/Conv2DTranspose
   stacks once scalograms are larger.
4. **224x224 images:** raise `config.IMAGE_SIZE` to `(224, 224)`. Expect
   proportionally higher CWT, GAN, and CNN compute/memory cost.
5. Re-run: `python run_pipeline.py --n_subjects 88 --folds 88 --gan_epochs 500`.

## Dataset

Source: OpenNeuro **ds004504** (BIDS-formatted, `.set`/`.fdt` EEGLAB
files per subject, `participants.tsv` at the dataset root for group/age/
MMSE metadata).

```bash
python download_dataset.py --n-subjects 20   # or -1 for all 88
```

Tries `aws s3 sync` against the public OpenNeuro S3 mirror first (no
credentials needed), then `openneuro-cli`, and prints manual-download
instructions (HTTPS) if neither is available.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Runs correctly on CPU (slower) or GPU; `run_pipeline.py` logs whether a
GPU was detected via `tf.config.list_physical_devices('GPU')` but never
requires one.

## Testing (run this first)

```bash
pytest tests/test_pipeline_smoke.py -v
```

Generates synthetic EEG-shaped arrays (matching the real dataset's
sampling rate and channel count) and chains CWT -> GAN -> CNN -> Grad-CAM
-> Evaluator, asserting correct shapes and metric ranges at every stage.
Preprocessing (ICA etc.) is skipped for synthetic data since ICA on
random noise is meaningless. Runs in well under a minute and is meant to
catch integration bugs before spending real time on the actual dataset.
**Only proceed to real data once this passes cleanly.**

## Running the full pipeline

```bash
python run_pipeline.py --n_subjects 20 --folds 3 --gan_epochs 50
python -m src.plots   # after the run, generates result plots
```

All three flags are config-backed and overridable; see `config.py` for
every other hyperparameter.

## Running inference on one subject

After `run_pipeline.py` has produced `outputs/models/cnn_classifier_final.keras`:

```bash
python predict_subject.py \
  --eeg-file data/sub-005/eeg/sub-005_task-eyesclosed_eeg.set \
  --chronological-age 70
```

Prints a JSON summary (predicted class, per-class probabilities,
predicted brain age, brain age gap if `--chronological-age` is given) and
saves a few Grad-CAM overlays to `outputs/gradcam/`. This walks through
the Phase 1 report's Sequence Diagram (Fig 4.5) for a single new subject
and is the intended panel-demo script.

## Switching to Colab

Nothing in this codebase is laptop-specific -- `run_pipeline.py` already
checks for a GPU via `tf.config.list_physical_devices('GPU')` and uses
one automatically if present, with no code changes needed. A single real
20-subject, 3-fold, 50-GAN-epoch run took ~2.5 hours on CPU here; the
same run on Colab's free GPU should be substantially faster, which
matters most once scaling up (more subjects, full LOSO, more GAN epochs,
224x224 images -- see **Scaling up** above).

```python
!git clone https://github.com/afreenanz/alzheimer-eeg-progression-prediction.git
%cd alzheimer-eeg-progression-prediction
!pip install -r requirements.txt
!pip install awscli   # not preinstalled on Colab
!python download_dataset.py --n-subjects 65
```

No venv needed on Colab -- each notebook session is already an isolated
environment.

### Running in stages (recommended on Colab -- avoids losing progress to disconnects)

Free-tier Colab sessions can disconnect mid-run, and `run_pipeline.py`'s
default single-process mode keeps everything in memory -- a disconnect
during a multi-hour run loses all of it. Running each stage as its own
cell avoids this: a disconnect only costs whichever single stage was
running, and re-running a completed stage is a no-op (it loads the saved
result instead of redoing the work).

```python
# Stage 1: build + cache the dataset (run once; safe to re-run if it fails)
!python run_pipeline.py --stage build --n_subjects 65

# Stage 2: one fold per cell -- run these one at a time
!python run_pipeline.py --stage fold --fold 1 --n_subjects 65 --folds 3 --gan_epochs 50
!python run_pipeline.py --stage fold --fold 2 --n_subjects 65 --folds 3 --gan_epochs 50
!python run_pipeline.py --stage fold --fold 3 --n_subjects 65 --folds 3 --gan_epochs 50

# Stage 3: combine all fold results into the final report (only once all folds are done)
!python run_pipeline.py --stage aggregate --folds 3

!python -m src.plots
```

If a fold's cell disconnects partway through, just re-run that exact
same cell -- earlier folds are untouched, and this fold starts fresh
(no partial-fold resume within a single fold, only across folds).

## Outputs

- `outputs/results/metrics_report.txt` -- accuracy, recall, F1, AUC-ROC
  (mean ± std across folds) for classification; MAE, RMSE, R²,
  correlation for brain age regression; with-GAN vs without-GAN
  comparison.
- `outputs/results/training_loss.png`, `roc_curve.png`,
  `confusion_matrix.png`, `brain_age_scatter.png` -- from `src/plots.py`.
- `outputs/gradcam/` -- Grad-CAM heatmap overlays (correct and incorrect
  predictions, both classes where available).
- `outputs/models/cnn_classifier_final.keras` -- saved trained model.
- `outputs/scalograms/{AD,HC}/` -- generated CWT scalogram arrays.

## Reproducibility

Random seeds (Python `random`, NumPy, TensorFlow) are fixed via
`config.RANDOM_SEED` and set explicitly at the top of `run_pipeline.py`.

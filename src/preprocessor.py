"""Bandpass filtering, ICA-based artifact removal, and epoching."""

import logging

import numpy as np
from scipy.stats import kurtosis

import config

logger = logging.getLogger(__name__)


class Preprocessor:
    """Cleans raw EEG and slices it into fixed-length epochs.

    Attributes
    ----------
    bandpass_range : tuple of float
        (low_freq, high_freq) in Hz for the bandpass filter.
    ica_components : int
        Number of ICA components to fit (capped to available channels).
    """

    def __init__(
        self,
        bandpass_range=config.BANDPASS_RANGE,
        ica_components=config.ICA_N_COMPONENTS,
    ):
        self.bandpass_range = bandpass_range
        self.ica_components = ica_components

    def filter(self, raw):
        """Bandpass-filter and re-reference the raw recording.

        Also applies an average reference (Fig 4.1 Stage 2 of the Phase 1
        report lists "Re-referencing" alongside bandpass filtering and
        artifact removal). Average reference is the standard choice when
        no dedicated reference channel is recorded, as is the case for
        this dataset's 19-channel 10-20 montage.

        Parameters
        ----------
        raw : mne.io.Raw

        Returns
        -------
        mne.io.Raw
        """
        l_freq, h_freq = self.bandpass_range
        raw = raw.copy()
        raw.filter(l_freq=l_freq, h_freq=h_freq, verbose="ERROR")
        raw.set_eeg_reference("average", projection=False, verbose="ERROR")
        return raw

    def remove_artifacts(self, raw):
        """Fit ICA, auto-flag likely artifact components, exclude, apply.

        Uses `ica.find_bads_eog()` when an EOG-proxy channel is available
        (e.g. Fp1/Fp2 acting as frontal EOG proxies, common in this
        dataset which has no dedicated EOG channel). Falls back to
        flagging components whose kurtosis exceeds a threshold, which is
        a standard heuristic for eye-blink / muscle artifacts (they
        produce highly non-Gaussian, peaky component time courses).

        Parameters
        ----------
        raw : mne.io.Raw

        Returns
        -------
        mne.io.Raw
            Cleaned raw object with artifact components excluded.
        """
        import mne

        n_channels = len(raw.ch_names)
        n_components = min(self.ica_components, max(n_channels - 1, 1))
        if n_components < 1:
            logger.warning("Too few channels for ICA; skipping artifact removal.")
            return raw

        ica = mne.preprocessing.ICA(
            n_components=n_components,
            random_state=config.RANDOM_SEED,
            max_iter="auto",
            verbose="ERROR",
        )
        try:
            ica.fit(raw, verbose="ERROR")
        except Exception as exc:  # noqa: BLE001
            logger.warning("ICA fit failed (%s); returning filtered-only raw.", exc)
            return raw

        excluded = []
        eog_proxy_candidates = [
            ch for ch in ("Fp1", "Fp2", "FP1", "FP2") if ch in raw.ch_names
        ]
        if eog_proxy_candidates:
            try:
                eog_indices, _ = ica.find_bads_eog(
                    raw, ch_name=eog_proxy_candidates, verbose="ERROR"
                )
                excluded.extend(eog_indices)
            except Exception as exc:  # noqa: BLE001
                logger.info("find_bads_eog failed (%s); falling back to kurtosis.", exc)

        if not excluded:
            sources = ica.get_sources(raw).get_data()
            comp_kurtosis = kurtosis(sources, axis=1, fisher=True)
            threshold = 5.0
            excluded = [
                int(i) for i, k in enumerate(comp_kurtosis) if k > threshold
            ]
            if excluded:
                logger.info(
                    "Flagged %d component(s) via kurtosis fallback (>%.1f): %s",
                    len(excluded),
                    threshold,
                    excluded,
                )

        ica.exclude = excluded
        cleaned = raw.copy()
        ica.apply(cleaned, verbose="ERROR")
        logger.info("ICA excluded %d/%d components.", len(excluded), n_components)
        return cleaned

    def extract_epochs(
        self,
        raw,
        duration=config.EPOCH_DURATION_SEC,
        overlap=config.EPOCH_OVERLAP_SEC,
    ):
        """Slice raw into fixed-length overlapping epochs.

        Also applies baseline correction and per-channel normalization
        (Fig 4.1 Stage 2 of the Phase 1 report), matching the two steps
        not otherwise covered by `filter`/`remove_artifacts`. Since these
        are fixed-length resting-state epochs with no stimulus onset,
        "baseline" is each epoch's own mean (DC-offset removal) rather
        than a pre-stimulus window; normalization is a per-channel,
        per-epoch z-score, a standard choice before feeding EEG into a
        CWT/CNN pipeline. This is distinct from -- and in addition to --
        the per-image [0,1] normalization CWTTransformer applies later to
        the scalogram itself.

        Parameters
        ----------
        raw : mne.io.Raw
        duration : float
            Epoch length in seconds.
        overlap : float
            Overlap between consecutive epochs in seconds.

        Returns
        -------
        mne.Epochs
        """
        import mne

        epochs = mne.make_fixed_length_epochs(
            raw, duration=duration, overlap=overlap, preload=True, verbose="ERROR"
        )
        if len(epochs) > 0:
            epochs.apply_baseline(baseline=(None, None), verbose="ERROR")
            data = epochs.get_data()
            std = data.std(axis=2, keepdims=True)
            std[std < 1e-12] = 1.0
            epochs._data = data / std

        n_epochs = len(epochs)
        if n_epochs == 0:
            logger.error(
                "0 epochs extracted from a %.1fs recording -- subject yields "
                "no usable data and should be skipped.",
                raw.times[-1] if len(raw.times) else 0.0,
            )
        elif n_epochs < 3:
            logger.warning(
                "Only %d epoch(s) extracted -- subject yields near-zero "
                "usable data; downstream results for this subject may be "
                "unreliable.",
                n_epochs,
            )
        else:
            logger.info("Extracted %d epochs.", n_epochs)
        return epochs

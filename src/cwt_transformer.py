"""Continuous Wavelet Transform: epochs -> multi-channel scalogram images.

Multi-channel strategy (option (c) from the module spec): rather than
averaging across all 19 channels (loses spatial/lateralization info) or
stacking all 19 as depth (blows up the CNN input and compute budget for
the 12-hour build), a fixed subset of diagnostically-relevant
frontal/temporal channels is selected and stacked as image depth. Frontal
(Fp1/Fp2, F7/F8) and temporal (T3/T4) channels are chosen because
frontotemporal and temporal-lobe changes are well documented in AD/FTD
EEG literature (slowing of frontal rhythms, temporal asymmetries). Actual
channel names are resolved against the loaded montage at runtime (see
`resolve_channel_subset`) since dataset montages do not always match the
preferred names exactly (e.g. upper vs lower case).
"""

import logging
import os

import numpy as np
import pywt
from PIL import Image

import config

logger = logging.getLogger(__name__)


def resolve_channel_subset(
    available_channels, preferred=config.CHANNEL_SUBSET_PREFERRED
):
    """Map preferred channel names onto whatever the montage actually has.

    Matches case-insensitively and falls back to the first N available
    channels (logging a warning) if fewer than requested preferred
    channels are found, so the pipeline never hard-crashes on a montage
    naming mismatch.

    Parameters
    ----------
    available_channels : list of str
        Channel names present in the loaded recording.
    preferred : list of str
        Preferred channel names, in priority order.

    Returns
    -------
    list of str
        Resolved channel names (subset of `available_channels`), length
        <= len(preferred).
    """
    lookup = {ch.lower(): ch for ch in available_channels}
    resolved = [lookup[p.lower()] for p in preferred if p.lower() in lookup]

    if len(resolved) < len(preferred):
        missing = [p for p in preferred if p.lower() not in lookup]
        logger.warning(
            "Channel subset mismatch: %s not found in montage %s. "
            "Falling back to first available channels to fill the gap.",
            missing,
            available_channels,
        )
        fallback = [ch for ch in available_channels if ch not in resolved]
        n_needed = len(preferred) - len(resolved)
        resolved.extend(fallback[:n_needed])

    if not resolved:
        raise ValueError("No usable channels found for CWT channel subset.")
    return resolved


class CWTTransformer:
    """Converts epoch data into normalized multi-channel scalogram images.

    Attributes
    ----------
    wavelet_type : str
        PyWavelets continuous wavelet name (default 'morl').
    frequency_range : tuple of int
        (low_hz, high_hz) frequency band to render in the scalogram.
    image_size : tuple of int
        (height, width) each channel's scalogram is resized to. Full
        report spec is 224x224; reduced to 64x64 here for the 12-hour
        build -- bump config.IMAGE_SIZE up when compute allows.
    """

    def __init__(
        self,
        wavelet_type=config.WAVELET_TYPE,
        frequency_range=config.CWT_FREQ_RANGE,
        image_size=config.IMAGE_SIZE,
    ):
        self.wavelet_type = wavelet_type
        self.frequency_range = frequency_range
        self.image_size = image_size

    def _scales_for_freq_range(self, sampling_rate):
        fmin, fmax = self.frequency_range
        central_freq = pywt.central_frequency(self.wavelet_type)
        # scale = central_freq * sampling_rate / freq
        scale_max = central_freq * sampling_rate / max(fmin, 1e-6)
        scale_min = central_freq * sampling_rate / fmax
        n_scales = self.image_size[0]
        return np.linspace(scale_min, scale_max, n_scales)

    def _resize(self, arr_2d):
        img = Image.fromarray(arr_2d.astype(np.float32), mode="F")
        img = img.resize((self.image_size[1], self.image_size[0]), Image.BILINEAR)
        return np.array(img, dtype=np.float32)

    def transform(self, epoch_data, sampling_rate=config.SAMPLING_RATE):
        """Apply CWT per channel and stack into a multi-channel image.

        Parameters
        ----------
        epoch_data : np.ndarray, shape (n_channels, n_timepoints)
            Single epoch's data for the already-selected channel subset.
        sampling_rate : float
            Sampling rate of the epoch data in Hz.

        Returns
        -------
        np.ndarray, shape (image_size[0], image_size[1], n_channels)
            Stacked, resized, [0, 1]-normalized scalogram.
        """
        n_channels = epoch_data.shape[0]
        scales = self._scales_for_freq_range(sampling_rate)
        channel_images = []
        for ch_idx in range(n_channels):
            coeffs, _ = pywt.cwt(
                epoch_data[ch_idx], scales, self.wavelet_type, sampling_period=1.0 / sampling_rate
            )
            power = np.abs(coeffs)
            resized = self._resize(power)
            ch_min, ch_max = resized.min(), resized.max()
            if ch_max - ch_min < 1e-12:
                normalized = np.zeros_like(resized)
            else:
                normalized = (resized - ch_min) / (ch_max - ch_min)
            channel_images.append(normalized)
        return np.stack(channel_images, axis=-1).astype(np.float32)

    def save_scalogram(self, image, label, subject_id, epoch_idx):
        """Save a scalogram array to outputs/scalograms/{label}/.

        Parameters
        ----------
        image : np.ndarray
            Scalogram array, shape (H, W, C).
        label : str
            Class label (e.g. "AD" or "HC"), used as the subdirectory.
        subject_id : str
            Subject identifier, used in the filename.
        epoch_idx : int
            Epoch index within the subject, used in the filename.

        Returns
        -------
        str
            Path the scalogram was saved to.
        """
        out_dir = os.path.join(config.SCALOGRAMS_DIR, label)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{subject_id}_epoch{epoch_idx:04d}.npy")
        np.save(out_path, image)
        return out_path

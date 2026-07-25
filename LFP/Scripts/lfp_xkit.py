"""lfp_xkit.py

LFP cleaning toolkit with two-sided Z-score EKG detection, robust dead segment 
identification, and spectral reconstruction.

Pipeline:
    1. MNE bandpass filter
    2. Soft Gaussian notch filter
    3. Multi-scale dead/flat segment detection with morphological closing
    4. Spectral reconstruction of dead segments (prevents zero-variance gaps)
    5. QRS-only EKG template subtraction (zero-offset, high-pass filtered model)
    6. Optional short-segment spike detection

Delta-band preservation safeguards:
    - T-wave subtraction DISABLED by default (occupies 0.5-4 Hz directly)
    - Baseline offset forced to 0.0 (prevents slow drift absorption)
    - QRS model high-pass filtered at 4 Hz (removes phase-locked Delta
      embedded during template averaging)
    - Dead segments reconstructed via spectral matching (NOT interpolation)

Dead segment detection improvements:
    - Small base window (0.2s) for sensitivity to gradual onsets
    - Morphological closing merges nearby detections (<1.0s apart)
    - Minimum duration filter avoids false positives from brief neural silences

This module does NOT plot. Use viz_lfp.py for visualization.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

import numpy as np
import mne
from numpy.typing import NDArray
from scipy.signal import butter, filtfilt, welch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default pipeline configuration
# ---------------------------------------------------------------------------

BANDPASS_L_FREQ: float = 1.0
BANDPASS_H_FREQ: float = 100.0

NOTCH_FREQ: float = 60.0
NOTCH_MAX_DB: float = 1.0
NOTCH_SIGMA_HZ: float = 2.0

EKG_THR: float | str = 50e-6
EKG_AUTO_SENSITIVITY: float = 2.0
EKG_POLARITY: int = 1
SUBTRACT_T_WAVE: bool = False
QRS_HIGHPASS_CUTOFF_HZ: float = 4.0

# Dead segment detection & reconstruction parameters
DEAD_SEGMENT_SEC: float = 0.2          # Reduced from 0.5 for sensitivity
DEAD_STD_RATIO: float = 0.05
MIN_GAP_MERGE_SEC: float = 1.0         # Merge gaps < 1s apart
MIN_DEAD_DURATION_SEC: float = 0.3     # Minimum dead region duration
RECONSTRUCT_DEAD: bool = True          # Enable spectral reconstruction
RECON_CONTEXT_SEC: float = 2.0         # Seconds of healthy context for PSD

SPIKE_SEGMENT_SEC: float = 0.5
SPIKE_MAD_THRESH: float = 8.0          # Increased to reduce false positives

_ART_WIDTH_SEARCH_SEC: float = 0.04
_ART_TIME_B4_PEAK_SEC: float = 0.06
_ART_TIME_AFTER_PEAK_SEC: float = 0.30
_TWAVE_TIME_SEC: float = 0.15
_SHIFT_SEARCH_RANGE: int = 4


class LFPCleanResult(TypedDict, total=False):
    """Result dictionary returned by lfp_xkit."""

    data: NDArray[np.float64]
    roi_names: list[str]
    subject_id: str
    sfreq: float

    raw_filtered: NDArray[np.float64]
    removed: NDArray[np.float64]
    spike_mask: NDArray[np.bool_]
    dead_mask: NDArray[np.bool_]
    reconstructed_mask: NDArray[np.bool_]  # NEW: tracks reconstructed samples

    ekg_peaks: list[list[int]]
    ekg_templates: list[NDArray[np.float64] | None]
    ekg_counts: list[int]
    ekg_thresholds: list[float]


@dataclass(frozen=True)
class _EKGParams:
    """Sample-domain EKG parameters."""

    fs: float
    thr_pos: float
    thr_neg: float
    art_width_search_pts: int
    art_pts_b4_peak: int
    art_pts_after_peak: int
    twave_pts: int

    @property
    def art_pts_plus_t_after_peak(self) -> int:
        return self.art_pts_after_peak + self.twave_pts

    @property
    def art_width_pts(self) -> int:
        return self.art_pts_b4_peak + self.art_pts_plus_t_after_peak + 1

    @property
    def qrs_width(self) -> int:
        return self.art_pts_b4_peak + self.art_pts_after_peak + 1


# ---------------------------------------------------------------------------
# Numerical helpers
# ---------------------------------------------------------------------------

def _matlab_round(x: float) -> int:
    """MATLAB-style rounding: half away from zero."""
    return int(np.floor(x + 0.5))


def _compute_robust_stats(sig: NDArray[np.float64]) -> tuple[float, float]:
    """Compute median and robust standard deviation (MAD-based)."""
    finite = sig[np.isfinite(sig)]
    if finite.size == 0:
        return 0.0, 0.0

    med = float(np.median(finite))
    mad = float(np.median(np.abs(finite - med)))
    robust_std = mad * 1.4826
    return med, robust_std


def _resolve_two_sided_threshold(
    sig: NDArray[np.float64],
    thr: float | str | None,
    sensitivity: float,
) -> tuple[float, float]:
    """Resolve two-sided Z-score thresholds for EKG detection."""
    if isinstance(thr, str):
        if thr != "auto":
            raise ValueError("String threshold must be 'auto'.")
    elif thr is not None and not isinstance(thr, str):
        return float(thr), -float(thr)

    med, robust_std = _compute_robust_stats(sig)

    if robust_std > 0.0:
        return med + sensitivity * robust_std, med - sensitivity * robust_std

    p99 = float(np.percentile(sig[np.isfinite(sig)], 99.0))
    p01 = float(np.percentile(sig[np.isfinite(sig)], 1.0))
    return p99, p01


def _compute_ekg_params(fs: float, thr_pos: float, thr_neg: float) -> _EKGParams:
    """Convert time constants to sample counts."""
    return _EKGParams(
        fs=fs,
        thr_pos=thr_pos,
        thr_neg=thr_neg,
        art_width_search_pts=max(1, _matlab_round(_ART_WIDTH_SEARCH_SEC * fs)),
        art_pts_b4_peak=max(1, _matlab_round(_ART_TIME_B4_PEAK_SEC * fs)),
        art_pts_after_peak=max(1, _matlab_round(_ART_TIME_AFTER_PEAK_SEC * fs)),
        twave_pts=max(1, _matlab_round(_TWAVE_TIME_SEC * fs)),
    )


def _highpass_template(
    template: NDArray[np.float64],
    fs: float,
    cutoff_hz: float,
) -> NDArray[np.float64]:
    """Zero-phase high-pass filter for QRS template only.

    Removes embedded low-frequency neural activity from the averaging process
    while preserving the sharp QRS transient morphology (>4 Hz).
    """
    nyq = fs / 2.0
    if cutoff_hz >= nyq or len(template) < 10:
        return template.copy()

    b, a = butter(3, cutoff_hz / nyq, btype="high")
    padlen = min(3 * max(len(a), len(b)), len(template) - 1)
    if padlen < 1:
        return template.copy()

    return filtfilt(b, a, template, padlen=padlen)


# ---------------------------------------------------------------------------
# Multi-scale dead / flat segment detection with morphological closing
# ---------------------------------------------------------------------------

def _detect_dead_segments(
    sig: NDArray[np.float64],
    sfreq: float,
    window_sec: float = DEAD_SEGMENT_SEC,
    std_ratio: float = DEAD_STD_RATIO,
    min_gap_merge_sec: float = MIN_GAP_MERGE_SEC,
    min_dead_duration_sec: float = MIN_DEAD_DURATION_SEC,
) -> NDArray[np.bool_]:
    """Multi-scale dead segment detection with morphological closing.

    Uses small windows for sensitivity to gradual onsets, then merges nearby
    detections to catch hardware failures that span multiple windows. Only
    flags regions >= min_dead_duration_sec to avoid false positives from
    brief neural silences.

    Args:
        sig: Input signal.
        sfreq: Sampling frequency.
        window_sec: Base window size for initial detection (smaller = more sensitive).
        std_ratio: Std ratio threshold relative to median healthy std.
        min_gap_merge_sec: Merge detected dead regions separated by < this duration.
        min_dead_duration_sec: Only flag regions >= this duration as dead.

    Returns:
        Boolean mask where True = dead segment.
    """
    n = len(sig)
    if n < 2:
        return np.zeros(n, dtype=bool)

    # Step 1: Small-window detection for sensitivity
    window_len = max(1, int(sfreq * window_sec))
    n_windows = int(np.ceil(n / window_len))

    window_stds = np.zeros(n_windows)
    for i in range(n_windows):
        start = i * window_len
        end = min(start + window_len, n)
        seg = sig[start:end]
        finite = seg[np.isfinite(seg)]
        window_stds[i] = float(np.std(finite)) if finite.size > 1 else 0.0

    valid_stds = window_stds[window_stds > 0]
    if valid_stds.size == 0:
        logger.warning("No valid std values found. Marking entire signal as dead.")
        return np.ones(n, dtype=bool)

    median_std = float(np.median(valid_stds))
    threshold = median_std * std_ratio

    # Initial binary mask at window level
    raw_mask = np.zeros(n_windows, dtype=bool)
    for i in range(n_windows):
        if window_stds[i] <= threshold:
            raw_mask[i] = True

    # Convert to sample-level mask
    sample_mask = np.zeros(n, dtype=bool)
    for i in range(n_windows):
        if raw_mask[i]:
            start = i * window_len
            end = min(start + window_len, n)
            sample_mask[start:end] = True

    # Step 2: Morphological closing - merge nearby detections
    diff = np.diff(sample_mask.astype(int))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0] + 1

    if sample_mask[0]:
        starts = np.concatenate([[0], starts])
    if sample_mask[-1]:
        ends = np.concatenate([ends, [n]])

    merge_samples = int(min_gap_merge_sec * sfreq)
    merged_starts = []
    merged_ends = []

    if len(starts) > 0:
        current_start = starts[0]
        current_end = ends[0]

        for s, e in zip(starts[1:], ends[1:]):
            if s - current_end <= merge_samples:
                current_end = e
            else:
                merged_starts.append(current_start)
                merged_ends.append(current_end)
                current_start = s
                current_end = e

        merged_starts.append(current_start)
        merged_ends.append(current_end)

    # Step 3: Filter out regions shorter than min_dead_duration_sec
    min_dead_samples = int(min_dead_duration_sec * sfreq)
    final_mask = np.zeros(n, dtype=bool)

    for s, e in zip(merged_starts, merged_ends):
        if (e - s) >= min_dead_samples:
            final_mask[s:e] = True

    n_dead = int(final_mask.sum())
    pct = 100.0 * n_dead / n
    logger.info(
        "Dead segments: %d samples (%.1f%% of recording) | %d regions detected",
        n_dead, pct, len(merged_starts),
    )

    return final_mask


# ---------------------------------------------------------------------------
# Spectral reconstruction of dead segments
# ---------------------------------------------------------------------------

def _reconstruct_dead_segments(
    sig: NDArray[np.float64],
    dead_mask: NDArray[np.bool_],
    sfreq: float,
    context_sec: float = RECON_CONTEXT_SEC,
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Reconstruct dead segments via spectral matching from surrounding context.

    Generates surrogate data with matched PSD amplitude but random phase,
    preserving statistical properties without inventing phase-locked oscillations.
    Applies smooth fade-in/fade-out at boundaries to avoid discontinuities.

    This is NOT interpolation. Interpolation creates artificial low-frequency
    coherence that corrupts Delta analysis. Spectral matching preserves the
    actual PSD shape of healthy surrounding data.

    Args:
        sig: Input signal with dead segments.
        dead_mask: Boolean mask identifying dead samples.
        sfreq: Sampling frequency.
        context_sec: Seconds of healthy data for PSD estimation.

    Returns:
        (reconstructed_signal, reconstruction_mask)
    """
    out = sig.copy()
    recon_mask = np.zeros_like(dead_mask)

    if not dead_mask.any():
        return out, recon_mask

    # Find contiguous dead regions
    diff = np.diff(dead_mask.astype(int))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0] + 1

    if dead_mask[0]:
        starts = np.concatenate([[0], starts])
    if dead_mask[-1]:
        ends = np.concatenate([ends, [len(dead_mask)]])

    context_samples = int(context_sec * sfreq)

    for start, end in zip(starts, ends):
        length = end - start
        if length == 0:
            continue

        # Get healthy context before and after
        ctx_before_start = max(0, start - context_samples)
        ctx_before = sig[ctx_before_start:start]

        ctx_after_end = min(len(sig), end + context_samples)
        ctx_after = sig[end:ctx_after_end]

        context_data = np.concatenate([ctx_before, ctx_after])
        if len(context_data) < 10:
            out[start:end] = 0.0
            recon_mask[start:end] = True
            logger.warning(
                "Insufficient context at [%d:%d]. Filling with zeros.", start, end
            )
            continue

        # Estimate PSD from healthy context
        nperseg = min(len(context_data), int(sfreq * 1.0))
        freqs, psd = welch(context_data, fs=sfreq, nperseg=nperseg, window="hann")

        # Generate surrogate with matched spectrum, random phase
        phases = np.random.uniform(0, 2 * np.pi, len(freqs))
        amplitude = np.sqrt(psd * sfreq * nperseg / 2)
        spectrum = amplitude * np.exp(1j * phases)
        surrogate = np.fft.irfft(spectrum, n=length)

        # Scale to match local RMS of context
        context_rms = np.sqrt(np.mean(context_data ** 2))
        surrogate_rms = np.sqrt(np.mean(surrogate ** 2))
        if surrogate_rms > 0:
            surrogate *= context_rms / surrogate_rms

        # Smooth fade-in/fade-out at boundaries
        fade_len = min(20, length // 4)
        if fade_len > 0:
            fade_in = np.linspace(0, 1, fade_len)
            fade_out = np.linspace(1, 0, fade_len)
            surrogate[:fade_len] *= fade_in
            if length > fade_len:
                surrogate[-fade_len:] *= fade_out

        out[start:end] = surrogate
        recon_mask[start:end] = True

    n_recon = int(recon_mask.sum())
    if n_recon > 0:
        logger.info(
            "Reconstructed %d dead samples (%.1f%%) via spectral matching",
            n_recon,
            100.0 * n_recon / len(sig),
        )

    return out, recon_mask


# ---------------------------------------------------------------------------
# EKG detection and subtraction
# ---------------------------------------------------------------------------

def _detect_ekg_peaks(
    sig: NDArray[np.float64],
    params: _EKGParams,
) -> list[int]:
    """Detect threshold crossings using TWO-SIDED Z-score thresholds."""
    n = len(sig)
    peaks: list[int] = []

    if n < 2:
        return peaks

    prev_pt = sig[0]
    data_index = 1

    while True:
        if data_index >= n:
            break

        found_threshold_crossing = False

        if prev_pt <= params.thr_pos and sig[data_index] > params.thr_pos:
            found_threshold_crossing = True

        if found_threshold_crossing:
            if data_index + params.art_width_search_pts >= n:
                break

            peak_dat = sig[data_index]
            peak_index = data_index

            for k in range(1, params.art_width_search_pts):
                idx = data_index + k
                if sig[idx] > peak_dat:
                    peak_dat = sig[idx]
                    peak_index = idx

            peaks.append(int(peak_index))
            data_index = peak_index + params.art_pts_plus_t_after_peak

            if data_index >= n - 1:
                break

            prev_pt = sig[data_index - 1]
        else:
            prev_pt = sig[data_index]
            data_index += 1

    return peaks


def _fit_scale_fixed_offset(
    template: NDArray[np.float64],
    segment: NDArray[np.float64],
    offset: float,
) -> float:
    """Fit segment = scale * template + offset, with offset fixed."""
    assert len(template) == len(segment)
    y = segment - offset
    denom = float(np.dot(template, template))
    if denom <= 0.0:
        return 0.0
    return float(np.dot(template, y) / denom)


def _ekg_artifact_removal(
    sig: NDArray[np.float64],
    fs: float,
    thr: float | str | None,
    polarity: int,
    auto_sensitivity: float,
    subtract_t_wave: bool = SUBTRACT_T_WAVE,
    qrs_highpass_hz: float = QRS_HIGHPASS_CUTOFF_HZ,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    list[int],
    NDArray[np.float64] | None,
    float,
]:
    """QRS-only template subtraction with three Delta-preservation safeguards.

    Safeguards:
      1. T-wave subtraction disabled by default (occupies 0.5-4 Hz directly)
      2. Baseline offset forced to 0.0 (prevents slow drift absorption)
      3. QRS model high-pass filtered at qrs_highpass_hz (removes phase-locked
         Delta embedded during template averaging)
    """
    sig = np.asarray(sig, dtype=np.float64).ravel()
    n = len(sig)

    if n == 0:
        return sig.copy(), np.zeros_like(sig), [], None, 0.0

    if not np.all(np.isfinite(sig)):
        fill = float(np.nanmedian(sig[np.isfinite(sig)])) if np.any(np.isfinite(sig)) else 0.0
        sig = np.nan_to_num(sig, nan=fill, posinf=fill, neginf=fill)

    if polarity not in (-1, 1):
        raise ValueError("polarity must be +1 or -1.")

    detect_sig = -sig if polarity == -1 else sig
    thr_pos, thr_neg = _resolve_two_sided_threshold(detect_sig, thr, auto_sensitivity)
    params = _compute_ekg_params(fs, thr_pos, thr_neg)

    peaks = _detect_ekg_peaks(detect_sig, params)

    if not peaks:
        logger.warning(
            "No EKG peaks detected. Thresholds: pos=%.6g, neg=%.6g",
            thr_pos, thr_neg,
        )
        return sig.copy(), np.zeros_like(sig), peaks, None, thr_pos

    # Build QRST template for averaging
    art_sum = np.zeros(params.art_width_pts, dtype=np.float64)
    art_avg_count = 0

    for pk in peaks:
        start = pk - params.art_pts_b4_peak
        end = pk + params.art_pts_plus_t_after_peak + 1
        if 0 <= start and end <= n:
            art_sum += sig[start:end]
            art_avg_count += 1

    if art_avg_count == 0:
        logger.warning("No valid EKG windows for template averaging.")
        return sig.copy(), np.zeros_like(sig), peaks, None, thr_pos

    qrst = art_sum / float(art_avg_count)
    sigin = sig.copy()
    sigout = sigin.copy()

    mn = float(np.min(qrst))
    qrst = qrst - mn

    if np.allclose(qrst, 0.0):
        logger.warning("QRST template is effectively zero.")
        return sigout, np.zeros_like(sig), peaks, None, thr_pos

    qrs_width = params.qrs_width
    t_width = len(qrst) - qrs_width
    if t_width < 0:
        raise ValueError("Widths do not match QRST length.")

    # SAFEGUARD 3: High-pass filter QRS model to remove embedded Delta
    qrs_model_raw = qrst[:qrs_width]
    qrs_model = _highpass_template(qrs_model_raw, fs, cutoff_hz=qrs_highpass_hz)

    pre = params.art_pts_b4_peak
    post = params.art_pts_after_peak

    for pk in peaks:
        qrs_loc1 = pk + 1

        if not (
            qrs_loc1 > pre + _SHIFT_SEARCH_RANGE
            and qrs_loc1 + post + _SHIFT_SEARCH_RANGE < n
        ):
            continue

        # --- QRS fit with ZERO offset against HIGH-PASS filtered model ---
        best_qrs_err = np.inf
        qrs_scale = 1.0
        qrs_shift = 0
        qrs_ok = False

        for shift in range(-_SHIFT_SEARCH_RANGE, _SHIFT_SEARCH_RANGE + 1):
            start = pk - pre + shift
            end = pk + post + shift + 1
            if start < 0 or end > n:
                continue
            match_qrs = sigin[start:end]
            if len(match_qrs) != len(qrs_model) or not np.all(np.isfinite(match_qrs)):
                continue

            # SAFEGUARD 2: offset fixed at 0.0
            scale = _fit_scale_fixed_offset(qrs_model, match_qrs, offset=0.0)
            err = float(np.sum((match_qrs - scale * qrs_model) ** 2))

            if shift == -_SHIFT_SEARCH_RANGE or err < best_qrs_err:
                best_qrs_err = err
                qrs_scale = scale
                qrs_shift = shift
                qrs_ok = True

        if not qrs_ok:
            continue

        # Subtract QRS: zero offset, high-pass filtered model
        qrs_start = pk - pre + qrs_shift
        qrs_end = qrs_start + len(qrs_model)
        if 0 <= qrs_start and qrs_end <= n:
            sigout[qrs_start:qrs_end] -= qrs_scale * qrs_model

        # --- T-wave subtraction (only if explicitly enabled) ---
        if subtract_t_wave and t_width > 0:
            t_model = qrst[qrs_width:]

            best_t_err = np.inf
            t_scale = 1.0
            t_shift = 0
            t_ok = False

            for shift in range(-_SHIFT_SEARCH_RANGE, _SHIFT_SEARCH_RANGE + 1):
                start = pk + post + 1 + shift
                end = pk + post + t_width + shift + 1
                if start < 0 or end > n:
                    continue
                match_t = sigin[start:end]
                if len(match_t) != len(t_model) or not np.all(np.isfinite(match_t)):
                    continue

                scale = _fit_scale_fixed_offset(t_model, match_t, offset=0.0)
                err = float(np.sum((match_t - scale * t_model) ** 2))

                if shift == -_SHIFT_SEARCH_RANGE or err < best_t_err:
                    best_t_err = err
                    t_scale = scale
                    t_shift = shift
                    t_ok = True

            if t_ok:
                overlap = max(0, qrs_shift - t_shift)
                if overlap < len(t_model):
                    t_trimmed = t_model[overlap:]
                    t_start = pk + post + 1 + t_shift + overlap
                    t_end = t_start + len(t_trimmed)
                    if 0 <= t_start and t_end <= n:
                        sigout[t_start:t_end] -= t_scale * t_trimmed

    removed = sigin - sigout
    return sigout, removed, peaks, qrst, thr_pos


# ---------------------------------------------------------------------------
# Soft notch filter
# ---------------------------------------------------------------------------

def _soft_notch(
    data: NDArray[np.float64],
    sfreq: float,
    freq: float,
    max_db: float,
    sigma_hz: float,
) -> NDArray[np.float64]:
    """Frequency-domain Gaussian soft notch filter."""
    squeezed = False
    arr = np.asarray(data, dtype=np.float64)

    if arr.ndim == 1:
        arr = arr[np.newaxis, :]
        squeezed = True

    n_times = arr.shape[-1]
    freqs = np.fft.rfftfreq(n_times, d=1.0 / sfreq)
    max_gain = 10.0 ** (-max_db / 20.0)
    attenuation = 1.0 - (1.0 - max_gain) * np.exp(
        -0.5 * ((freqs - freq) / sigma_hz) ** 2
    )

    out = np.zeros_like(arr)
    for ch in range(arr.shape[0]):
        spectrum = np.fft.rfft(arr[ch])
        out[ch] = np.fft.irfft(spectrum * attenuation, n=n_times)

    return out.squeeze(axis=0) if squeezed else out


# ---------------------------------------------------------------------------
# Spike detection
# ---------------------------------------------------------------------------

def _detect_spikes(
    sig: NDArray[np.float64],
    sfreq: float,
    segment_sec: float,
    mad_threshold: float,
) -> NDArray[np.bool_]:
    """Detect large spikes in short segments using robust MAD thresholding."""
    n = len(sig)
    segment_len = max(1, int(sfreq * segment_sec))
    n_segments = int(np.ceil(n / segment_len))
    mask = np.zeros(n, dtype=bool)

    for i in range(n_segments):
        start = i * segment_len
        end = min(start + segment_len, n)
        segment = sig[start:end]

        finite = segment[np.isfinite(segment)]
        if finite.size < 3:
            continue

        med = float(np.median(finite))
        mad = float(np.median(np.abs(finite - med)))
        if mad <= 0.0:
            continue

        robust_std = mad * 1.4826
        upper = med + mad_threshold * robust_std
        lower = med - mad_threshold * robust_std
        mask[start:end] = (segment > upper) | (segment < lower)

    return mask


def _apply_spike_mask(
    sig: NDArray[np.float64],
    mask: NDArray[np.bool_],
) -> NDArray[np.float64]:
    """Set spike samples to NaN. No interpolation."""
    out = sig.copy()
    out[mask] = np.nan
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def lfp_xkit(
    fif_path: str | Path,
    *,
    return_removed: bool = False,
    return_raw: bool = False,
    spike_removal: bool = False,
    detect_spikes: bool = True,
    detect_dead_segments: bool = True,
    reconstruct_dead: bool = RECONSTRUCT_DEAD,
    ekg_thr: float | str | None = EKG_THR,
    ekg_polarity: int = EKG_POLARITY,
    ekg_auto_sensitivity: float = EKG_AUTO_SENSITIVITY,
    subtract_t_wave: bool = SUBTRACT_T_WAVE,
    qrs_highpass_hz: float = QRS_HIGHPASS_CUTOFF_HZ,
    spike_segment_sec: float = SPIKE_SEGMENT_SEC,
    spike_mad_thresh: float = SPIKE_MAD_THRESH,
    dead_segment_sec: float = DEAD_SEGMENT_SEC,
    dead_std_ratio: float = DEAD_STD_RATIO,
    min_gap_merge_sec: float = MIN_GAP_MERGE_SEC,
    min_dead_duration_sec: float = MIN_DEAD_DURATION_SEC,
    recon_context_sec: float = RECON_CONTEXT_SEC,
    bandpass_l_freq: float | None = BANDPASS_L_FREQ,
    bandpass_h_freq: float | None = BANDPASS_H_FREQ,
    notch_freq: float | None = NOTCH_FREQ,
    notch_max_db: float = NOTCH_MAX_DB,
    notch_sigma_hz: float = NOTCH_SIGMA_HZ,
) -> LFPCleanResult:
    """Clean an LFP FIF file with Delta-safe QRS-only EKG subtraction.

    Three safeguards protect Delta-band integrity:
      1. T-wave subtraction disabled by default
      2. Baseline offset forced to 0.0
      3. QRS model high-pass filtered at qrs_highpass_hz (default 4 Hz)

    Dead segments are detected via multi-scale approach with morphological
    closing, then reconstructed via spectral matching BEFORE EKG removal
    to ensure continuous data for template subtraction.

    Args:
        fif_path: Path to FIF file.
        return_removed: Include removed EKG signal in output.
        return_raw: Include bandpassed raw signal in output.
        spike_removal: Set detected spikes to NaN in cleaned data.
        detect_spikes: Compute spike masks.
        detect_dead_segments: Detect flat/dead signal segments.
        reconstruct_dead: Reconstruct dead segments via spectral matching.
        ekg_thr: EKG threshold. Float in signal units or "auto".
        ekg_polarity: +1 for positive-going, -1 for negative-going EKG.
        ekg_auto_sensitivity: Z-score multiplier for auto-threshold.
        subtract_t_wave: If True, also subtract T-wave. WARNING: destroys Delta.
        qrs_highpass_hz: High-pass cutoff for QRS template in Hz.
        spike_segment_sec: Window size for spike detection in seconds.
        spike_mad_thresh: MAD multiplier for spike detection.
        dead_segment_sec: Base window size for dead segment detection.
        dead_std_ratio: Std ratio threshold for dead segment detection.
        min_gap_merge_sec: Merge dead regions separated by < this duration.
        min_dead_duration_sec: Minimum duration to flag as dead.
        recon_context_sec: Seconds of healthy context for spectral reconstruction.
        bandpass_l_freq: Lower bandpass frequency. None disables.
        bandpass_h_freq: Upper bandpass frequency. None disables.
        notch_freq: Soft notch center frequency. None disables.
        notch_max_db: Maximum notch attenuation in dB.
        notch_sigma_hz: Gaussian width of soft notch.

    Returns:
        LFPCleanResult dictionary with data, masks, and diagnostics.
    """
    path = Path(fif_path)
    if not path.exists():
        raise FileNotFoundError(f"FIF file not found: {path}")

    logger.info("Loading %s", path.name)
    raw = mne.io.read_raw_fif(str(path), preload=True, verbose=False)

    picks = mne.pick_types(raw.info, eeg=True)
    if len(picks) == 0:
        raise ValueError(f"No EEG channels found in {path.name}")

    roi_names = [raw.ch_names[p] for p in picks]
    sfreq = float(raw.info["sfreq"])

    stem = path.stem
    parts = stem.split("_")
    subject_id = parts[0] if parts else "unknown"

    # Step 1: Bandpass
    if bandpass_l_freq is not None or bandpass_h_freq is not None:
        logger.info(
            "Bandpass %.3g-%.3g Hz",
            bandpass_l_freq if bandpass_l_freq is not None else 0.0,
            bandpass_h_freq if bandpass_h_freq is not None else np.inf,
        )
        raw.filter(
            l_freq=bandpass_l_freq,
            h_freq=bandpass_h_freq,
            picks=picks,
            verbose=False,
        )

    eeg = raw.get_data(picks=picks)

    # Step 2: Soft notch
    if notch_freq is not None and notch_freq > 0.0:
        logger.info(
            "Soft notch at %.3g Hz, max attenuation %.3g dB",
            notch_freq,
            notch_max_db,
        )
        notched = _soft_notch(eeg, sfreq, notch_freq, notch_max_db, notch_sigma_hz)
    else:
        notched = eeg.copy()

    # Step 3: Per-channel processing
    n_channels, n_times = notched.shape
    cleaned = np.zeros_like(notched)
    removed = np.zeros_like(notched)
    spike_masks = np.zeros((n_channels, n_times), dtype=bool)
    dead_masks = np.zeros((n_channels, n_times), dtype=bool)
    recon_masks = np.zeros((n_channels, n_times), dtype=bool)

    ekg_peaks: list[list[int]] = []
    ekg_templates: list[NDArray[np.float64] | None] = []
    ekg_counts: list[int] = []
    ekg_thresholds: list[float] = []

    t_wave_status = "ENABLED" if subtract_t_wave else "DISABLED (Delta-safe)"
    logger.info(
        "T-wave subtraction: %s | QRS high-pass: %.1f Hz",
        t_wave_status,
        qrs_highpass_hz,
    )

    for i, name in enumerate(roi_names):
        # Dead segment detection FIRST (multi-scale with morphological closing)
        if detect_dead_segments:
            dead_masks[i] = _detect_dead_segments(
                notched[i],
                sfreq,
                window_sec=dead_segment_sec,
                std_ratio=dead_std_ratio,
                min_gap_merge_sec=min_gap_merge_sec,
                min_dead_duration_sec=min_dead_duration_sec,
            )

            # SPECTRAL RECONSTRUCTION before EKG removal
            if reconstruct_dead and dead_masks[i].any():
                reconstructed_sig, recon_mask = _reconstruct_dead_segments(
                    notched[i], dead_masks[i], sfreq, context_sec=recon_context_sec
                )
                recon_masks[i] = recon_mask
                # Run EKG removal on reconstructed (continuous) signal
                input_for_ekg = reconstructed_sig
            else:
                input_for_ekg = notched[i].copy()
        else:
            input_for_ekg = notched[i].copy()

        # EKG removal with all three Delta safeguards
        cleaned_ekg, removed_ch, peaks_ch, template_ch, thr_used = _ekg_artifact_removal(
            input_for_ekg,
            fs=sfreq,
            thr=ekg_thr,
            polarity=ekg_polarity,
            auto_sensitivity=ekg_auto_sensitivity,
            subtract_t_wave=subtract_t_wave,
            qrs_highpass_hz=qrs_highpass_hz,
        )

        ekg_peaks.append(peaks_ch)
        ekg_templates.append(template_ch)
        ekg_counts.append(len(peaks_ch))
        ekg_thresholds.append(thr_used)

        logger.info(
            "Channel %s: %d EKG peaks, thr=%.6g, dead=%.1f%%, recon=%.1f%%",
            name,
            len(peaks_ch),
            thr_used,
            100.0 * dead_masks[i].sum() / n_times if detect_dead_segments else 0.0,
            100.0 * recon_masks[i].sum() / n_times if reconstruct_dead else 0.0,
        )

        # Spike detection on cleaned signal
        if detect_spikes:
            mask_ch = _detect_spikes(
                cleaned_ekg, sfreq, spike_segment_sec, spike_mad_thresh
            )
            spike_masks[i] = mask_ch
            if spike_removal:
                cleaned[i] = _apply_spike_mask(cleaned_ekg, mask_ch)
            else:
                cleaned[i] = cleaned_ekg
        else:
            cleaned[i] = cleaned_ekg

        if return_removed:
            removed[i] = removed_ch

    # Assemble result
    result: LFPCleanResult = {
        "data": cleaned[np.newaxis, :, :],
        "roi_names": roi_names,
        "subject_id": subject_id,
        "sfreq": sfreq,
        "ekg_peaks": ekg_peaks,
        "ekg_templates": ekg_templates,
        "ekg_counts": ekg_counts,
        "ekg_thresholds": ekg_thresholds,
    }

    if return_raw:
        result["raw_filtered"] = eeg[np.newaxis, :, :]
    if return_removed:
        result["removed"] = removed[np.newaxis, :, :]
    if detect_spikes:
        result["spike_mask"] = spike_masks[np.newaxis, :, :]
    if detect_dead_segments:
        result["dead_mask"] = dead_masks[np.newaxis, :, :]
    if reconstruct_dead:
        result["reconstructed_mask"] = recon_masks[np.newaxis, :, :]

    logger.info(
        "Done: %s -> shape=%s, sfreq=%.3g Hz, channels=%d",
        path.name,
        result["data"].shape,
        sfreq,
        len(roi_names),
    )
    return result
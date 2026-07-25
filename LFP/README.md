# LFP xkit

**Pre-processing toolkit for single-channel LFP recordings.**

A Python toolkit for preprocessing intracranial Local Field Potential (LFP) data with specific safeguards for preserving  neural oscillations during EKG artifact removal. 


## Framework

```
Raw FIF
  │
  ├─ 1. MNE Bandpass Filter (1–100 Hz, zero-phase FIR)
  │
  ├─ 2. Soft Gaussian Notch (60 Hz, ≤1 dB attenuation)
  │
  ├─ 3. Multi-Scale Dead Segment Detection
  │     └─ Morphological closing + minimum duration filter
  │
  ├─ 4. Spectral Reconstruction of Dead Segments
  │     └─ PSD-matched surrogate data (NOT interpolation)
  │
  ├─ 5. QRS-Only EKG Template Subtraction
  │     ├─ Two-sided Z-score auto-threshold detection
  │     ├─ Zero baseline offset (safeguard #1)
  │     ├─ T-wave disabled by default (safeguard #2)
  │     └─ QRS model high-pass filtered at 4 Hz (safeguard #3)
  │
  └─ 6. Optional Spike Detection (MAD-based, no NaN masking by default)
```


## Quick Start

```python
from lfp_xkit import lfp_xkit
from viz_lfp import viz_lfp

result = lfp_xkit(
    "path/to/subject_condition_lfp_raw.fif",
    return_removed=True,
    return_raw=True,
    detect_dead_segments=True,
    reconstruct_dead=True,
    ekg_thr="auto",
    ekg_auto_sensitivity=2.0,
)

viz_lfp(result)
```

## Configuration Reference

### EKG Removal Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ekg_thr` | `50e-6` | Detection threshold in signal units. Use `"auto"` for per-channel adaptive thresholding. |
| `ekg_auto_sensitivity` | `2.0` | Z-score multiplier for auto-threshold. **Lower = more sensitive.** Range: 1.5–4.0. Start at 2.0; decrease if peak counts < 60 in 80s recording. |
| `ekg_polarity` | `1` | `+1` for positive-going EKG, `-1` for negative-going. |
| `subtract_t_wave` | `False` | **Keep False for Delta analysis.** T-wave occupies 0.5–4 Hz and its removal systematically destroys neural Delta. |
| `qrs_highpass_hz` | `4.0` | High-pass cutoff applied to QRS template before subtraction. Removes phase-locked Delta embedded during averaging. Increase to 5.0 if Delta ratio < 0.85. |

### Dead Segment Detection & Reconstruction

| Parameter | Default | Description |
|-----------|---------|-------------|
| `detect_dead_segments` | `True` | Enable multi-scale flat-signal detection. |
| `dead_segment_sec` | `0.2` | Base window size. Smaller = catches gradual onsets. |
| `dead_std_ratio` | `0.05` | Threshold: std must be > 5% of median healthy std. |
| `min_gap_merge_sec` | `1.0` | Merge detected regions separated by < this duration. Catches partial gaps. |
| `min_dead_duration_sec` | `0.3` | Minimum region length to flag. Prevents false positives from brief neural silences. |
| `reconstruct_dead` | `True` | Fill dead segments with spectrally-matched surrogate data. |
| `recon_context_sec` | `2.0` | Seconds of healthy data before/after gap used for PSD estimation. |

### Spike Detection

| Parameter | Default | Description |
|-----------|---------|-------------|
| `detect_spikes` | `True` | Compute spike masks via MAD thresholding. |
| `spike_removal` | `False` | **Keep False unless validated.** Sets spikes to NaN. No interpolation. |
| `spike_segment_sec` | `0.5` | Window size for local MAD estimation. |
| `spike_mad_thresh` | `8.0` | MAD multiplier. Higher = fewer false positives on neural transients. |

### Filtering

| Parameter | Default | Description |
|-----------|---------|-------------|
| `bandpass_l_freq` | `1.0` | Lower bandpass edge. Set to `0.5` for full Delta; `1.0` reduces drift. |
| `bandpass_h_freq` | `100.0` | Upper bandpass edge. |
| `notch_freq` | `60.0` | Line noise frequency. Set to `50.0` for European recordings. |
| `notch_max_db` | `1.0` | Maximum notch attenuation. Soft Gaussian profile prevents ringing. |
| `notch_sigma_hz` | `2.0` | Notch bandwidth. Narrower = less collateral attenuation. |

## Output Structure

`lfp_xkit()` returns an `LFPCleanResult` dictionary:

| Key | Shape | Description |
|-----|-------|-------------|
| `data` | `(1, C, T)` | Cleaned LFP tensor |
| `roi_names` | `(C,)` | Channel names |
| `subject_id` | `str` | Parsed from filename |
| `sfreq` | `float` | Sampling frequency |
| `raw_filtered` | `(1, C, T)` | Bandpassed raw (if `return_raw=True`) |
| `removed` | `(1, C, T)` | Subtracted EKG signal (if `return_removed=True`) |
| `dead_mask` | `(1, C, T)` | Boolean mask of detected dead segments |
| `reconstructed_mask` | `(1, C, T)` | Boolean mask of spectrally reconstructed samples |
| `spike_mask` | `(1, C, T)` | Boolean mask of detected spikes |
| `ekg_peaks` | `list[list[int]]` | Per-channel EKG peak indices |
| `ekg_templates` | `list[ndarray]` | Per-channel averaged QRST templates |
| `ekg_counts` | `list[int]` | Per-channel detected EKG peak counts |
| `ekg_thresholds` | `list[float]` | Per-channel thresholds used |

## Validation Checklist

After running on a new dataset, verify:

1.  **EKG peak counts**: Should be ~60–120 for an 80s recording at normal heart rate
2.  **Delta ratio**: Log output should show 0.85–1.05 for all channels
3.  **Removed trace**: Should show regularly spaced QRST morphology, not random noise
4.  **Dead segments**: Orange dots in visualization should cover entire flat regions
5.  **Reconstructed mask**: Should exactly match dead mask when `reconstruct_dead=True`



### Spectral Reconstruction vs. Interpolation

Dead segments are time-domain gaps with zero variance. Linear/cubic interpolation creates smooth curves that correspond to artificial low-frequency energy in the frequency domain, corrupting Delta analysis.

Spectral reconstruction instead:
- Estimates PSD from 2s of healthy surrounding data
- Generates surrogate data with matched amplitude spectrum but random phase
- Applies smooth fade-in/fade-out at boundaries
- Preserves statistical properties without inventing phase-locked oscillations


## Files

| File | Purpose |
|------|---------|
| `lfp_xkit.py` | Core cleaning library. No plotting. |
| `viz_lfp.py` | Visualization module. Separate from cleaning logic. |
| `README.md` | This file. |


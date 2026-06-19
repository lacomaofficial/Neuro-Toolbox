# iMat: A Toolkit for Exploring and Analyzing Matlab PD iEEG Data

## Overview

**iMat** is a Python toolkit designed for exploring, extracting, and analyzing intracranial EEG (iEEG) data from Parkinson's Disease (PD) patients. It provides utilities for working with two primary data formats:
- **Percept** files (Medtronic Percept PC deep brain stimulation devices)
- **RCS** files (Abbott Infinity DBS systems)

The toolkit handles common challenges in clinical iEEG data, including NaN interpolation, channel length mismatches, detrending, and quality assessment.



---

## Quick Start

### 1. Inspect a MATLAB File

Before extracting data, inspect the file structure to understand its contents:

```python
from imat import inspect_mat_file

filepath = '/path/to/your/file.mat'
inspect_mat_file(filepath)
```

This will display:
- File size
- Top-level keys in the `.mat` file
- Nested structure details (up to specified depth)

### 2. Extract Percept Data

```python
from imat import extract_percept_data

# Extract all conditions
data_dict = extract_percept_data('Percept_JSON_parsed_Left.mat')

# Extract specific condition (case-insensitive)
data_dict = extract_percept_data('Percept_JSON_parsed_Left.mat', 
                                  condition_keyword='bima')
```

**Returns:** Dictionary mapping condition names to:
- `data`: LFP signal array
- `sfreq`: Sampling frequency (Hz)
- `channel`: Channel identifier
- `timestamp`: Recording timestamp
- `duration_sec`: Duration in seconds

### 3. Extract RCS Data

```python
from imat import extract_rcs_data

# Basic extraction
rcs_data = extract_rcs_data('RCS_DBSOFF_bima.mat')

# With detrending (recommended for PSD analysis)
rcs_data = extract_rcs_data('RCS_DBSOFF_bima.mat', detrend=True)
```

**Returns:** Dictionary with:
- `sfreq`: Sampling frequency (Hz)
- `channels`: Dictionary mapping channel names to signal arrays

Available channels:
- `STN-L`, `STN-R` (subthalamic nucleus)
- `ECOG-8-9-L`, `ECOG-10-11-L` (left cortical electrodes)
- `ECOG-8-9-R`, `ECOG-10-11-R` (right cortical electrodes)

### 4. Visualize Data

#### Raw Trace Plot

```python
from imat import plot_raw_trace

plot_raw_trace(data, sfreq, title='Raw LFP', duration=10, ch_names=['STN-L'])
```

#### Power Spectral Density (PSD)

```python
from imat import plot_psd

plot_psd(data, sfreq, title='PSD Analysis', 
         ch_names=['STN-L'], fmax=50, interpolate_nans=True)
```

Features:
- Automatic NaN interpolation (critical for Welch's method)
- Beta band (13-30 Hz) highlighted in red
- Log-scale power display

#### Inter-channel Coherence

```python
from imat import plot_coherence

plot_coherence(signal_left, signal_right, sfreq, 
               title='STN-L ↔ STN-R Coherence')
```

Outputs beta-band coherence value to console.

### 5. Quality Metrics

```python
from imat import compute_quality_metrics

metrics = compute_quality_metrics(data, sfreq)
```

**Returns per channel:**
- `duration_sec`: Recording duration
- `nan_pct`: Percentage of NaN values
- `std_uv`: Standard deviation (μV)
- `beta_power`: Mean power in beta band (13-30 Hz)
- `theta_power`: Mean power in theta band (4-8 Hz)
- `beta_theta_ratio`: Ratio of beta to theta power

---

## Example Workflows

### Example 1: Inspect and Extract Percept Data

```python
from imat import example_inspect_percept, example_extract_and_plot

# Inspect file structure
example_inspect_percept()

# Extract 'bima' condition and visualize
example_extract_and_plot()
```

### Example 2: Bilateral STN Analysis

```python
from imat import example_bilateral_analysis

# Merge left and right STN recordings
# Plot raw traces, PSDs, and inter-hemispheric coherence
example_bilateral_analysis()
```

### Example 3: RCS Data Extraction

```python
from imat import example_rcs_extraction

# Extract sbj1 RCS data (DBS OFF, bima condition)
example_rcs_extraction()
```

---

## Key Features

### Robust Data Handling

- **NaN Interpolation**: Linear interpolation for missing values (<5% NaN threshold validated)
- **Length Mismatch Correction**: Automatic truncation to shortest channel for multi-channel RCS data
- **Detrending**: Optional linear trend removal for improved spectral analysis
- **Type Safety**: Comprehensive input validation with informative error messages

### Analysis Capabilities

- **Welch's PSD**: Power spectral density estimation with configurable parameters
- **Coherence Analysis**: Inter-channel coherence with beta-band summary statistics
- **Quality Assessment**: Automated metrics for data usability evaluation

### Visualization

- Multi-channel raw trace plotting
- PSD plots with annotated frequency bands
- Coherence plots with statistical summaries
- Customizable titles, labels, and display parameters

---

## API Reference

### Core Functions

| Function | Description |
|----------|-------------|
| `inspect_mat_file(filepath, max_depth=2)` | Inspect `.mat` file structure |
| `extract_percept_data(filepath, condition_keyword=None)` | Extract Percept LFP data |
| `extract_rcs_data(filepath, detrend=False)` | Extract RCS LFP data |
| `plot_raw_trace(data, sfreq, ...)` | Plot raw time-domain signals |
| `plot_psd(data, sfreq, ...)` | Plot power spectral density |
| `plot_coherence(data1, data2, sfreq)` | Plot inter-channel coherence |
| `compute_quality_metrics(data, sfreq)` | Compute data quality metrics |

### Helper Functions

| Function | Description |
|----------|-------------|
| `_interpolate_nans(data)` | Linear interpolation for NaN values |
| `_inspect_recursive(val, depth, max_depth)` | Recursive structure inspection |

### Configuration

| Variable | Description |
|----------|-------------|
| `BASE_PATH` | Root directory for PD iEEG data |
| `__version__` | Current version (1.2) |

---

## Data Format Notes

### Percept Files
- Structure: `BrainSenseTimeDomain` nested within parsed JSON struct
- Conditions identified via `Notes` field
- May have separate Left/Right hemisphere files
- Typical sampling rate: ~250 Hz

### RCS Files
- Structure: `SDATA` with hemisphere-specific substructures
- Channels accessed via predefined mapping
- May contain unilateral recordings (missing channels handled gracefully)
- Default sampling rate fallback: 250 Hz

---

## Troubleshooting

### Common Issues

**"No BrainSenseTimeDomain found"**
- Verify file contains Percept data structure
- Use `inspect_mat_file()` to examine actual structure

**"No SDATA struct found"**
- Verify file contains RCS data structure
- Check file naming conventions

**NaN warnings in PSD computation**
- Enable `interpolate_nans=True` in `plot_psd()`
- Verify NaN percentage <5% using `compute_quality_metrics()`

**Length mismatch in multi-channel data**
- Automatically handled by `extract_rcs_data()`
- Warning message indicates truncation occurred

---

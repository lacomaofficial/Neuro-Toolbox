# Neuro-Toolbox

A curated collection of tutorials, templates, and reusable frameworks for EEG signal processing, fMRI machine learning, and psychometric assessment.


<br>

### EEG
- **Preprocessing pipeline**: 4-step walkthrough using MEEGKit (ASR, STAR, SNS) and ICLabel for automatic artifact removal
- **Artifact detection**: Notebooks for identifying artifacts in scalp EEG and intracranial LFP recordings
- **Machine learning**: XGBoost-based feature extraction and classification on EEG-derived features

### fMRI
- **ADHD classification**: SVM with linear and RBF kernels on resting-state functional connectivity
- **ASD classification**: Transformer encoder with multi-modal inputs (fMRI + demographic features)
- **PyTorch tutorials**: Ready-to-adapt templates for classification, regression, and graph-based deep learning on neuroimaging data

### PSY
- **Psychometric**: Five-Factor personality assessment tools (English & Spanish)

<br>

## Related Packages

| Package | Description |
|---------|-------------|
| [`xeeg_kit`](https://github.com/cimt-unia/xeeg_kit) | HD-EEG artifact removal pipeline |
| [`lcmv_xtra`](https://github.com/cimt-unia/lcmv_xtra) | LCMV beamforming and source connectivity |
| [`gt_map`](https://github.com/cimt-unia/gt_map) | Glasser+Tian fMRI parcellation |

<br>

## Requirements

- MNE-Python
- Nilearn
- PyTorch
- Scikit-learn
- XGBoost

<br>

## License

MIT — use freely for academic and educational purposes.

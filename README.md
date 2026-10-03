# Audio Deepfake Detection Using Classical Machine Learning

A Python-based project for detecting synthetic speech using statistical audio analysis and classical machine learning techniques.

This project was developed as the practical component of a university coursework paper titled *Deepfakes in IT Security: Analysis of Threats and Detection Methods* (2026).

The objective is to investigate acoustic differences between authentic and synthetically generated speech and evaluate the effectiveness of traditional machine learning methods for audio deepfake detection.

## 1. Project Overview

The project uses the **ASVspoof 2019 Logical Access (LA)** dataset to compare authentic (bonafide) and synthetically generated (spoof) speech recordings.

The analysis consists of four main stages:

1. Audio preprocessing and feature extraction.
2. Statistical analysis of acoustic differences.
3. Training and comparing machine learning classifiers.
4. Evaluating classification performance and feature importance.

The experiment uses 600 audio recordings, consisting of 300 authentic and 300 synthetic samples.

## 2. Technologies

**Programming Language:** Python

**Libraries:**
- NumPy and Pandas — Data processing
- Librosa — Audio processing and feature extraction
- SciPy — Statistical analysis
- Scikit-learn — Machine learning and model evaluation
- Matplotlib — Data visualization

## 3. Methodology

### Dataset

The experiments use a balanced subset of the ASVspoof 2019 Logical Access training dataset.

- 300 authentic speech recordings
- 300 synthetic speech recordings
- Sampling rate: 16 kHz
- Random seed: 42

### Audio Feature Extraction

Several acoustic features are extracted to investigate differences between authentic and synthetic speech.

**Spectral Features**
- Spectral Centroid
- Spectral Bandwidth
- Spectral Rolloff
- Spectral Flatness

**Cepstral Features**
- 13 Mel-Frequency Cepstral Coefficients (MFCCs)
- Delta-MFCCs
- Statistical summaries of individual coefficients

**Pitch Features**
- Mean fundamental frequency
- Pitch standard deviation
- Pitch range
- Voiced fraction

Additional features describe the relative energy distribution across different frequency bands.

### Statistical Analysis

The Mann-Whitney U test is used to compare acoustic characteristics between authentic and synthetic speech.

Rank-biserial correlation is calculated to estimate effect sizes.

### Machine Learning

Three classical machine learning classifiers are evaluated:

- Logistic Regression
- Support Vector Machine (RBF kernel)
- Random Forest

The dataset is divided into 80% training data and 20% held-out test data.

Model comparison and selection are performed using five-fold stratified cross-validation on the training data.

The selected model is subsequently trained on the complete training partition and evaluated on the held-out test set.

## 4. Experimental Results

### Cross-Validation Results

The updated experiment produced the following mean cross-validation results on the training partition:

| Model | Accuracy | ROC-AUC |
|---|---:|---:|
| Logistic Regression | 86.67% | 0.925 |
| Support Vector Machine | 92.71% | 0.982 |
| Random Forest | 90.63% | 0.958 |

The Support Vector Machine achieved the highest mean cross-validation accuracy and ROC-AUC among the evaluated models.

### Held-Out Test Evaluation

The selected SVM achieved **95% classification accuracy** on the held-out test dataset containing 120 audio recordings.

The confusion matrix was:

| Actual / Predicted | Bonafide | Spoof |
|---|---:|---:|
| Bonafide | 59 | 1 |
| Spoof | 5 | 55 |

The model correctly classified 114 out of 120 recordings.

However, five synthetic recordings were incorrectly classified as authentic.

### Feature Importance

A separate Random Forest feature importance analysis identified several influential acoustic features.

The two highest-ranked features were:

- MFCC coefficient 4 standard deviation: 6.99%
- Delta-MFCC coefficient 4 standard deviation: 6.22%

These findings suggest that the temporal variability of acoustic characteristics provides useful information for distinguishing authentic and synthetic speech within the selected dataset.

## 5. Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/audio-deepfake-detection.git
cd audio-deepfake-detection
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Dataset Setup

Download the ASVspoof 2019 Logical Access dataset from the official source:

https://www.asvspoof.org/

The project requires the LA training audio files and the corresponding protocol file.

Before running the experiment, update `base_dir` in `ExperimentConfig` to match the local location of your dataset.

The expected dataset structure is:

```text
LA/
├── ASVspoof2019_LA_cm_protocols/
│   └── ASVspoof2019.LA.cm.train.trn.txt
└── ASVspoof2019_LA_train/
    └── flac/
```

The dataset is not included in this repository.

## 6. Running the Experiment

Execute the main Python script:

```bash
python audio_artifact_analysis.py
```

The script performs feature extraction, statistical testing, visualization, cross-validation, model selection and final test evaluation.

It generates CSV files containing experimental results and PNG files containing visualizations.

Output directories include:

- `data/` — Selected samples and extracted features
- `tables/` — Statistical tests and machine learning results
- `figures/` — Visualizations and model evaluation plots

## 7. Limitations

The experiments use a relatively small balanced subset of the ASVspoof 2019 Logical Access dataset.

The reported classification accuracy applies only to the selected experimental dataset and evaluation procedure.

The model has not been independently evaluated against newer speech synthesis systems or external datasets.

Additionally, the train/test split is stratified by class rather than explicitly separated by speaker identity.

Therefore, the reported accuracy should not be interpreted as evidence of equivalent performance in real-world applications.

## 8. Academic Context

This project was developed as part of an individual university coursework paper on deepfake threats and detection techniques in IT security.

The theoretical component reviewed deepfake generation technologies, cybersecurity threats and existing detection approaches.

The practical component investigated acoustic differences between authentic and synthetic speech and evaluated classical machine learning methods.

The implementation demonstrates a reproducible experimental approach to audio feature extraction, statistical analysis and supervised classification.

## 9. Dataset Acknowledgment

This project uses samples from the ASVspoof 2019 dataset.

The dataset belongs to its respective creators and contributors. Please consult the official ASVspoof website for dataset access, documentation and applicable usage conditions.

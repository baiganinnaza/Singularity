# Singularity: Gravitational Wave Detection & Classification

## Overview
**Singularity** is a scientific machine learning pipeline designed for the detection and classification of Gravitational Waves (GW) utilizing data from the Laser Interferometer Gravitational-Wave Observatory (LIGO). The system employs a hybrid architecture combining unsupervised and supervised learning techniques to isolate signals from noise and classify them accurately.

## Methodology

The pipeline consists of two primary stages:

1.  **Unsupervised Anomaly Detection (AE)**:
    *   Utilizes a **Autoencoder (AE)** to learn the latent representation of background noise.
    *   Detects potential gravitational wave events as anomalies (high reconstruction loss) when they deviate from the learned background distribution.
    *   Includes Latent Space analysis via PCA and Amplitude Spectral Density (ASD) validation.

2.  **Supervised Classification (EfficientNet)**:
    *   Extracts candidate events and converts them into time-frequency spectrograms.
    *   Uses a transfer-learning approach with **EfficientNetB0** to classify inputs into specific categories (e.g., Chirps/Mergers, Glitches, Background).
    *   Validated with confusion matrices and class distribution analysis.

## Datasets

This project relies on the following datasets:

| Dataset | Description | Link |
| :--- | :--- | :--- |
| **Fixed Processed Data** | Pre-processed spectrograms and signal chunks ready for training. | [Kaggle Link](https://www.kaggle.com/datasets/azamatbaiganin/fixed-processed-ligo-data) |
| **Raw LIGO Data** | Raw HDF5 time-series data from LIGO detectors (used for background injection). | [Kaggle Link](https://www.kaggle.com/datasets/azamatbaiganin/raw-ligo-dataset) |

## Repository Structure

```text
Singularity/
├── src/
│   ├── singularity_pipeline.ipynb  # End-to-end pipeline (Injection -> Detection -> Classification)
│   ├── ligo-classifier.ipynb       # EfficientNet classifier training and validation
│   └── vae-ligo (1).ipynb          # VAE training for anomaly detection
├── models/                         # Saved model weights (.keras, .h5)
├── images/                         # Generated plots and visualization assets
└── README.md                       # Project documentation
```

## Scientific Visualization
The codebase generate publication-ready figures, including:
*   **Power Spectral Density (PSD)**: To ensure noise properties match physical reality.
*   **Spectrograms**: Log-scaled time-frequency representations (vmin=-11, vmax=-7).
*   **Latent Space PCA**: Visualizing the separation between noise and signals in 2D space.
*   **Reconstruction Loss Histograms**: Statistical bounds for anomaly triggers.

## Requirements
*   Python 3.8+
*   TensorFlow
*   SciPy, NumPy, Pandas
*   Matplotlib, Seaborn
*   h5py

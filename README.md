# Singularity: Gravitational Wave Detection & Classification

## Overview
**Singularity** is a scientific machine learning pipeline designed for the detection and classification of Gravitational Waves (GW) utilizing data from the Laser Interferometer Gravitational-Wave Observatory (LIGO). The system employs a hybrid architecture combining unsupervised and supervised learning techniques.

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
│   ├── singularity_pipeline.ipynb  # End-to-end pipeline
│   ├── ligo-classifier.ipynb       # Classifier training
│   └── vae-ligo (1).ipynb          # VAE training
├── models/                         # Saved model weights
├── images/                         # Generated plots
└── README.md                       # Project documentation
```

# Singularity: Deep Learning Pipeline for Gravitational Wave Detection

## 1. Project Goal & Scientific Objective
The primary objective of **Singularity** is to automate the detection and classification of Gravitational Wave (GW) signals buried within the highly noisy data streams of the Laser Interferometer Gravitational-Wave Observatory (LIGO).

LIGO data is characterized by a low signal-to-noise ratio (SNR) and the presence of non-Gaussian transient noise artifacts known as "glitches." Traditional matched-filtering methods are computationally expensive and can be sensitive to templates. **Singularity** proposes a hybrid Deep Learning approach:
1.  **Filter out trivial background noise** using an unsupervised anomaly detector (VAE).
2.  **Classify remaining anomalies** using a robust supervised classifier (EfficientNet).

The goal is to provide a low-latency, high-accuracy triggers system that distinguishes real astrophysical events (Black Hole Mergers) from terrestrial instrument noise.

---

## 2. System Architecture
The pipelined architecture works in two distinct stages, processing time-series strain data sampled at 4096 Hz.

### Stage I: Anomaly Detection (Variational Autoencoder)
The first filter is a **Variational Autoencoder (VAE)** trained exclusively on *background noise*.
*   **Input**: Raw time-series segments (Windowed strain data).
*   **Mechanism**: The VAE compresses the input into a lower-dimensional latent space and attempts to reconstruct it. Since it learns the statistical properties of "normal" background noise, it fails to accurately reconstruct anomalous waveforms (signals or glitches).
*   **Trigger**: A high **Reconstruction Loss** (Mean Squared Error) between the input and output acts as a trigger, flagging the segment as a "Candidate Event."

### Stage II: Event Classification (EfficientNetB0)
Once a candidate event is triggered, it is passed to the classification engine.
*   **Preprocessing**: The 1-second time-series candidate is converted into a **Log-Spectrogram** (`vmin=-11`, `vmax=-7`, `nperseg=128`), aligning the data with the visual domain.
*   **Model**: A fine-tuned **EfficientNetB0** (Transfer Learning from ImageNet).
*   **Output**: A probability distribution over defined classes:
    *   `Chirp` (Simulated Black Hole Merger)
    *   `Glitches` (Transient Noise)
    *   `Background` (False Alarm)
    *   `Air_Compressor` (Specific Hardware Noise)

---

## 3. Development Methodology

### Data Engineering & Simulation
Since detection of real GW events is rare, the project relies on **Signal Injection**:
*   **Background**: Real HDF5 strain data from LIGO (Hanford/Livingston detectors).
*   **Injections**: Synthetic **Quadratic Chirps** are injected into the background to simulate inspiraling binary black holes.
*   **Whitening & Normalization**: Data is preprocessed to remove spectral lines and normalize amplitude variance.

### Scientific Validation
To ensure the pipeline is scientifically rigorous, the following validation steps were implemented:
*   **Power Spectral Density (PSD)**: Verified that the noise floor of processed data matches LIGO's sensitivity curves.
*   **Latent Space Analysis**: Used PCA to visualize how the VAE clusters "Background" vs "Signals" in the latent vector space.
*   **Spectrogram Consistency**: Fixed domain shifts by strictly aligning spectrogram generation parameters (Log10 scale, dB limits) between the training set and the inference pipeline.

---

## 4. Datasets
The project utilizes data sourced from the LIGO Open Science Center (via Kaggle):

| Dataset | Description | Link |
| :--- | :--- | :--- |
| **Fixed Processed Data** | Curated training set containing spectrograms of Signals, Background, and Glitches. | [Kaggle Link](https://www.kaggle.com/datasets/azamatbaiganin/fixed-processed-ligo-data) |
| **Raw LIGO Data** | Large-scale HDF5 files containing raw strain recordings (used for background modeling). | [Kaggle Link](https://www.kaggle.com/datasets/azamatbaiganin/raw-ligo-dataset) |

---

## 5. Repository Structure
```text
Singularity/
├── src/
│   ├── singularity_pipeline.ipynb  # MAIN PIPELINE: Loads real data, runs VAE + Classifier
│   ├── ligo-classifier.ipynb       # TRAINING: EfficientNet supervised model generation
│   └── vae-ligo (1).ipynb          # TRAINING: VAE anomaly detector generation
├── models/                         # Serialized weights (.keras)
├── images/                         # Visual outputs (Confusion Matrices, Latent Plots)
└── README.md                       # Documentation
```

## 6. Requirements
*   **Python 3.8+**
*   **TensorFlow / Keras** (Deep Learning framework)
*   **SciPy** (Signal processing: spectrograms, windows, filtering)
*   **NumPy & Pandas** (Data manipulation)
*   **h5py** (LIGO data file handling)
*   **Matplotlib & Seaborn** (Scientific Visualization)

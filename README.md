# Speech Emotion Recognition using SVD and HMM

## Overview
This project implements **Speech Emotion Recognition (SER)** using **Singular Value Decomposition (SVD)** and **Hidden Markov Models (HMM)**. The system is designed to classify emotions from speech signals using three different datasets: **Berlin**, **Normal**, and **Urdu**. The implementation is done in **MATLAB** and is structured into separate folders based on datasets.

## Project Structure
```
├── SEP_Berlin
│   ├── train_data/         # Folder containing training data for the Berlin dataset
│   ├── database.mat        # Preprocessed dataset for Berlin
│   ├── final.m             # Main script for training and testing
│   ├── hmm.m               # HMM training and classification
│   ├── hmmdecode.m         # HMM decoding for classification
│
├── SEP_Normal
│   ├── train_data/         # Folder containing training data for the Normal dataset
│   ├── database.mat        # Preprocessed dataset for Normal
│   ├── final.m             # Main script for training and testing
│   ├── hmm.m               # HMM training and classification
│   ├── hmmdecode.m         # HMM decoding for classification
│
├── SEP_Urdu
│   ├── train_data/         # Folder containing training data for the Urdu dataset
│   ├── database.mat        # Preprocessed dataset for Urdu
│   ├── final.m             # Main script for training and testing
│   ├── hmm.m               # HMM training and classification
│   ├── hmmdecode.m         # HMM decoding for classification
```

## Features
- **Speech Emotion Recognition** using **HMM** and **SVD**.
- Three datasets: **Berlin**, **Normal**, and **Urdu**.
- MATLAB implementation with structured code.
- **Feature extraction**, **training**, and **classification** implemented.

## Installation & Setup
### Requirements
Ensure you have **MATLAB** installed on your system before running the project.

### Steps to Run
1. Clone the repository:
   ```sh
   git clone https://github.com/Koushik7893/Speech-Emotion-Recognition-using-SVD-and-HMM.git
   ```
2. Open MATLAB and navigate to the project folder.
3. Choose a dataset folder (e.g., `SEP_Berlin`).
4. Run the `final.m` script:
   ```matlab
   run('final.m')
   ```
   This script will train the model and test it on the given dataset.
5. The classification results will be displayed in MATLAB.

## How It Works
1. **Data Preprocessing**:
   - Speech signals are processed and stored in `database.mat`.
2. **Feature Extraction**:
   - Features are extracted from speech signals using **Singular Value Decomposition (SVD)**.
3. **Training**:
   - Hidden Markov Model (HMM) is trained using extracted features.
4. **Classification**:
   - HMM classifies speech emotions based on trained models.

## Contributing
If you'd like to contribute, feel free to fork the repository and submit a pull request!

## License
This project is licensed under the **Apache License**.

---



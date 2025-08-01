# Deep Learning Log Analysis with Autoencoder & XGBoost

This project demonstrates an end-to-end pipeline for anomaly detection and classification on the UNSW-NB15 cybersecurity dataset using a combination of deep learning (Autoencoder) and gradient boosting (XGBoost). The workflow includes data preprocessing, unsupervised anomaly detection, supervised classification, model evaluation, and inference.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Workflow](#workflow)
- [Requirements](#requirements)
- [Usage](#usage)
- [Model Inference](#model-inference)
- [Results](#results)
- [Files](#files)
- [References](#references)

---

## Project Overview

- **Goal:** Detect and classify network intrusions using log data.
- **Approach:**  
  1. Use an Autoencoder to learn normal (benign) traffic patterns and detect anomalies via reconstruction error.
  2. Use XGBoost for supervised classification, leveraging both original features and autoencoder reconstruction errors.

---

## Dataset

- **Source:** [UNSW-NB15](https://research.unsw.edu.au/projects/unsw-nb15-dataset)
- **Files Used:**  
  - `UNSW_NB15_training-set.csv`
  - `UNSW_NB15_testing-set.csv`
- **Features:** Network traffic attributes, labels for benign/malicious.

---

## Workflow

1. **Data Preprocessing**
   - Drop unnecessary columns (`id`, `attack_cat`, `label`).
   - One-hot encode categorical features.
   - Align train and test feature columns.
   - Standardize features using `StandardScaler`.

2. **Autoencoder Training**
   - Train an Autoencoder neural network on benign samples.
   - Use early stopping to prevent overfitting.
   - Visualize training loss.

3. **Anomaly Detection**
   - Compute reconstruction error for test samples.
   - Classify samples as anomalous if error exceeds a threshold.

4. **Supervised Classification**
   - Concatenate original features with reconstruction error.
   - Train XGBoost classifier on combined features.
   - Evaluate using confusion matrix and classification report.
   - Visualize feature importance.

5. **Model Saving**
   - Save trained Autoencoder, XGBoost model, scaler, and threshold for future inference.

6. **Inference Pipeline**
   - currently developing..

---

## Usage

1. **Download the Dataset:**  
   Download CSV files from the [UNSW-NB15 dataset page](https://research.unsw.edu.au/projects/unsw-nb15-dataset) and place them in the `dataset/` directory.

2. **Run the Notebook:**  
   Open `dl_model.ipynb` in Jupyter or VS Code and run all cells sequentially.

3. **Training & Evaluation:**  
   - The notebook will preprocess data, train the Autoencoder, compute reconstruction errors, train XGBoost, and evaluate results.

4. **Model Saving:**  
   - Models and scaler are saved in the `saved_model/` directory for later use.

---


## Results

- **Autoencoder:** Detects anomalies based on reconstruction error.
- **XGBoost:** Improves classification by combining original features and autoencoder error.
- **Evaluation:** Confusion matrix and classification report are printed for both models.
- **Feature Importance:** Visualized for XGBoost.

---


## References

- [UNSW-NB15 Dataset](https://research.unsw.edu.au/projects/unsw-nb15-dataset)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [XGBoost Documentation](https://xgboost.readthedocs.io/en/stable/)
- [scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)

---

**For questions or improvements, please open an issue
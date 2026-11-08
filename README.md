# Autoencoder for Sensor Data Analysis

## Project Overview
This project explores the use of Autoencoders for analysing vibration sensor data from the **Hell Bridge Test Arena (HBTA)** dataset, with a focus on identifying potential structural damage through reconstruction errors.

Developed as part of the **Master of Computing (COMP6016)** at Curtin University.

## Methodology
- Preprocessed vibration data from 17 acceleration sensor channels.
- Applied standardisation and 500-sample window segmentation.
- Implemented a fully connected **Dense Autoencoder** trained on healthy sensor data.
- Evaluated eight structural damage conditions (DS1–DS8).
- Measured performance using reconstruction errors and ROC-AUC scores.

## Results

| Evaluation | Mean ROC-AUC |
|---|---:|
| Held-out healthy test data | 0.6172 |
| Separate healthy recording | 0.4969 |

The results indicate limited damage detection performance, particularly when compared against a separate healthy recording.

## Repository Structure
- **Final baseline:** Final Dense Autoencoder implementation.
- **Previous_experiments:** Earlier exploratory experiments, including Conv1D Autoencoders.
- **Results:** Evaluation results, training history, and visualisations.

## Technologies
Python, TensorFlow/Keras, Scikit-learn, NumPy, Pandas, Matplotlib, and Jupyter Notebook.

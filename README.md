# Fault-diagnosis-2D
Intelligent Fault Classification Across Different Rolling Bearings Using Envelope Morphology

# Intelligent Fault Classification Across Different Rolling Bearings

## Introduction
This project aims to classify faults in rolling bearings through envelope morphology analysis. 
Utilizing data from sensor readings, the project compares traditional envelope spectrum graphs with Continuous Wavelet Transform (CWT) time-frequency representations to demonstrate superior experimental performance.

## Files
- `train.py`: Training script using envelope spectrum images generated from one-dimensional sensor data for each bearing fault.
- `test.py`: Testing script corresponding to the training process.
- `net.py`: Defines the CNN model tailored for fault diagnosis, including hyperparameter optimization and loss function.
- `CWT_train.py`: Training script using CWT time-frequency representations for comparison.
- `CWT_test.py`: Testing script for the CWT approach.

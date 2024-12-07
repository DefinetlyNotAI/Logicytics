# VulnScan Documentation

## Overview

VulnScan is designed to detect sensitive data across various file formats. It offers a modular framework to train models using diverse algorithms, from traditional ML classifiers to advanced Neural Networks. This document outlines the system's naming conventions, lifecycle, and model configuration.

> The model that is being used is `Model SenseMini 3n3` with a vectorizer from `tools/_vectorizer.py` (Used the random dataset)

---

## Naming Conventions

### Model Naming Format
`Model {Type of model} .{Version}`

- **Type of Model**: Describes the training data configuration.
  - `Sense`: Sensitive data set with 50k files, each 50KB in size.
  - `SenseNano`: Test set with 5-10 files, each 5KB, used for error-checking.
  - `SenseMacro`: Large dataset with 1M files, each 10KB. This is computationally intensive, so some corners were cut in training.
  - `SenseMini`: Dataset with 10K files, each between 10-200KB. Balanced size for effective training and resource efficiency.

- **Version Format**: `{Version#}{c}{Repeat#}`
  - **Version#**: Increment for major code updates.
  - **c**: Model identifier (e.g., NeuralNetwork, BERT, etc.). See below for codes.
  - **Repeat#**: Number of times the same model was trained without significant code changes, used to improve consistency.
  - **-F**: Denotes a failed model or a corrupted model.

### Model Identifiers

| Code | Model Type                |
|------|---------------------------|
| `b`  | BERT                      |
| `dt` | DecisionTree              |
| `et` | ExtraTrees                |
| `g`  | GBM                       |
| `l`  | LSTM                      |
| `n`  | NeuralNetwork (preferred) |
| `nb` | NaiveBayes                |
| `r`  | RandomForestClassifier    |
| `lr` | Logistic Regression       |
| `v`  | SupportVectorMachine      |
| `x`  | XGBoost                   |

### Example
`Model Sense .1n2`: 
- Dataset: `Sense` (50k files, 50KB each).
- Version: 1 (first major version).
- Model: `NeuralNetwork` (`n`).
- Repeat Count: 2 (second training run with no major code changes).

---

## Life Cycle Phases

### Version 1 (Deprecated)
- **Removed**: Small and weak codebase, replaced by `v3`.

1. Generate data.
2. Index paths.
3. Read paths.
4. Train models and iterate through epochs.
5. Produce outputs: data, graphs, and `.pkl` files.

---

### Version 2 (Deprecated)
- **Deprecation Reason**: Outdated methods for splitting and vectorizing data.

1. Load Data.
2. Split Data.
3. Vectorize Text.
4. Initialize Model.
5. Train Model.
6. Evaluate Model.
7. Save Model.
8. Track Progress.

---

### Version 3 (Current)
1. **Read Config**: Load model and training parameters.
2. **Load Data**: Collect and preprocess sensitive data.
3. **Split Data**: Separate into training and validation sets.
4. **Vectorize Text**: Transform textual data using `TfidfVectorizer`.
5. **Initialize Model**: Define traditional ML or Neural Network models.
6. **Train Model**: Perform iterative training using epochs.
7. **Validate Model**: Evaluate with metrics and generate classification reports.
8. **Save Model**: Persist trained models and vectorizers for reuse.
9. **Track Progress**: Log and visualize accuracy and loss trends over epochs.

---

## Preferred Model
**NeuralNetwork (`n`)**  
- Proven to be the most effective for detecting sensitive data in the project.

---

## Notes
- **Naming System**: Helps track model versions, datasets, and training iterations for transparency and reproducibility.
- **Current Focus**: Transition to `v3` for improved accuracy, flexibility, and robust performance.

---

## Additional Features

- **Progress Tracking**: Visualizes accuracy and loss per epoch with graphs.
- **Error Handling**: Logs errors for missing files, attribute issues, or unexpected conditions.
- **Extensibility**: Supports plug-and-play integration for new algorithms or datasets.

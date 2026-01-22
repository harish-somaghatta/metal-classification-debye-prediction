# Metal/Non-Metal Classification + Debye Temperature Prediction (From Scratch)
This repository contains an end-to-end machine learning pipeline for materials property prediction.
The project focuses on:
1. **Classification**: Classify whether a compound behaves like a metal or non-metal
2. **Regression**: Predict whether a compound behaves like a metal or non-metal

The workflow includes feature engineering (descriptors), encoding, dimensionality reduction (PCA), and testing (unit + functional) using PyTest.
Full report and results: [ppp_report.pdf](ppp_report.pdf)

## Project Goals
 - Convert raw chemical compound information into usable machine learning descriptors
 - Build classification and regression models using pure Python implementations
 - Evaluate performance using standard ML metrics
 - Ensure reliability using unit tests and functional/integration tests
 - Support both model development + validation workflows

## Implemented Tasks
 - Descriptor generation
 - Encodings
 - Principal Component Analysis (PCA)
 - Classification and regression models (NumPy-based)
 - ANN optimization algorithms
 - Unit & functional testing (PyTest)
 - Hyperparameter tuning
 - Gradient check for ANN backpropagation

## Repository Structure (with explanations)

```text
├── main.py
│   └── Runs the full pipeline end-to-end (classification + regression)
│
├── input_file_regression.csv
│   └── Main dataset used for regression (Debye temperature prediction)
│
├── non_zero_columns.csv
│   └── Stores filtered feature list used after preprocessing/PCA
│
├── report.pdf
│   └── Full documentation: theory, methodology, models, results, scaling tests
│
├── data_preprocessing/
│   ├── raw_compounds_data.csv
│   ├── Elements_data.csv / Elements_data_updated.csv
│   ├── compounds_to_element_count.py
│   ├── label_encoding.py
│   ├── one_hot_encode_ele_composition.py
│   ├── Desc1_comp_to_ind_ele.py
│   ├── Desc2_weighted_atomic_properties.py
│   └── ...
│   └── Converts raw compound strings into ML-ready numeric features
│
├── pca/
│   ├── pca.py
│   ├── compare_biplot_pca.py
│   └── prinical_component_vector_alignment_check.py
│   └── PCA implementation + verification and visualization
│
├── Multiple_linear_regression/
│   └── multiple_linear_regression.py
│   └── Regression baseline model for predicting Debye temperature
│
├── knn/
│   └── knn_classification.py
│   └── KNN classifier implementation + tuning utilities
│
├── Decision_tree_based_models/
│   ├── classification_and_regression_tree.py
│   ├── random_forest_classification.py
│   ├── gradient_boosting_regression.py
│   └── Tree-based ML models:
│       - Random Forest (classification)
│       - Gradient Boosting (regression)
│
├── ANN/
│   ├── ANN_classification.py
│   ├── ANN_regression.py
│   ├── optimization_algorithms.py
│   ├── hyperparameter_tuning.py
│   └── Neural network models (classification + regression)
│
└── Testing/
    ├── Data_preprocessing_testing/
    ├── ANN_testing/
    ├── Evaluation_metrics_testing/
    ├── Functional_testing/
    ├── pca_testing/
    └── ...
    └── Complete unit + functional test suite using pytest
```
## Requirements
This project uses Python and standard scientific libraries.
 - Python 3.10.7     - numpy 1.24.3
 - matplotlib 3.5.2  - pytest 7.1.2
## How to Run the Full Project
From the project root:
```bash
python main.py
```
This will execute the full workflow:
 - preprocessing → descriptors → encoding → PCA → training → evaluation
## Testing (Unit + Functional + Integrated Testing)




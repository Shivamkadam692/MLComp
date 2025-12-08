# Project Structure

This document describes the organization of the ML Analysis Dashboard project.

## Directory Structure

```
MLComp/
├── app.py                 # Main Flask application
├── main.py               # Command-line interface
├── master.py             # Linear regression example
├── requirements.txt      # Python dependencies
├── README.md             # Project overview and usage guide
├── cars.csv              # Sample dataset
├── sample_regression_dataset.csv  # Sample regression dataset
├── test_regression.csv   # Test dataset
├── test_regression_large.csv     # Large test dataset
├── test_uploaded_dataset.csv     # Test uploaded dataset
├── Dataset/              # Predefined datasets
│   ├── iris.csv
│   ├── titanic.csv
│   └── mtcars.csv
├── data/                 # Additional datasets
│   ├── iris.csv
│   └── titanic.csv
├── docs/                 # Documentation files
│   └── project_structure.md
├── models/               # Machine learning model implementations
│   ├── classification.py
│   ├── regression.py
│   ├── clustering.py
│   └── sentiment.py
├── src/                  # Legacy source code (will be deprecated in future versions)
│   ├── preprocess.py
│   ├── supervised.py
│   └── unsupervised.py
├── templates/            # HTML templates
│   ├── index.html
│   └── results.html
├── uploads/              # Uploaded datasets (auto-created)
│   └── [uploaded files]
└── utils/                # Utility functions
    ├── data_utils.py
    └── model_utils.py
```

## Component Descriptions

### Core Application Files
- `app.py`: Main Flask application with routing and API endpoints
- `main.py`: Command-line interface for running analyses
- `master.py`: Example implementation of linear regression

### Data Directories
- `Dataset/`: Original predefined datasets
- `data/`: Additional datasets used in the application
- `uploads/`: Directory for user-uploaded datasets

### Code Modules

#### Models (`models/`)
Contains implementations of specific machine learning algorithms:
- `classification.py`: Decision Tree, Logistic Regression, and SVM classifiers
- `regression.py`: Decision Tree Regressor, Linear Regression, and SVR
- `clustering.py`: K-Means clustering implementation
- `sentiment.py`: Sentiment analysis model

#### Utilities (`utils/`)
Helper functions for data processing and model management:
- `data_utils.py`: Functions for loading and preprocessing datasets
- `model_utils.py`: Functions for problem type detection and data preparation

#### Source (`src/`)
Legacy source code organization (will be deprecated in future versions):
- `preprocess.py`: Data preprocessing functions
- `supervised.py`: Supervised learning model training
- `unsupervised.py`: Unsupervised learning model training

### Templates
- `index.html`: Main dashboard page
- `results.html`: Detailed results view

### Documentation
- `README.md`: Main project documentation
- `docs/project_structure.md`: This file
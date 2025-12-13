# ML Analysis Dashboard v1.6.0

A comprehensive machine learning analysis dashboard built with Python Flask that provides automated model training, evaluation, and comparison across multiple datasets.

## Features

- **Multi-Dataset Support**: Analyze various datasets including structured (Iris, Titanic) and unstructured (IMDb reviews) data
- **Multiple ML Algorithms**: Decision Trees, Logistic Regression, SVM, K-Means Clustering, and Sentiment Analysis
- **Automated Model Training**: One-click training of all applicable models on selected datasets
- **Performance Metrics**: Comprehensive evaluation with accuracy, precision, recall, F1-score, MSE, RMSE, and R²
- **Model Comparison**: Cross-dataset performance visualization for informed model selection
- **Interactive Dashboard**: Real-time results display with Bootstrap UI
- **File Upload Support**: Analyze your own CSV datasets with automatic preprocessing
- **API Endpoints**: RESTful API for programmatic access to analysis results

## Quick Start

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the application:
   ```bash
   python app.py
   ```

3. Access the dashboard at `http://localhost:5000`

## Project Structure

For detailed information about the project structure, see [Project Structure Documentation](docs/project_structure.md).

```
MLComp/
├── app.py                 # Main Flask application
├── main.py               # Command-line interface
├── master.py             # Linear regression example
├── requirements.txt      # Python dependencies
├── README.md             # This file
├── cars.csv              # Sample dataset
├── sample_regression_dataset.csv  # Sample regression dataset
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
├── src/                  # Legacy source code (will be deprecated)
│   ├── preprocess.py
│   ├── supervised.py
│   └── unsupervised.py
├── templates/            # HTML templates
│   ├── index.html
│   └── results.html
├── uploads/              # Uploaded datasets (auto-created)
└── utils/                # Utility functions
    ├── data_utils.py
    └── model_utils.py
```

## Model Comparison

### Available Models

#### For Classification Tasks
- **Decision Tree Classifier**: Creates a tree-like model of decisions based on feature values
  - **Strengths**: Easy to interpret, handles both numerical and categorical data, requires little data preprocessing
  - **Best For**: Problems where interpretability is important and datasets have mixed data types

- **Logistic Regression**: Predicts the probability of a binary outcome using a logistic function
  - **Strengths**: Provides probability scores, less prone to overfitting, computationally efficient
  - **Best For**: Linearly separable problems and when probability estimates are needed

- **Support Vector Machine (SVM)**: Finds the optimal hyperplane that separates different classes with maximum margin
  - **Strengths**: Effective in high-dimensional spaces, memory efficient, versatile with different kernel functions
  - **Best For**: Complex classification problems with clear margins of separation

#### For Regression Tasks
- **Decision Tree Regressor**: Creates a tree-like model to predict continuous values
  - **Strengths**: Easy to interpret, handles non-linear relationships, requires little data preprocessing
  - **Best For**: Problems where interpretability is important and relationships may be non-linear

#### K-Means Clustering (Unsupervised)
- **Purpose**: Groups data points into clusters based on similarity
- **Strengths**: Simple to implement, scales to large datasets, adapts to new examples
- **Best For**: Exploratory data analysis and customer segmentation tasks

### Datasets Used

#### Iris Dataset
- **Purpose**: Classic classification problem predicting flower species based on measurements
- **Features**: Sepal length, sepal width, petal length, petal width
- **Classes**: Setosa, Versicolor, Virginica

#### Titanic Dataset
- **Purpose**: Predicts passenger survival based on demographic and ticket information
- **Features**: Passenger class, sex, age, family size, fare, embarkation port
- **Classes**: Survived (1) or Did Not Survive (0)

#### IMDb Reviews Dataset
- **Purpose**: Sentiment analysis of movie reviews
- **Features**: Review text
- **Classes**: Positive or Negative sentiment

## API Endpoints

- `GET /` - Main dashboard page
- `GET /api/results` - Get all analysis results in JSON format
- `POST /api/run-analysis` - Trigger a new analysis run
- `GET /api/compare-models` - Get model comparison data
- `POST /api/upload-dataset` - Upload and analyze a new CSV dataset

## Version History

### v1.6.0 (Current)
- Refactored code into modular structure with separate model files
- Added sentiment analysis module for unstructured text data
- Improved documentation and code organization
- Fixed pandas deprecation warnings
- Removed unused imports and cleaned up codebase

### v1.5.0
- Added file upload functionality for custom dataset analysis
- Implemented automatic data preprocessing for uploaded datasets
- Enhanced error handling and validation
- Improved UI with responsive design

### v1.0.0
- Initial release with basic ML model training and evaluation
- Support for Iris and Titanic datasets
- Classification and clustering algorithms
- Simple web dashboard interface

## Requirements

See [requirements.txt](requirements.txt) for a complete list of dependencies.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

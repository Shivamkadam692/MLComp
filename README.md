# Machine Learning Analysis Dashboard

A web-based application for comparative machine learning analysis with file upload capability and model comparison features.

## Table of Contents
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [API Endpoints](#api-endpoints)
- [File Upload](#file-upload)
- [Model Comparison](#model-comparison)
- [Enhancements](#enhancements)
- [License](#license)

## Features

- **Predefined Dataset Analysis**: Analyze popular datasets (Iris, Titanic, IMDb)
- **File Upload**: Upload your own CSV datasets for analysis (handles mixed data types)
- **Multiple ML Models**: Compare Decision Tree, Logistic Regression, SVM, and K-Means Clustering models
- **Comprehensive Model Comparison**: Cross-dataset performance comparison with detailed metrics
- **Educational Interface**: Built-in explanations of ML metrics for non-experts
- **Web Interface**: User-friendly dashboard with visualization
- **Real-time Results**: Instant analysis and comparison results
- **Automatic Problem Type Detection**: Automatically distinguishes between classification and regression tasks
- **Enhanced Frontend**: Modern UI with improved visualizations and user experience

## Technologies Used

### Core Technologies
- **Python** 3.13.2
- **Flask** 3.1.0
- **scikit-learn** 1.7.2
- **pandas** 2.2.3
- **numpy** 2.1.1
- **Jinja2** 3.1.5

### Frontend Technologies
- **Bootstrap** 5.3.0
- **JavaScript** (ES6)
- **HTML5**
- **CSS3**

### Dependencies
```
pandas==2.2.3
numpy==2.1.1
scikit-learn==1.7.2
matplotlib==3.10.0
seaborn==0.13.2
nltk==3.9.2
flask==3.1.0
```

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd MLComp
   ```

2. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   python app.py
   ```

4. **Access the dashboard**
   Open your browser and navigate to `http://127.0.0.1:5000`

## Usage

1. **Predefined Analysis**: The application automatically runs analysis on predefined datasets when started
2. **Upload Dataset**: Use the "Upload Your Dataset" section to analyze your own CSV files (now supports both classification and regression tasks)
3. **Model Comparison**: Click "Compare Models Across Datasets" to see detailed cross-dataset performance comparisons
4. **Detailed Views**: Click "View Details" on any dataset card for comprehensive results
5. **Learn Metrics**: Hover over ⓘ icons to understand ML metrics without leaving the page
6. **Switch Views**: Toggle between grid and list views for results
7. **Refresh Data**: Update results without reloading the page

## Project Structure

```
MLComp/
├── app.py                 # Main Flask application
├── main.py               # Command-line interface
├── master.py             # Linear regression example
├── requirements.txt      # Python dependencies
├── README.md             # This file
├── Dataset/              # Predefined datasets
│   ├── iris.csv
│   ├── titanic.csv
│   └── mtcars.csv
├── data/                 # Additional datasets
│   ├── iris.csv
│   ├── titanic.csv
│   └── imdb_reviews.csv
├── src/                  # Source code
│   ├── preprocess.py     # Data preprocessing functions
│   ├── supervised.py     # Supervised learning models
│   └── unsupervised.py   # Unsupervised learning models
├── templates/            # HTML templates
│   ├── index.html        # Main dashboard
│   └── results.html      # Detailed results view
└── uploads/              # Uploaded datasets (created automatically)
```

## API Endpoints

### GET Routes
- `GET /` - Main dashboard page
- `GET /results/<analysis_type>` - Detailed results for specific analysis
- `GET /api/results` - JSON response with all analysis results
- `GET /api/compare-models` - JSON response with model comparisons

### POST Routes
- `POST /api/run-analysis` - Trigger predefined dataset analysis
- `POST /api/upload-dataset` - Upload and analyze a CSV file

## File Upload

### Supported Formats
- CSV files only

### Requirements
- Must contain at least 2 columns
- Last column is treated as the target variable
- For classification: Target column must have at least 2 unique classes

### Enhanced Capabilities
- Automatically handles mixed data types (text and numeric)
- Processes missing values intelligently
- Encodes categorical variables for machine learning
- Supports both binary and multi-class classification targets
- **NEW**: Automatically detects and handles regression tasks with continuous target variables
- **NEW**: Applies appropriate algorithms based on target variable type

### Process
1. Select a CSV file using the upload form
2. System validates the file format and content
3. Dataset is automatically preprocessed to handle data type conversions
4. System detects if the problem is classification or regression using sklearn's `type_of_target`
5. Dataset is analyzed using appropriate supervised learning models
6. Results are displayed in the dashboard with relevant metrics

### Example CSV Format
```csv
feature1,feature2,feature3,target
1.0,2.0,3.0,classA
2.0,3.0,4.0,classB
3.0,4.0,5.0,classA
```

### Regression Example
```csv
feature1,feature2,price
1.0,2.0,150000
2.0,3.0,200000
3.0,4.0,250000
```

## ML Metric Explanations

To help non-machine learning experts understand the results, the interface includes built-in explanations for all key metrics:

### Classification Metrics
- **Accuracy**: Percentage of correct predictions out of all predictions
- **Precision**: Of all positive predictions, how many were actually positive
- **Recall**: Of all actual positives, how many were correctly predicted
- **F1-Score**: Harmonic mean of precision and recall, providing a balance between the two
- **Support**: Number of actual occurrences of the class in the dataset

### Regression Metrics
- **R² Score**: Proportion of variance in the target variable explained by the model (ranges from 0 to 1, higher is better)
- **RMSE**: Root Mean Square Error - standard deviation of prediction errors (lower is better)
- **MSE**: Mean Square Error - average squared differences between predicted and actual values (lower is better)

### Clustering Metrics
- **Silhouette Score**: Measures how similar an object is to its own cluster compared to other clusters (ranges from -1 to 1)

### How to Use
- Hover over any ⓘ icon next to a metric to see a brief explanation
- All detailed results pages include comprehensive explanations of metrics
- Summary cards provide context for key values
- Visual progress bars indicate performance levels

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

- **Linear Regression**: Models the relationship between features and a continuous target variable
  - **Strengths**: Simple to implement, computationally efficient, provides coefficient interpretation
  - **Best For**: Linear relationships between features and target variable

- **Support Vector Regression (SVR)**: Uses support vector machines for regression tasks
  - **Strengths**: Effective for non-linear relationships, robust to outliers
  - **Best For**: Complex regression problems with non-linear patterns

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
- **Purpose**: Classifies movie reviews as positive or negative sentiment
- **Features**: Text reviews (processed using TF-IDF)
- **Classes**: Positive or Negative sentiment

### Purpose of Comparisons

Comparing different models across multiple datasets serves several important purposes:

1. **Model Selection**: Identify which algorithms perform best for specific types of data
2. **Performance Benchmarking**: Establish baseline performance expectations for each algorithm
3. **Algorithm Suitability**: Understand which models work best for structured vs. unstructured data
4. **Robustness Testing**: Evaluate how models generalize across different domains
5. **Educational Value**: Demonstrate strengths and weaknesses of different approaches

### Comparison Metrics
- **Classification**: Accuracy, Precision, Recall, F1-Score
- **Regression**: R² Score, RMSE, MSE
- **Clustering**: Silhouette Score

### Visualization
- Tabular comparison across all datasets
- Performance metrics for each model
- Interactive dashboard for detailed analysis

## Enhancements

### Frontend Improvements
- **Modern UI Design**: Enhanced visual design with gradients, animations, and improved layouts
- **Statistics Overview**: Dashboard showing dataset count, model count, analysis count, and average accuracy
- **Responsive Design**: Optimized for different screen sizes and devices
- **Enhanced Visualizations**: Progress bars, improved charts, and better data representation
- **User Experience**: Better notifications, loading indicators, and interactive elements

### Backend Improvements
- **Automatic Problem Type Detection**: Uses sklearn's `type_of_target` to distinguish between classification and regression
- **Adaptive Model Selection**: Automatically applies appropriate algorithms based on problem type
- **Enhanced Error Handling**: Better error messages and graceful handling of edge cases
- **Improved Preprocessing**: More robust handling of mixed data types and missing values

### New Features
- **View Toggle**: Switch between grid and list views for results
- **Refresh Button**: Update data without reloading the page
- **Export Functionality**: Export comparison data and results
- **Enhanced Notifications**: Better user feedback with toast notifications

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Version Information

- **Application Version**: 1.4.0
- **Python**: 3.13.2
- **Flask**: 3.1.0
- **scikit-learn**: 1.7.2
- **pandas**: 2.2.3
- **numpy**: 2.1.1
- **Bootstrap**: 5.3.0
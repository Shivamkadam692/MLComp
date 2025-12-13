from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import sys
import os
import json
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import from new modular structure
from utils.data_utils import load_iris, load_titanic, load_imdb, preprocess_uploaded_dataset
from models.classification import train_decision_tree_classifier, train_logistic_regression, train_svm_classifier
from models.regression import train_decision_tree_regressor
from models.clustering import train_kmeans_clustering
from models.sentiment import train_sentiment_analysis
from utils.model_utils import detect_problem_type, prepare_data_for_training
import pandas as pd


class NumpyEncoder(json.JSONEncoder):
    """Special json encoder for numpy types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict()
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        return json.JSONEncoder.default(self, obj)


app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
app.secret_key = 'your_secret_key_here'  # Change this to a random secret key
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Max upload size: 16MB
app.config['UPLOAD_FOLDER'] = 'uploads'

# Global variable to store results
results_storage = {}
uploaded_datasets = {}


def run_analysis():
    """Run all machine learning analyses and store results"""
    global results_storage
    
    try:
        print("Running ML analysis...")
        
        # Structured Dataset 1: Iris
        print("[+] Loading Iris Dataset...")
        iris_df = load_iris()
        # Check what the actual target column is called
        print("Iris columns:", iris_df.columns.tolist())
        target_col = 'target'  # Based on our data/iris.csv file
        
        # Prepare data
        X_train, X_test, y_train, y_test = prepare_data_for_training(iris_df, target_col)
        
        # Train models
        iris_results = {}
        iris_results["Decision Tree"] = train_decision_tree_classifier(X_train, X_test, y_train, y_test)
        iris_results["Logistic Regression"] = train_logistic_regression(X_train, X_test, y_train, y_test)
        iris_results["SVM"] = train_svm_classifier(X_train, X_test, y_train, y_test)
        
        results_storage['iris'] = {
            'name': 'Iris Classification',
            'results': iris_results,
            'dataset_shape': iris_df.shape,
            'dataset_sample': iris_df.head().to_dict('records'),
            'columns': iris_df.columns.tolist()
        }

        # Structured Dataset 2: Titanic
        print("[+] Loading Titanic Dataset...")
        titanic_df = load_titanic()
        target_col = 'Survived'
        
        # Prepare data
        X_train, X_test, y_train, y_test = prepare_data_for_training(titanic_df, target_col)
        
        # Train models
        titanic_results = {}
        titanic_results["Decision Tree"] = train_decision_tree_classifier(X_train, X_test, y_train, y_test)
        titanic_results["Logistic Regression"] = train_logistic_regression(X_train, X_test, y_train, y_test)
        titanic_results["SVM"] = train_svm_classifier(X_train, X_test, y_train, y_test)
        
        results_storage['titanic'] = {
            'name': 'Titanic Survival Prediction',
            'results': titanic_results,
            'dataset_shape': titanic_df.shape,
            'dataset_sample': titanic_df.head().to_dict('records'),
            'columns': titanic_df.columns.tolist()
        }

        # Unstructured Dataset: IMDb Sentiment
        print("[+] Loading IMDb Dataset...")
        imdb_df = load_imdb()
        
        # Use the new sentiment analysis module
        imdb_results = train_sentiment_analysis(imdb_df)
        results_storage['imdb'] = {
            'name': 'IMDb Sentiment Analysis',
            'results': imdb_results,
            'dataset_shape': imdb_df.shape,
            'dataset_sample': imdb_df.head().to_dict('records'),
            'columns': imdb_df.columns.tolist()
        }

        # Unsupervised Clustering: K-Means on Iris
        print("[+] Running K-Means Clustering...")
        target_col = 'target'  # Based on our data/iris.csv file
        kmeans_results = train_kmeans_clustering(iris_df.drop(target_col, axis=1))
        results_storage['kmeans'] = {
            'name': 'K-Means Clustering (Iris)',
            'results': kmeans_results,
            'dataset_shape': iris_df.drop(target_col, axis=1).shape,
            'dataset_sample': iris_df.drop(target_col, axis=1).head().to_dict('records'),
            'columns': iris_df.drop(target_col, axis=1).columns.tolist()
        }
        
        print("Analysis complete!")
        return True
        
    except Exception as e:
        print(f"Error running analysis: {str(e)}")
        return False


@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html')


@app.route('/api/results')
def get_results():
    """API endpoint to get all results"""
    response = json.dumps(results_storage, cls=NumpyEncoder)
    return response, 200, {'Content-Type': 'application/json'}


@app.route('/api/run-analysis', methods=['POST'])
def run_analysis_endpoint():
    """API endpoint to trigger analysis"""
    success = run_analysis()
    response = json.dumps({'success': success}, cls=NumpyEncoder)
    return response, 200, {'Content-Type': 'application/json'}


@app.route('/api/compare-models', methods=['GET'])
def compare_models():
    """API endpoint to compare models across datasets"""
    try:
        comparisons = {}
        
        # Collect all datasets with supervised learning results
        supervised_datasets = {}
        for key, data in results_storage.items():
            if 'results' in data and isinstance(data['results'], dict) and not key.startswith('kmeans'):
                # Check if it's supervised learning results
                first_result = next(iter(data['results'].values()), None)
                # Check for any of the common supervised learning metrics
                has_classification_metrics = 'accuracy' in first_result
                has_regression_metrics = 'mse' in first_result or 'rmse' in first_result or 'r2' in first_result
                
                if has_classification_metrics or has_regression_metrics:
                    supervised_datasets[key] = data
        
        # Compare metrics across models
        model_comparison = {}
        for dataset_key, dataset_data in supervised_datasets.items():
            dataset_name = dataset_data.get('name', dataset_key)
            results = dataset_data.get('results', {})
            
            for model_name, model_results in results.items():
                if model_name not in model_comparison:
                    model_comparison[model_name] = {}
                
                # Handle both classification and regression results
                if isinstance(model_results, dict):
                    if 'accuracy' in model_results:
                        # Classification accuracy
                        accuracy = model_results.get('accuracy', 0)
                        model_comparison[model_name][dataset_name] = accuracy * 100  # Convert to percentage
                    elif 'r2' in model_results:
                        # Regression R² score
                        r2 = model_results.get('r2', 0)
                        model_comparison[model_name][dataset_name] = r2 * 100  # Convert to percentage for consistency
                    elif 'mse' in model_results:
                        # For MSE, we'll convert to a pseudo-accuracy (lower is better, so invert)
                        # This is a simplified approach - in practice, you might want to normalize this differently
                        mse = model_results.get('mse', 0)
                        # Convert MSE to a pseudo-accuracy (100 - normalized MSE)
                        # This is just for display purposes in the comparison table
                        pseudo_accuracy = max(0, min(100, 100 - (mse * 10)))  # Arbitrary scaling
                        model_comparison[model_name][dataset_name] = pseudo_accuracy
                    else:
                        # Default to 0 if no recognizable metric
                        model_comparison[model_name][dataset_name] = 0
                else:
                    model_comparison[model_name][dataset_name] = 0
        
        comparisons['model_accuracy'] = model_comparison
        
        # Prepare response
        response_data = {
            'success': True,
            'comparisons': comparisons,
            'datasets': [data.get('name', key) for key, data in supervised_datasets.items()]
        }
        
        return json.dumps(response_data, cls=NumpyEncoder), 200, {'Content-Type': 'application/json'}
    except Exception as e:
        print(f"Error comparing models: {str(e)}")
        import traceback
        traceback.print_exc()
        return json.dumps({'error': f'Error comparing models: {str(e)}'}, cls=NumpyEncoder), 500, {'Content-Type': 'application/json'}


@app.route('/api/upload-dataset', methods=['POST'])
def upload_dataset():
    """API endpoint to upload and process a dataset"""
    try:
        print("Received file upload request")
        print(f"Request headers: {dict(request.headers)}")
        print(f"Request files keys: {list(request.files.keys())}")
        
        if 'file' not in request.files:
            print("No file provided in request")
            return json.dumps({'error': 'No file provided'}, cls=NumpyEncoder), 400, {'Content-Type': 'application/json'}
        
        file = request.files['file']
        print(f"Received file: {file.filename}")
        print(f"File content type: {file.content_type}")
        
        if file.filename == '':
            print("Empty filename")
            return json.dumps({'error': 'No file selected'}, cls=NumpyEncoder), 400, {'Content-Type': 'application/json'}
        
        if file and file.filename.endswith('.csv'):
            try:
                # Save the file temporarily
                filename = file.filename
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                print(f"Saving file to: {file_path}")
                os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
                file.save(file_path)
                
                # Process the CSV file
                print("Reading CSV file")
                df = pd.read_csv(file_path)
                print(f"CSV file read successfully. Shape: {df.shape}")
                
                # Run analysis on the uploaded dataset
                # For simplicity, we'll assume it's a classification task
                # In a real application, you'd want to ask the user for the target column
                target_column = df.columns[-1]  # Assume last column is target
                print(f"Target column: {target_column}")
                            
                # Preprocess the data to handle non-numeric values
                print("Preprocessing data")
                df_processed, label_encoders = preprocess_uploaded_dataset(df, target_column)
                df = df_processed
                print(f"Data preprocessed. Shape: {df.shape}")
            except Exception as save_error:
                print(f"Error saving or reading file: {str(save_error)}")
                import traceback
                traceback.print_exc()
                return json.dumps({'error': f'Error saving or reading file: {str(save_error)}'}, cls=NumpyEncoder), 500, {'Content-Type': 'application/json'}
            
            # Validate the dataset
            if df.empty:
                print("Dataset is empty")
                return json.dumps({'error': 'The uploaded CSV file is empty'}, cls=NumpyEncoder), 400, {'Content-Type': 'application/json'}
            
            if len(df.columns) < 2:
                print("Dataset has less than 2 columns")
                return json.dumps({'error': 'The uploaded CSV file must have at least 2 columns'}, cls=NumpyEncoder), 400, {'Content-Type': 'application/json'}
            
            # Store the dataset
            dataset_name = os.path.splitext(filename)[0]
            uploaded_datasets[dataset_name] = df
            
            # Target column was already determined above
            
            # Validate target column
            if target_column not in df.columns:
                print(f"Target column {target_column} not found in dataset")
                return json.dumps({'error': f'Target column "{target_column}" not found in dataset'}, cls=NumpyEncoder), 400, {'Content-Type': 'application/json'}
            
            # Check if target column has enough classes for classification
            unique_classes = df[target_column].nunique()
            print(f"Unique classes in target column: {unique_classes}")
            if unique_classes < 2:
                print("Not enough unique classes for classification")
                return json.dumps({'error': 'Target column must have at least 2 unique classes for classification'}, cls=NumpyEncoder), 400, {'Content-Type': 'application/json'}
            
            # Detect problem type
            problem_type = detect_problem_type(df[target_column])
            
            # Prepare data for training
            X_train, X_test, y_train, y_test = prepare_data_for_training(df, target_column)
            
            # Run appropriate models based on problem type
            print("Training supervised models")
            try:
                if problem_type == 'classification':
                    results = {}
                    results["Decision Tree"] = train_decision_tree_classifier(X_train, X_test, y_train, y_test)
                    results["Logistic Regression"] = train_logistic_regression(X_train, X_test, y_train, y_test)
                    results["SVM"] = train_svm_classifier(X_train, X_test, y_train, y_test)
                else:  # regression
                    results = {}
                    results["Decision Tree"] = train_decision_tree_regressor(X_train, X_test, y_train, y_test)
                print("Models trained successfully")
            except Exception as model_error:
                print(f"Error training models: {str(model_error)}")
                return json.dumps({'error': f'Error training models: {str(model_error)}'}, cls=NumpyEncoder), 500, {'Content-Type': 'application/json'}
            
            # Store results
            results_storage[f'upload_{dataset_name}'] = {
                'name': f'{dataset_name} Analysis',
                'results': results,
                'dataset_shape': df.shape,
                'dataset_name': dataset_name,
                'target_column': target_column,
                'dataset_sample': df.head().to_dict('records'),
                'columns': df.columns.tolist()
            }
            
            response_data = {
                'success': True,
                'dataset_name': dataset_name,
                'results': results,
                'message': f'Successfully analyzed {filename}. Found {df.shape[0]} rows and {df.shape[1]} columns.'
            }
            
            print("Upload processed successfully")
            return json.dumps(response_data, cls=NumpyEncoder), 200, {'Content-Type': 'application/json'}
        else:
            print("Invalid file format")
            return json.dumps({'error': 'Invalid file format. Please upload a CSV file.'}, cls=NumpyEncoder), 400, {'Content-Type': 'application/json'}
    except Exception as e:
        print(f"Error uploading dataset: {str(e)}")
        import traceback
        traceback.print_exc()
        return json.dumps({'error': f'Internal server error: {str(e)}'}, cls=NumpyEncoder), 500, {'Content-Type': 'application/json'}


@app.route('/api/delete-dataset/<dataset_name>', methods=['DELETE'])
def delete_dataset(dataset_name):
    """API endpoint to delete an uploaded dataset"""
    try:
        # Check if dataset exists
        if dataset_name not in uploaded_datasets:
            return json.dumps({'error': 'Dataset not found'}, cls=NumpyEncoder), 404, {'Content-Type': 'application/json'}
        
        # Remove dataset from storage
        uploaded_datasets.pop(dataset_name, None)
        
        # Remove dataset results from storage
        dataset_key = f'upload_{dataset_name}'
        results_storage.pop(dataset_key, None)
        
        # Delete the file from uploads directory
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{dataset_name}.csv")
        if os.path.exists(file_path):
            os.remove(file_path)
        
        return json.dumps({'success': True, 'message': f'Dataset {dataset_name} deleted successfully'}, cls=NumpyEncoder), 200, {'Content-Type': 'application/json'}
    except Exception as e:
        print(f"Error deleting dataset: {str(e)}")
        return json.dumps({'error': f'Error deleting dataset: {str(e)}'}, cls=NumpyEncoder), 500, {'Content-Type': 'application/json'}


@app.route('/results/<analysis_type>')
def show_results(analysis_type):
    """Show detailed results for a specific analysis"""
    if analysis_type not in results_storage:
        return "Analysis not found", 404
    return render_template('results.html', analysis_type=analysis_type, data=results_storage[analysis_type])


if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    if not os.path.exists('templates'):
        os.makedirs('templates')
    
    # Run initial analysis
    run_analysis()
    
    # Start Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)
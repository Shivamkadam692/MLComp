"""
Model Utilities Module

This module contains utility functions for model training and evaluation.
"""

from sklearn.model_selection import train_test_split
from sklearn.utils.multiclass import type_of_target
import numpy as np


def detect_problem_type(y):
    """
    Detect if the problem is classification or regression based on target variable.
    
    Args:
        y: Target variable
        
    Returns:
        str: Problem type ('classification' or 'regression')
    """
    # Determine if this is a classification or regression problem
    y_type = type_of_target(y)
    print(f"Target variable type: {y_type}")
    
    # Enhanced detection for regression problems
    # If there are many unique values relative to sample size, it's likely regression
    unique_ratio = len(np.unique(y)) / len(y)
    print(f"Unique values ratio: {unique_ratio:.4f}")
    
    # Decide on problem type
    # Use regression if type is continuous or if there are too many classes for classification
    if y_type in ['continuous', 'continuous-multioutput'] or unique_ratio > 0.5:
        print("Treating as REGRESSION problem")
        return 'regression'
    else:
        print("Treating as CLASSIFICATION problem")
        return 'classification'


def prepare_data_for_training(df, target_col, test_size=0.2, random_state=42):
    """
    Prepare data for model training by splitting into train/test sets.
    
    Args:
        df: DataFrame containing features and target
        target_col: Name of the target column
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset. Available columns: {list(df.columns)}")
        
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
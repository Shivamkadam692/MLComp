"""
Data Utilities Module

This module contains utility functions for data loading and preprocessing.
"""

import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
import numpy as np


def load_iris():
    """Load the Iris dataset."""
    try:
        # Try multiple possible locations for the dataset
        possible_paths = [
            os.path.join('data', 'iris.csv'),
            os.path.join('Dataset', 'iris.csv'),
            'iris.csv'
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                df = pd.read_csv(path)
                return df
                
        raise FileNotFoundError("Could not find iris.csv in any expected location")
    except Exception as e:
        print(f"Error loading Iris dataset: {str(e)}")
        raise


def load_titanic():
    """Load and preprocess the Titanic dataset."""
    try:
        # Try multiple possible locations for the dataset
        possible_paths = [
            os.path.join('data', 'titanic.csv'),
            os.path.join('Dataset', 'titanic.csv'),
            'titanic.csv'
        ]
        
        df = None
        for path in possible_paths:
            if os.path.exists(path):
                df = pd.read_csv(path)
                break
                
        if df is None:
            raise FileNotFoundError("Could not find titanic.csv in any expected location")
        
        # Preprocess the Titanic dataset
        # Select relevant features and target
        relevant_columns = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Survived']
        df = df[relevant_columns]
        
        # Handle missing values - Fixed pandas warnings
        age_median = df['Age'].median()
        embarked_mode = df['Embarked'].mode()[0] if not df['Embarked'].mode().empty else 'S'
        
        df = df.fillna({'Age': age_median, 'Embarked': embarked_mode})
        
        # Encode categorical variables
        df['Sex'] = LabelEncoder().fit_transform(df['Sex'])
        df['Embarked'] = LabelEncoder().fit_transform(df['Embarked'])
        
        return df
    except Exception as e:
        print(f"Error loading Titanic dataset: {str(e)}")
        raise


def load_imdb():
    """Load the IMDb dataset."""
    try:
        # Try multiple possible locations for the dataset
        possible_paths = [
            os.path.join('data', 'imdb_reviews.csv'),
            os.path.join('Dataset', 'imdb_reviews.csv'),
            'imdb_reviews.csv'
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                df = pd.read_csv(path)
                return df
                
        raise FileNotFoundError("Could not find imdb_reviews.csv in any expected location")
    except Exception as e:
        print(f"Error loading IMDb dataset: {str(e)}")
        raise


def preprocess_uploaded_dataset(df, target_column):
    """
    Preprocess an uploaded dataset to make it suitable for machine learning.
    Handles missing values and converts categorical variables to numeric.
    
    Args:
        df: DataFrame to preprocess
        target_column: Name of the target column
        
    Returns:
        tuple: (processed DataFrame, label encoders dictionary)
    """
    try:
        # Make a copy to avoid modifying the original dataframe
        df_processed = df.copy()
        
        # Handle missing values
        for column in df_processed.columns:
            if df_processed[column].dtype == 'object':
                # For categorical columns, fill with mode or 'Unknown'
                if df_processed[column].isnull().any():
                    mode_value = df_processed[column].mode()
                    if not mode_value.empty:
                        df_processed[column] = df_processed[column].fillna(mode_value[0])
                    else:
                        df_processed[column] = df_processed[column].fillna('Unknown')
            else:
                # For numerical columns, fill with median
                if df_processed[column].isnull().any():
                    df_processed[column] = df_processed[column].fillna(df_processed[column].median())
        
        # Handle categorical variables
        label_encoders = {}
        for column in df_processed.columns:
            if df_processed[column].dtype == 'object':
                # Skip the target column if it's the only non-numeric column
                # We'll handle it separately
                le = LabelEncoder()
                df_processed[column] = le.fit_transform(df_processed[column])
                label_encoders[column] = le
        
        return df_processed, label_encoders
    except Exception as e:
        print(f"Error preprocessing dataset: {str(e)}")
        raise
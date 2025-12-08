"""
Regression Models Module

This module contains implementations of various regression algorithms
used in the ML Analysis Dashboard.
"""

from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np


def train_decision_tree_regressor(X_train, X_test, y_train, y_test):
    """
    Train a Decision Tree regressor and return results.
    
    Args:
        X_train: Training features
        X_test: Testing features
        y_train: Training targets
        y_test: Testing targets
        
    Returns:
        dict: Regression metrics (MSE, RMSE, R2)
    """
    model = DecisionTreeRegressor(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Calculate regression metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return {
        'mse': mse,
        'rmse': np.sqrt(mse),
        'r2': r2
    }


def train_linear_regression(X_train, X_test, y_train, y_test):
    """
    Train a Linear Regression model and return results.
    
    Args:
        X_train: Training features
        X_test: Testing features
        y_train: Training targets
        y_test: Testing targets
        
    Returns:
        dict: Regression metrics (MSE, RMSE, R2)
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Calculate regression metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return {
        'mse': mse,
        'rmse': np.sqrt(mse),
        'r2': r2
    }


def train_svr(X_train, X_test, y_train, y_test):
    """
    Train an SVR model and return results.
    
    Args:
        X_train: Training features
        X_test: Testing features
        y_train: Training targets
        y_test: Testing targets
        
    Returns:
        dict: Regression metrics (MSE, RMSE, R2)
    """
    model = SVR()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Calculate regression metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return {
        'mse': mse,
        'rmse': np.sqrt(mse),
        'r2': r2
    }
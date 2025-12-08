"""
Classification Models Module

This module contains implementations of various classification algorithms
used in the ML Analysis Dashboard.
"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import numpy as np


def train_decision_tree_classifier(X_train, X_test, y_train, y_test):
    """
    Train a Decision Tree classifier and return results.
    
    Args:
        X_train: Training features
        X_test: Testing features
        y_train: Training labels
        y_test: Testing labels
        
    Returns:
        dict: Classification report metrics
    """
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    return report


def train_logistic_regression(X_train, X_test, y_train, y_test):
    """
    Train a Logistic Regression model and return results.
    
    Args:
        X_train: Training features
        X_test: Testing features
        y_train: Training labels
        y_test: Testing labels
        
    Returns:
        dict: Classification report metrics
    """
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    return report


def train_svm_classifier(X_train, X_test, y_train, y_test):
    """
    Train an SVM classifier and return results.
    
    Args:
        X_train: Training features
        X_test: Testing features
        y_train: Training labels
        y_test: Testing labels
        
    Returns:
        dict: Classification report metrics
    """
    model = SVC(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    return report
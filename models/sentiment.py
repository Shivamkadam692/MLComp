"""
Sentiment Analysis Module

This module contains implementations of sentiment analysis models
used in the ML Analysis Dashboard.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def train_sentiment_analysis(df):
    """
    Train a sentiment analysis model on the provided dataset.
    
    Args:
        df: DataFrame with 'review' and 'sentiment' columns
        
    Returns:
        dict: Classification report metrics
    """
    try:
        if 'review' not in df.columns or 'sentiment' not in df.columns:
            raise ValueError("Required columns 'review' and/or 'sentiment' not found in dataset")
            
        tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
        X = tfidf.fit_transform(df['review'])
        y = (df['sentiment'] == 'positive').astype(int)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LogisticRegression(random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print("\n[Sentiment Analysis]")
        report = classification_report(y_test, y_pred, output_dict=True)
        print(classification_report(y_test, y_pred))
        
        return report
    except Exception as e:
        print(f"Error in train_sentiment_analysis: {str(e)}")
        raise
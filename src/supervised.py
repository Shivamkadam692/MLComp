from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils.multiclass import type_of_target
import numpy as np

def train_supervised_models(df, target_col):
    try:
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataset. Available columns: {list(df.columns)}")
            
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Determine if this is a classification or regression problem
        y_type = type_of_target(y)
        print(f"Target variable type: {y_type}")
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        if y_type in ['continuous', 'continuous-multioutput']:
            # Regression problem
            models = {
                "Decision Tree": DecisionTreeRegressor(random_state=42),
                "Linear Regression": LinearRegression(),
                "SVR": SVR()
            }
            
            results = {}
            for name, model in models.items():
                print(f"\n{name} Results:")
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                # Calculate regression metrics
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                results[name] = {
                    'mse': mse,
                    'rmse': np.sqrt(mse),
                    'r2': r2
                }
                print(f"MSE: {mse:.4f}, RMSE: {np.sqrt(mse):.4f}, R²: {r2:.4f}")
        else:
            # Classification problem
            models = {
                "Decision Tree": DecisionTreeClassifier(random_state=42),
                "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
                "SVM": SVC(random_state=42)
            }
            
            results = {}
            for name, model in models.items():
                print(f"\n{name} Results:")
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                report = classification_report(y_test, y_pred, output_dict=True)
                results[name] = report
                print(classification_report(y_test, y_pred))
        
        return results
    except Exception as e:
        print(f"Error in train_supervised_models: {str(e)}")
        raise

def train_sentiment_analysis(df):
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
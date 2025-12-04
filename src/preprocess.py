import pandas as pd
from sklearn.preprocessing import LabelEncoder
import nltk
nltk.download('punkt')

def load_iris():
    # Use the iris.csv from data directory as it has the correct format
    import os
    try:
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_path = os.path.join(script_dir, 'data', 'iris.csv')
        df = pd.read_csv(data_path)
        return df
    except FileNotFoundError:
        raise FileNotFoundError("iris.csv not found in data directory")
    except Exception as e:
        raise Exception(f"Error loading iris dataset: {str(e)}")

def load_titanic():
    import os
    try:
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_path = os.path.join(script_dir, 'data', 'titanic.csv')
        df = pd.read_csv(data_path)
        df.drop(columns=['Name', 'Ticket', 'Cabin'], inplace=True)
        df['Sex'] = LabelEncoder().fit_transform(df['Sex'])
        df['Embarked'] = df['Embarked'].fillna('S')
        df['Embarked'] = LabelEncoder().fit_transform(df['Embarked'])
        df['Age'] = df['Age'].fillna(df['Age'].mean())
        df['Fare'] = df['Fare'].fillna(df['Fare'].mean())

        return df
    except FileNotFoundError:
        raise FileNotFoundError("titanic.csv not found in data directory")
    except Exception as e:
        raise Exception(f"Error loading titanic dataset: {str(e)}")

def load_imdb():
    import os
    try:
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_path = os.path.join(script_dir, 'data', 'imdb_reviews.csv')
        df = pd.read_csv(data_path)
        df = df.sample(min(5000, len(df)))  # handles small dataset
        return df
    except FileNotFoundError:
        raise FileNotFoundError("imdb_reviews.csv not found in data directory")
    except Exception as e:
        raise Exception(f"Error loading IMDB dataset: {str(e)}")

def preprocess_uploaded_dataset(df, target_column):
    """
    Preprocess an uploaded dataset to make it suitable for machine learning.
    Handles missing values and converts categorical variables to numeric.
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
                if column != target_column or df_processed[target_column].dtype != 'object':
                    le = LabelEncoder()
                    df_processed[column] = le.fit_transform(df_processed[column].astype(str))
                    label_encoders[column] = le
        
        # Handle target column if it's categorical
        if target_column in df_processed.columns and df_processed[target_column].dtype == 'object':
            # Check if it's already binary
            unique_values = df_processed[target_column].unique()
            if len(unique_values) == 2:
                # Binary classification - map to 0 and 1
                le_target = LabelEncoder()
                df_processed[target_column] = le_target.fit_transform(df_processed[target_column])
                label_encoders[target_column] = le_target
            else:
                # Multi-class - keep as is for now, but ensure it's encoded properly
                le_target = LabelEncoder()
                df_processed[target_column] = le_target.fit_transform(df_processed[target_column])
                label_encoders[target_column] = le_target
        
        return df_processed, label_encoders
    
    except Exception as e:
        raise Exception(f"Error preprocessing uploaded dataset: {str(e)}")

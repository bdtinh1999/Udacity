import sys
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
import nltk
from nltk.tokenize import word_tokenize
from sqlalchemy import create_engine

# Download NLTK's punkt tokenizer data
nltk.download('punkt')


def load_data(database_filepath):
    """
    Loads data from the database and returns the feature and target variables.
    Assumes the database contains a table 'disaster_messages' with disaster messages and labels.
    """
    try:
        # Connect to the database using SQLAlchemy's create_engine
        engine = create_engine(f'sqlite:///{database_filepath}')
        # Read data from the 'disaster_messages' table
        df = pd.read_sql_table('disaster_messages', con=engine)

        # Extract features (X) and target labels (Y)
        X = df['message']
        Y = df.drop(columns=['message', 'id', 'original', 'genre'])  # Assuming 'id', 'original', 'genre' are non-relevant columns
        category_names = Y.columns.tolist()

        return X, Y, category_names

    except Exception as e:
        print(f"Error loading data from database: {e}")
        return None, None, None


def tokenize(text):
    """
    Tokenizes and cleans the input text. Uses word_tokenize from NLTK.
    """
    # Tokenize and clean the text
    tokens = word_tokenize(text.lower())
    # Remove non-alphabetic tokens
    tokens = [word for word in tokens if word.isalpha()]
    return tokens


def build_model():
    """
    Builds a machine learning model pipeline. A RandomForestClassifier
    with multi-output classifier for handling multiple labels.
    """
    # Create a pipeline with a TfidfVectorizer and RandomForestClassifier
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(tokenizer=tokenize)),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluates the model performance and prints out the classification report.
    """
    Y_pred = model.predict(X_test)
    print(classification_report(Y_test, Y_pred, target_names=category_names))


def save_model(model, model_filepath):
    """
    Saves the trained model as a pickle file.
    """
    try:
        with open(model_filepath, 'wb') as f:
            pickle.dump(model, f)
        print(f"Model saved to {model_filepath}")
    except Exception as e:
        print(f"Error saving model: {e}")


def load_model(model_filepath):
    """
    Loads a model from a pickle file.
    """
    try:
        with open(model_filepath, 'rb') as f:
            model = pickle.load(f)
        print(f"Model loaded from {model_filepath}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print(f'Loading data...\n    DATABASE: {database_filepath}')
        X, Y, category_names = load_data(database_filepath)

        if X is None or Y is None:
            print("Error loading data. Exiting.")
            return

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        # Define grid search parameters for RandomForest
        param_grid = {
            'clf__estimator__n_estimators': [10, 50, 100],
            'clf__estimator__max_depth': [10, 50, None]
        }
        
        grid_search = GridSearchCV(model, param_grid, cv=3, verbose=1, n_jobs=-1)
        
        print('Training model with GridSearchCV...')
        grid_search.fit(X_train, Y_train)
        
        print('Best parameters from GridSearchCV:')
        print(grid_search.best_params_)

        print('Evaluating model...')
        evaluate_model(grid_search, X_test, Y_test, category_names)

        print(f'Saving model...\n    MODEL: {model_filepath}')
        save_model(grid_search, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()

import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Load messages and categories datasets and merge them.

    Args:
        messages_filepath (str): Filepath for the messages CSV file.
        categories_filepath (str): Filepath for the categories CSV file.

    Returns:
        pd.DataFrame: Merged dataset.
    """
    try:
        messages = pd.read_csv(messages_filepath)
        categories = pd.read_csv(categories_filepath)
        df = pd.merge(messages, categories, on='id')
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

def clean_data(df):
    """
    Clean the merged dataset by splitting categories into separate columns,
    converting values to numeric, handling invalid values, and removing duplicates.

    Args:
        df (pd.DataFrame): Merged dataset.

    Returns:
        pd.DataFrame: Cleaned dataset.
    """
    try:
        # Split `categories` into separate category columns
        categories = df['categories'].str.split(';', expand=True)

        # Use the first row to extract category names
        row = categories.iloc[0]
        category_colnames = row.apply(lambda x: x[:-2])
        categories.columns = category_colnames

        # Convert category values to numbers 0 or 1
        for column in categories:
            categories[column] = categories[column].str[-1].astype(int)

            # Ensure binary values (0 or 1)
            categories[column] = categories[column].apply(lambda x: 1 if x == 1 else 0)

        # Replace `categories` column in `df` with new category columns
        df.drop('categories', axis=1, inplace=True)
        df = pd.concat([df, categories], axis=1)

        # Remove duplicates
        df = df.drop_duplicates()

        # Sanity check: Ensure all values in the category columns are 0 or 1
        assert df.iloc[:, 4:].isin([0, 1]).all().all(), "Data contains invalid category values."

        return df
    except Exception as e:
        print(f"Error cleaning data: {e}")
        sys.exit(1)

def save_data(df, database_filepath):
    """
    Save the clean dataset into an SQLite database.

    Args:
        df (pd.DataFrame): Cleaned dataset.
        database_filepath (str): Filepath for the SQLite database.
    """
    try:
        engine = create_engine(f'sqlite:///{database_filepath}')
        df.to_sql('disaster_messages', engine, index=False, if_exists='replace')
        print(f"Data successfully saved to {database_filepath}")
    except Exception as e:
        print(f"Error saving data: {e}")
        sys.exit(1)

def main():
    """
    Main function to load, clean, and save data.

    Command line arguments:
        messages_filepath (str): Path to the messages CSV file.
        categories_filepath (str): Path to the categories CSV file.
        database_filepath (str): Path to the SQLite database file.
    """
    if len(sys.argv) == 4:
        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print(f'Loading data...\n    MESSAGES: {messages_filepath}\n    CATEGORIES: {categories_filepath}')
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print(f'Saving data...\n    DATABASE: {database_filepath}')
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'TinhDatabaseName.db')

if __name__ == '__main__':
    main()

import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    This function reads raw messages and categories into pandas dataframe
    and merge into one dataset
    
    Parameters:
    messages_filepath (str): path of messages.csv
    categories_filepath (str): path of categories.csv
    
    Returns:
    df: merged dataset
    """

    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on ='id')

    return df


def clean_data(df):
    """
    This function reads in merged dataset and performs
    a series of cleaning, including column renaming, 
    splitting columns, deduplication, etc.
    
    Parameters:
    df: uncleaned merged dataset
    
    Returns:
    df: dataset after cleaning
    """
    
    categories = df['categories'].str.split(';', expand=True)

    row = categories.iloc[0]
    category_colnames = [name[:-2] for name in row.tolist()]
    categories.columns = category_colnames

    for column in categories:
        categories[column] = categories[column].str[-1:]
        categories[column] = pd.to_numeric(categories[column])

    df.drop(columns='categories', inplace=True)
    df = pd.concat([df, categories], axis=1)
    
    df = df.drop_duplicates()
    
    return df


def save_data(df, database_filename):
    """
    This function saves the cleaned data into a SQLite database
    
    Parameters:
    database_filename (str): the custom name of the database
    
    """
        
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('disaster_data', engine, index=False, if_exists='replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
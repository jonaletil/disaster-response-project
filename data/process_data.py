import sys
import os
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Loads messages and categories dataframes and merge them

    Args:
        messages_filepath (str): The file location of the messages df
        categories_filepath (str): The file location of the categories df

    Returns:
        merged dataframe
    """
    messages = pd.read_csv(messages_filepath, dtype=str)
    categories = pd.read_csv(categories_filepath, dtype=str)
    df = pd.merge(messages, categories, how='inner', on='id')
    return df


def clean_data(df):
    """
    Cleans the data:
        - split categories into separate category columns
        - convert category values to just numbers 0 or 1
        - replace categories column in df with new category columns
        - remove duplicates

    Args:
        df: merged dataframe from load_data() function

    Returns:
        cleaned dataframe
    """

    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(';', expand=True)
    # select the first row of the categories dataframe
    row = categories.iloc[0]
    # extract a list of new column names for categories
    category_colnames = list(row.map(lambda x: x[:-2]))
    # rename the columns of `categories`
    categories.columns = category_colnames

    for column in categories.columns:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1]
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
        # convert any values different than 0 and 1 to 1
        categories[column].loc[(categories[column] != 0) & (categories[column] != 1)] = 1

    # drop the original categories column from `df`
    df.drop(['categories'], axis=1, inplace=True)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    # drop the child_alone column from `df` - all values are 0
    df.drop(['child_alone'], axis=1, inplace=True)
    # drop duplicates
    df.drop_duplicates(inplace=True)

    return df


def save_data(df, database_filename):
    """
    Save the clean dataset into a sqlite database

    Args:
        df: clean data from clean_data() function
        database_filename (str): file name of SQL database in which the clean dataset will be stored
    """
    engine = create_engine('sqlite:///' + database_filename)
    table_name = os.path.basename(database_filename).split('.')[0]
    df.to_sql(table_name, engine, index=False, if_exists='replace')


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
        print('Please provide the filepaths of the messages and categories ' \
              'datasets as the first and second argument respectively, as ' \
              'well as the filepath of the database to save the cleaned data ' \
              'to as the third argument. \n\nExample: python process_data.py ' \
              'disaster_messages.csv disaster_categories.csv ' \
              'DisasterResponse.db')


if __name__ == '__main__':
    main()

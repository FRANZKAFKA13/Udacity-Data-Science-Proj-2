import sys
import pandas as pd
from sqlalchemy import create_engine


def extract_data(messages_filepath, categories_filepath):
    """
    Extract step of the ETL pipeline.

    :param messages_filepath: The filepath of the message.csv file
    :param categories_filepath: The filepath of the categories.csv file
    :return: The two .csv files combined in a pandas dataframe
    """

    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    print('    ' + str(messages.shape[0]) + ' messages extracted from .csv.')
    print('    ' + str(categories.shape[0]) + ' categories extracted from .csv.')

    df = messages.merge(categories, how='left', on='id')
    return df


def transform_data(df):
    """
    Transform step of the ETL pipeline.

    :param df: The dataframe created from the .csv files
    :return: Returns a cleaned dataframe with categories transformed to dummy variables
    """
    category_df = df['categories'].str.split(pat=";", expand=True)

    categories_raw = category_df.loc[0]
    categories_clean = [x[:-2] for x in categories_raw]
    category_df.columns = categories_clean

    print('    ' + str(category_df.shape[1]) + ' categories transformed to dummy variables.')

    for category in category_df:
        category_df[category] = category_df[category].str[-1]
        category_df[category] = category_df[category].astype('int32')

    df = df.drop(labels=['categories'], axis=1)
    df = pd.concat([df, category_df], axis=1)

    duplicate_count = df.duplicated('id').sum()
    df = df.drop_duplicates('id')
    print('    ' + str(duplicate_count) + ' duplicates dropped.')

    faulty_message_count = (df['related'] == 2).sum()
    df = df[df['related'] != 2]
    print('    ' + str(faulty_message_count) + ' faulty messages dropped.')

    return df


def load_data(df, database_filename):
    """
    Extract step of the ETL pipeline, saving the cleaned dataset in a SQLite database.

    :param df: The cleaned dataframe
    :param database_filename: The target database for the load step of the ETL process
    :return:
    """
    engine = create_engine('sqlite:///' + str(database_filename))

    try:
        df.to_sql('messages_categorized', engine, index=False, if_exists='replace')
        print('Cleaned data saved to database!')
    except ValueError:
        print("Warning: Database error.")


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Extracting data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = extract_data(messages_filepath, categories_filepath)

        print('Transforming data...')
        df = transform_data(df)

        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        load_data(df, database_filepath)

    else:
        print('Please provide the filepaths of the messages and categories '
              'datasets as the first and second argument respectively, as '
              'well as the filepath of the database to save the cleaned data '
              'to as the third argument. \n\nExample run command:\npython process_data.py disaster_messages.csv '
              'disaster_categories.csv disaster_response.db')


if __name__ == '__main__':
    main()

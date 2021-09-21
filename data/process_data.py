import sys
import pandas as pd
from sqlalchemy.engine import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Load message and category files
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on = 'id')
    return df

def clean_data(df):
    '''Clean Dataframe
    Creates dataframe by splitting "categories" column string content
    and only keeping numeric part.
    merge it with original dataframe and remove duplicates
    
    Input:
    df (dataframe) : name of dataframe
    
    Output:
    df (dataframe) : name of dataframe
    '''
    categories = df['categories'].str.split(';',expand=True)
    # select the first row of the categories dataframe
    row = categories.iloc[0]

       # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    #category_colnames = 
     #print(category_colnames)
    category_colnames = row.apply(lambda x:x[:-2])
    # rename the columns of `categories`
    categories.columns = category_colnames
    for column in categories:
    # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x[-1])
        categories[column] = categories[column].astype(int)
    # convert column from string to numeric
    #categories[column] = categories[column]
    l=[]
    for col in categories.columns:
        l.append(categories[col].unique())
    for col in categories.columns:
        categories.loc[(categories[col]!=1)&(categories[col]!=0)]=1
    # drop the original categories column from `df`
    df.drop(columns='categories', inplace=True)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    # drop duplicates
    df.drop_duplicates(inplace=True)
    return df
def save_data(df, database_filename):
    '''
    Save data in SQL database
    
    Input:
    
    df (dataframe) : name of dataframe
    database_filename : filename of database
    Returns: None
    '''
    engine = create_engine('sqlite:///DisasterResponse.db')
    df.to_sql('data', engine, index=False, if_exists='replace')


def main():
    '''
    Main function to call all the functions
    '''
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

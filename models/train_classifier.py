import sys
import sqlite3
import pandas as pd
import re
import nltk
nltk.download(['punkt','stopwords','wordnet'])
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sqlalchemy.engine import create_engine
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
#from sklearn import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import pickle
def load_data(database_filepath):
    '''
    Load messages and categories from database
    Input:
    database_filepath(str): name of database containing data
    
    Return:
    X(array): Name of array for features
    Y(array): Name of array for multioutput labels
    
    '''
    conn=sqlite3.connect(database_filepath)
    df= pd.read_sql('select  * from data',conn)
    X = df['message'].values
    Y = df[df.columns[4:]]
    category_names = list(Y.columns)
    return X,Y,category_names

def tokenize(text):
    '''Process text into clean tokens
    Text is processed by keeping it in lower case,
    remove stopwords & words lemmatized into their original stem
    
    Input:
    text (str) : message in text form
    
    Output:
    clean_tokens (array): array of words after processing
    '''
    text=text.lower()
    text = re.sub(r"[^a-zA-Z0-9]"," ", text.lower())
    tokens=word_tokenize(text)
    stop=stopwords.words("english")
    words= [t for t in tokens if t not in stop]
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens=[]
    for tok in words:
        clean_tok=lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens
def build_model():
    '''Creates a ML pipeline with GridSearchCV
    
    Input:
    None
    
    Output:
    scikit learn pipeline model
    '''
    pipeline = Pipeline([
    ('vect',CountVectorizer(tokenizer=tokenize)),
    ('tfidf',TfidfTransformer()),
    ('clf',MultiOutputClassifier(RandomForestClassifier(n_jobs=-1)))
    ])
    parameters = parameters = {
    'clf__estimator__max_depth':[15,25],
    'clf__estimator__n_estimators':[100,200],
    }

    model = GridSearchCV(pipeline,param_grid=parameters)
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Shows performance metrics of the best model
    
    Input:
    model: name of the model
    X_test (dataframe) : test data features
    Y_test (dataframe): multioutput test data labels
    category_names:
    
    Returns:
    report: classification report
    '''
    Y_pred=model.predict(X_test)
    report=classification_report(Y_test,Y_pred,target_names=category_names)
    print(report)
    return report

def save_model(model, model_filepath):
    '''
    Saves model to pickle file
    
    Input:
    model: Name of model
    model_filepath: name of database containing data
    '''
    pickle.dump(model,open('Classifier.pkl','wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()

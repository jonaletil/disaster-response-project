import sys
import os
import re
import time
import pandas as pd
from sqlalchemy import create_engine

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download(['punkt', 'wordnet', 'stopwords'])

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report

import pickle


def load_data(database_filepath):
    """
    Loads the data from SQL database

    Args:
        database_filepath (str): The file location

    Returns:
        X: features dataframe
        y: target dataframe
    """

    engine = create_engine('sqlite:///' + database_filepath)
    table_name = os.path.basename(database_filepath).split('.')[0]
    df = pd.read_sql_table(table_name, con=engine)

    X = df['message']
    y = df.iloc[:, 4:]
    category_names = y.columns

    return X, y, category_names


def tokenize(text):
    """
    Process the raw texts:
        - if any urls replace with the string 'urlplaceholder'
        - remove punctuation
        - tokenize
        - remove stop words
        - normalize and lemmatize

    Args:
        text (str): raw text

    Return: a list of clean words
    """
    # check for urls
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    urls = re.findall(url_regex, text)

    gen_urls = (url for url in urls if len(urls))

    for url in gen_urls:
        text = text.replace(url, "urlplaceholder")

    # remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)

    # tokenize
    tokens = word_tokenize(text)

    # remove stop words
    tokens = [token for token in tokens if token not in stopwords.words("english")]

    # lemmatization
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for token in tokens:
        clean_token = lemmatizer.lemmatize(token).lower().strip()
        clean_tokens.append(clean_token)

    return clean_tokens


def build_model():
    """
    A pipeline that includes text processing steps and a classifier (random forest).

    Return: GridSearch output
    """

    model = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters = {
        'clf__estimator__n_estimators': [50, 100, 200],
        'clf__estimator__min_samples_split': [2, 3, 4],
    }

    cv = GridSearchCV(model, param_grid=parameters)

    return cv

def evaluate_model(model, X_test, y_test, category_names):
    """
    Evaluates the model performance for each category

    Args:
        model: the classification returned with optimized parameters
        X_test: feature variable from test set
        y_test: target variable from test set
        category_names: column names

    OUTPUT
        Classification report and accuracy score
    """
    # predict
    y_pred = model.predict(X_test)

    # classification report
    print(classification_report(y_test.values, y_pred, target_names=category_names))

    # accuracy score
    accuracy = (y_pred == y_test.values).mean()
    print('The model accuracy score is {:.3f}'.format(accuracy))


def save_model(model, model_filepath):
    """
    Saves the pipeline to local disk
    """

    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        start_time = time.time()
        model.fit(X_train, Y_train)
        print("\nThis took %s seconds." % (time.time() - start_time))

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print(
            'Please provide the filepath of the disaster messages database ''as the first argument and the filepath of the pickle file to ''save the model to as the second argument. \n\nExample: python ''train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()

import sys
import pandas as pd
from sqlalchemy import create_engine
import re

import nltk

nltk.download("wordnet")
from nltk.stem.wordnet import WordNetLemmatizer

nltk.download("punkt")
from nltk.tokenize import word_tokenize

nltk.download("stopwords")
from nltk.corpus import stopwords

from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from sklearn.model_selection import GridSearchCV

import pickle


def load_data(database_filepath):
    """
    Loading data from the database that was cleaned previously.
    :param database_filepath: The filepath to the database storing the clean table
    :return: Returns predictor variables X, target variables Y and the category names
    """

    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('messages_categorized', engine)

    categories = df.columns[4:]
    Y = df[categories]
    X = df['message']

    # Replace 2 with 1 in related category
    Y['related'][Y['related'] == 2] = 1

    return X, Y, categories


def tokenize(text):
    """
    Tokenizer used in the pipeline
    :param text: The respective message that was sent as a string to be manipulated by the tokenizer.
    :return: Tokenized messages.
    """
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()

    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    return tokens


def build_model():
    """
    Function building the ML pipeline, including CountVectorizer, TF-IDF transformer and a multi output classifier using random forest estimator.
    Using GridSearch for parameter optimization.
    :return: Returns the ML pipeline.
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters = {
        # 'vect__ngram_range': ((1,1),(1,2)),
        # 'vect__max_df': (0.5, 0.75, 1.0),
        # 'vect__max_features': (None, 5000, 10000),
        'clf__estimator__n_estimators': [50, 100, 200]
        # 'clf__estimator__min_samples_split': [2, 3, 4]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=10, cv=2)

    return cv


def evaluate_model(model, X_test, Y_test, categories):
    """
    The function evaluating how the model predicts the respective message categories.
    :param model: The model to be evaluated.
    :param X_test: Test data for predictor variables
    :param Y_test: The data for target variables
    :param categories: Category names for prediction
    """
    Y_pred = pd.DataFrame(model.predict(X_test))
    Y_pred.columns = categories

    print("Overall score: " + str(model.score(X_test, Y_test)))

    for category in categories:
        print("Report for category: " + category + '\n')
        print(classification_report(Y_test[category], Y_pred[category]))
        print('\n')
    pass


def save_model(model, model_filepath):
    """
    Saves the model in a persistent format as .pkl file
    :param model: Model that was fit
    :param model_filepath: Path to folder where model is to be saveds
    :return:
    """
    pickle.dump(model, open(model_filepath, 'wb'))
    pass


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))

        X, Y, categories = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

        print('Building model...')
        model = build_model()

        print('Training model with grid search...')
        model.fit(X_train, Y_train)

        print('Best params of grid search...')
        print(model.best_params_)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, categories)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database ' \
              'as the first argument and the filepath of the pickle file to ' \
              'save the model to as the second argument. \n\nExample: python ' \
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()

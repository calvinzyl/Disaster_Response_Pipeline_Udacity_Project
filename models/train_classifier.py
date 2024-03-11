import sys
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import re

import nltk
nltk.download(['punkt', 'wordnet','stopwords','averaged_perceptron_tagger'])

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

class HasVerbExtractor(BaseEstimator, TransformerMixin):
    """
    This class defines a custom feature transformation
    that extracts an indicator from raw document showing
    if the document has verbs in it
    
    Attributes:
    -----------
    text str:
        an array-like collection of text documents
        
    Methods:
    --------
    fit_transform:
        Learn if each document contains verbs and return
        a Boolean
    """

    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            if any(pos_tags):
                word_list = [word[1] for word in pos_tags]
                if 'VB' in word_list or 'VBP' in word_list:
                    return True
        return False

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)


def load_data(database_filepath):
    """
    This function loads the post-ETL data from the SQLite
    database and split it into features and labels for ML
    modeling purposes
    
    Parameters:
    database_filepath (str): name of the database
    
    Returns:
    X (array-like): array of feature inputs
    y (array-like): array of labels/outputs
    categories_names (list): a list of category names
    """
    
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql("select * from disaster_data", engine)

    df = df.dropna()
    X = df.message
    Y = df.loc[:, 'related':]
    category_names = list(Y.columns)

    return X, Y, category_names


def tokenize(text):
    """
    This function cleans the text document, tokenize
    it, and lemmatizes the tokens
    
    Parameters:
    text (str): raw text document
    
    Returns:
    lemms (list): list of lemmas
    """
    
    if type(text) == float:
        return ''
    tmp = text.strip().lower()
    tmp = re.sub("'", ' ', tmp)
    tmp = re.sub('@[A-Za-z0-9_]+', ' ', tmp)
    tmp = re.sub('#[A-Za-z0-9_]+', ' ', tmp)
    tmp = re.sub(r'http\S+', ' ', tmp)
    tmp = re.sub(r'www.\S+', ' ', tmp)
    tmp = re.sub('[()!?]', ' ', tmp)
    tmp = re.sub('\[.*?\]',' ', tmp)
    tmp = re.sub('[^a-z0-9]',' ', tmp)
    tmp = re.sub('\d+', ' ', tmp)

    tokens = word_tokenize(tmp)
    tokens = [w for w in tokens if w not in stopwords.words('english')]
    lemms = [WordNetLemmatizer().lemmatize(w) for w in tokens]

    return lemms


def build_model():
    """
    This function combines the feature engineering
    and modeling parts into a pipeline and uses GridSearchCV
    to find the best parameter of the model
    
    Returns:
    cv: instance of a fitted estimator with best parameter
    """
    
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),
            ('starting_verb', HasVerbExtractor())
        ])),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters = {
        "clf__estimator__max_features": ['sqrt','log2']
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    This function evaluate the model prediction result against
    the test data based on precision, recall, and F1-score
    
    Parameters:
    model: instance of fitted estimator
    X_test (array-like): array of input features in the test data
    Y_test (array-like): array of output labels in the test data
    category names (list): list of category names
    
    Returns:
    None
    """
    
    y_pred = model.predict(X_test)

    print(f"Labels:\n{category_names}")

    for col, i in zip(Y_test.columns, range(Y_test.shape[0])):
        print(f"class-output: {col}")
        print(classification_report(np.array(Y_test)[:, i], y_pred[:, i]))

    accuracy = (y_pred == Y_test).mean()
    print(f"\nAccuracy:\n{accuracy}")
    print(f"\nBest Parameters:\n{model.best_params_}")


def save_model(model, model_filepath):
    """
    This function saves the fitted model to a pickle file
    
    Parameters:
    model: instance of fitted estimator
    model_filepath: name and path of the model pickle file

    Returns:
    None
    """
    
    import pickle
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
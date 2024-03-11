import json
import plotly
import sys
import pandas as pd
import numpy as np

import nltk
nltk.download(['punkt', 'wordnet','stopwords','averaged_perceptron_tagger'])

from nltk.stem import WordNetLemmatizer
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

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Scatter, Pie
#import plotly.graph_objects as go
from sklearn.externals import joblib
from sqlalchemy import create_engine

import re




app = Flask(__name__)

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

    #lemmatizer = WordNetLemmatizer()
    return lemms



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

# load data
engine = create_engine('sqlite:///../data/DisasterDB.db')
df = pd.read_sql_table('disaster_data', engine)

#engine = create_engine('sqlite:///DisasterDB.db')
#df = pd.read_sql("select * from disaster_data", engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')

def index():
    """
    This function renders the webpage that displays visuals and receives user
    input text for model
    
    Parameters:
    None
    
    Returns:
    Flask.render_template() result based on master.html and Plotly visuals
    """
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    top10_categories = df.loc[:, 'related':].sum().sort_values(ascending=False).head(10)
    top10_categories_names = list(top10_categories.index)
    top10_categories_values = list(top10_categories.values)
    
    df['message_len'] = df['message'].apply(len)
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    
    fig2_data = []
    for cat in top10_categories_names:
        cat_df = df[df[cat]==1].groupby('genre').agg({'message_len':np.mean})
        fig2_data.append(Bar(
                    x=cat_df.index,
                    y=cat_df['message_len'],
                    name=cat)
                    )
    
    graphs = [
        {
            'data': [
                Bar(x=top10_categories_names, y=top10_categories_values),
            ],

            'layout': {
                'title': 'Top 10 Message Categories',
                'yaxis': {'title': "Count"},
                'xaxis': {'title': "Category"}
            }
        },
        {
            'data': fig2_data,

            'layout': {
                'title': 'Message Length of Top 10 Categories by Genre',
                'yaxis': {'title': "average number of words"},
                'xaxis': {'title': "Genre"}
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    """
    This function handles user query and from the web page and displays model results
    
    Parameters:
    None
    
    Returns:
    Flask.render_template() method based on go.html and model results
    """
    
    
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()

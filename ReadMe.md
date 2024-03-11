# Overview

This project is part of the Udacity Data Scientist Nanodegree, the second project under the Data Engineering course, which includes the ETL process of the disaster text data, ML pipeline training a classification model, and a web app utilizing the model predictions and various visuals to showcase quite a bit of the data engineering and software engineering techniques.

# Data

Two raw data files are being used in this project: **categories.csv** and **messages.csv**, which originally came from [Appen](https://www.appen.com/). Messages.csv contains real messages that were sent during disaster events, while categories.csv contains the categories of the events associated with each of every messages in the messages dataset.

# Library

The following libraries and packages are required to run the scripts in the web app:
* json
* plotly
* sys
* pandas
* numpy as np
* nltk
  * (modules that need to be manually downloaded and installed) punkt, wordnet, stopwords, average perceptron tagger
* sklearn
* flask
* sqlalchemy
* re

# Code

* **process_data.py** <br>
  This script essentially performs the ETL process. It loads the raw CSV datasets, cleans and transforms them, including renaming columns, merging, and deduplication, and finally saves the data into the SQLite database. The post-ETL data is non-null and consists of raw message documents and corresponding event categories, which can be easily split into features, and labels, and are ready for additional preprocessing.
* **train_classifier.py** <br>
  This is the ML pipeline script, which preprocesses the raw documents through an NLP pipeline (removing special characters, tokenization, lemmatization, and TF-IDF vectorization), trains a multi-output random forest classifier, grid-searches the best parameters based on model evaluation, and saves the model to a Pickle file. The whole process utilizes the sklearn Pipeline and Feature Union functions, which streamlines the training/re-training iteration and allows an additional custom feature transformation function *HasVerbExtractor* to concatenate a "has-verb" indicator to the existing TF-IDF features.
* **master.html** <br>
  This is the HTML file that specifies the elements and layout of the web app. Stylistically it applies the Bootstrap NavBar convention. The user can input a new message and get classification results in one or several categories, and two Plotly visualizations are below the input bar summarizing message category frequencies and lengths of messages by genre.
* **run.py** <br>
  This is the Python script to deploy the web app, which includes specifying Flask backend, indexing webpage displaying cool visuals and receiving user input text for the model.

# Folder Structure

```
project
│   README.md
│   run.py
|   train_classifier.py
|   classifier.pkl
|   model_prediction.jpg
|   plotly_graphs.jpg    
│
└───data
│   │   DisasterDB.db
│   │   disaster_categories.csv
|   |   disaster_messages.csv
|   |   process_data.py
│   │
│   
└───templates
|   │   go.html
|   │   master.html
```

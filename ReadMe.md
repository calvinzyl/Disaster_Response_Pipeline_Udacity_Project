# Overview

This project is part of the Udacity Data Scientist Nanodegree, the second project under the Data Engineering course. It includes the ETL process of the disaster text data, an ML pipeline training a classification model, and a web app utilizing the model predictions and various visuals to showcase various data engineering and software engineering techniques.

# Data

This project uses two raw data files: **categories.csv** and **messages.csv**, which originally came from [Appen](https://www.appen.com/). Messages.csv contains real messages sent during disaster events, while categories.csv contains the categories of the events associated with each message in the messages dataset.

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
  This script essentially performs the ETL process. It loads the raw CSV datasets, cleans and transforms them, renames columns, merges, and deduplicates, and finally saves the data into the SQLite database. The post-ETL data is non-null and consists of raw message documents and corresponding event categories, which can be easily split into features and labels and are ready for additional preprocessing.
* **train_classifier.py** <br>
  This is the ML pipeline script, which preprocesses the raw documents through an NLP pipeline (removing special characters, tokenization, lemmatization, and TF-IDF vectorization), trains a multi-output random forest classifier, grid-searches the best parameters based on model evaluation, and saves the model to a Pickle file. The whole process utilizes the sklearn Pipeline and Feature Union functions, which streamlines the training/re-training iteration and allows an additional custom feature transformation function *HasVerbExtractor* to concatenate a "has-verb" indicator to the existing TF-IDF features.
* **master.html** <br>
  This is the HTML file specifying the web app's elements and layout. Stylistically, it applies the Bootstrap NavBar convention. The user can input a new message and get classification results in one or several categories. Two Plotly visualizations are below the input bar summarizing message category frequencies and lengths by genre.
* **run.py** <br>
  This is the Python script used to deploy the web app, which includes specifying the Flask backend, indexing the webpage, displaying cool visuals, and receiving user input text for the model.

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

# How to run the app?

1. Run the ETL pipeline that cleans data and stores it in the database
```
python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterDB.db
```
2. Run the ML pipeline that trains the classifier and saves the model
```
python models/train_classifier.py data/DisasterDB.db models/classifier.pkl
```
3. Run the web app
```
python app/run.py
```



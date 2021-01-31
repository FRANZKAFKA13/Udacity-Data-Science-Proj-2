# Udacity Data Science Project 2 (Disaster Response Pipeline)

This repository contains the code I wrote for tackling project 2 in the Data Scientist Nanodegree by Udacity.

## Dataset

In this project I used the [Placeholder](https://www.kaggle.com/). 

## Questions Answered

Lorem ipsum

## Deployment

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/disaster_response.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## Built With

* [Jupyter Notebook](https://jupyter.org) - Used for coding and documentation


## Authors

* **Carsten Granig**



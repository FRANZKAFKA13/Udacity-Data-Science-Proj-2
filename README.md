# Udacity Data Science Project 2: Disaster Response Pipeline

This repository contains the code I wrote for tackling project 2 in the [Data Scientist Nanodegree](https://www.udacity.com/course/data-scientist-nanodegree--nd025) by Udacity.


## Functionality

The web app allows the user to input a message via browser. The trained model is then used to find out if this message is related to a disaster and one or more related categories, such as "water", "food" or "medical help".


## Training Dataset

In this project I used two .csv files (`data`) with pre-labelled messages.


## Web App Preview

The search bar allows to type any kind of message

![Search Bar](assets/search_bar.jpg?raw=true "Search Bar")


The classified categories of the input message are highlighted

![Output Result](assets/result_classification.jpg?raw=true "Output Result")


## Deployment

1. Run the following commands in the project's root directory to set up your database and model:

    - To run ETL pipeline that cleans data from CSV and stores the result in a SQLite database:
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/disaster_response.db`
    - To run ML pipeline that trains classifier and saves a model in pickle format:
        `python models/train_classifier.py data/disaster_response.db models/classifier.pkl`

2. Run the following command in the app's directory (`cd app`) to run your web app:
    `python run.py`

3. Go to http://0.0.0.0:3001/ or http://localhost:3001/ to use the web app

## Built With

* [Jupyter Notebook](https://jupyter.org) - Used for coding and documentation
* [scikit-learn](https://scikit-learn.org/stable/install.html) - Used for training and storing the ML model
* [SQLAlchemy](https://www.sqlalchemy.org/) - Used for creating a SQLite database
* [Flask](https://flask.palletsprojects.com/en/1.1.x/) - Used for hosting the web app


## Authors

* [**Carsten Granig**](https://www.linkedin.com/in/carsten-granig/)



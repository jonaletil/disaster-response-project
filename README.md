# Disaster Response Pipeline Project
Udacity Data Scientists Nanodegree - Disaster Response Pipeline Project

- [Overview](#Project-Overview)
- [Project Components](#Components)
- [File Descriptions](#File-Descriptions)
- [Instructions](#how-to)

## Project Overview <a name="Project-Overview"></a>
This Project is part of Data Science Nanodegree Program by Udacity in collaboration with Figure Eight. The dataset contains pre-labelled tweet and messages from real-life disaster events. The project aim is to build a Natural Language Processing (NLP) model to categorize messages on a real time basis.


## Project Components <a name="Components"></a>
There are 3 main components in this project.
### 1. ETL pipeline
- Load two datasets
- Merge the sets and clean the data
- Store the data in a SQLite database

### 2. ML pipeline
- Load the clean data from the SQLite database
- Split the data to train-test sets
- Build a text processing and machine learning model with NLTK
- Train and tune the model with GridSearchCV
- Evaluate the model
- Export the final model as a pickle file

### 3. Flask Web App
A web application displays some visualization about the dataset. Users can type in any new messages in the web app and receive the categories that the message may belong to.

## File Description <a name="File-Descriptions"></a>
This is the high level description of all the files in this project.
```
├── app
│   ├── run.py--------------------------------# Flask file runs the web app
│   └── templates
│       ├── go.html---------------------------# Result page
│       └── master.html-----------------------# Main page
├── data
│   ├── DisasterResponse.db-------------------# *Database storing the processed data
│   ├── disaster_categories.csv---------------# Pre-labelled dataset
│   ├── disaster_messages.csv-----------------# Data
│   ├── process_data.py-----------------------# ETL pipeline processing the data
|   └── ETL Pipeline Preparation_NP.ipynb-----# Jupyter notebook with details
├── img---------------------------------------# Visualizations captured from the web app
├── models
|   ├── train_classifier.py-------------------# Machine learning pipeline
│   ├── ML Pipeline Preparation_NP.ipynb------# Jupyter notebook with details
|   └── classifier.pkl------------------------# *Pickle file

*Files that will be generated when the python scripts .py are executed.
```

## Instructions <a name="how-to"></a>
### 1. Download the files or clone this repository
  ```
  git clone https://github.com/jonaletil/disaster-response-project.git
  ```
### 2. Execute the scripts
a. Open a terminal <br>
b. Navigate to the project's root directory <br>
c. Run the following commands: <br>
- To run ETL pipeline that cleans data and stores in database
  ```
  python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
  ```
- To run ML pipeline that trains classifier and saves
  ```
  python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
  ```

d. Go to the app's directory and run the command
```sh
cd app
python run.py
```



# Disaster Response Pipeline Project
In this project, you will build a machine learning pipeline and a web application to classify disaster messages. The data comes from Appen (formerly known as Figure 8), containing real messages sent during disaster events. These messages need to be classified into various categories for effective disaster response by relief agencies.

The goal of this project is to demonstrate your skills in data engineering, machine learning, and web development. You will create:

ETL Pipeline: To clean, process, and store disaster data in a SQLite database.
ML Pipeline: To build a text classification model that processes disaster messages and categorizes them into relevant categories.
Flask Web Application: To allow emergency workers to input messages and see the classification results. The web app will also include visualizations of the data.
This project will showcase your ability to work with data pipelines, machine learning models, and web development to create a functional tool that can assist in disaster response.
### Here's the file structure of the project:

# Project Folder Structure

## app
- `template/`
  - `master.html`  
  - `go.html`
- `run.py`

## data
- `disaster_categories.csv`
- `disaster_messages.csv`
- `process_data.py`
- `TinhBDDatabaseName.db`

## models
- `train_classifier.py`
- `classifier.pkl`

## README.md

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/TinhDatabaseName.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/TinhDatabaseName.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3000/

## Web app
<img width="1246" alt="image" src="https://github.com/user-attachments/assets/76538c80-5050-4083-b9ab-0940512cf48d" />
<img width="1392" alt="image" src="https://github.com/user-attachments/assets/469f2eda-ba24-41a8-aba9-3c8f96827851" />


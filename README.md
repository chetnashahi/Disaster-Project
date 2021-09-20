# Disaster Response Pipeline Project
Disaster Response Pipeline project is to categorize the message sent during any diaster so that message can be sent to an appropriate Disaster Relief Agency.

### Installation
Below libraries are used:
- Pandas
- re
- sys
- json
- sklearn
- nltk
- sqlalchemy
- pickle
- Flask
- plotly
- sqlite3

### Project Descriptions
The project has three components:

#### ETL Pipeline: process_data.py file contains script to create ETL pipeline which:
-Loads the messages & categories datasets
-Merge two datasets
-Cleans data
-Store it in SQLite database

#### ML Pipeline: train_classifier.py file contains script to create ML pipeline which:
-Loads data from SQLite database
-Splits the dataset into training & test sets
-Builds a text processing & machine learning pipeline
-Trains & tunes a model using GridSearchCV
-Output the result on test set
-Export output as pickle file

#### Flask Web App: Web app enables user to enter disaster message & then view categories of message

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

import os
import sys
import numpy as np
import pandas as pd
from src.exception import CustomException
import dill #helps in creating pickle
from sklearn.metrics import r2_score

def save_object(file_path, obj):
    try:
        # Get the directory path from the file path
        dir_path=os.path.dirname(file_path)
        
        # Create the directory if it doesn't exist
        os.makedirs(dir_path, exist_ok=True)

        # Open the file for writing in binary mode
        with open(file_path, "wb") as file_obj:
            # Serialize the object and write it to the file
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e,sys)
    
#Evalution, similar to which we did in 2. MODEL TRAINING.ipynb
def evaluate_model(X_train, y_train, X_test, y_test, models):
    try:
        report={}

        for i in range(len(list(models))):
            model=list(models.values())[i]#stored as list containing values of dictionary, values are regression function
            model.fit(X_train, y_train) # Train model

            # Make predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_model_score=r2_score(y_train, y_train_pred)
            test_model_score=r2_score(y_test, y_test_pred)

            #Models and thier r2 scores 
            report[list(models.keys())[i]]=test_model_score

        return report
    
    except Exception as e:
        raise CustomException(e,sys)


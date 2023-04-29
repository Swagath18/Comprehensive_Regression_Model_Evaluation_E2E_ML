import os
import sys
import numpy as np
import pandas as pd
from src.exception import CustomException
import dill #helps in creating pickle

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
import os
import sys
from src.exception import CustomException #from exception we are importing custom expection which we created 
from src.logger import logging

import pandas as pd 
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

# This decorator is used to define class variables instead of init and we can directly define variables 
@dataclass #better to use only when defining variables
#inputs to the data ingestion components are gives below and dataingestion componet will save the output in given path below.
class DataIngestionConfig: 
    #output path for traindata stored in artifacts folder in train.csv file
    train_data_path: str=os.path.join('artifacts', "train.csv")
    # similary for test data 
    test_data_path: str=os.path.join('artifacts', "test.csv")
    # Initial data as raw data
    raw_data_path: str=os.path.join('artifacts', "data.csv")

class DataIngestion:
    def __init__(self):
        # initializing the inputs that is present in DataIngestionConfig class
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            #reading the data
            df= pd.read_csv('notebook\data\StudentsPerformance.csv') #by changing this line we can read data from other sources 
            logging.info("Read the dataset as dataframe")
            
            #This creates a directory to store the training data file using the path specified in self.ingestion_config.train_data_path
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train test split initiated")
            train_set, test_set=train_test_split(df, test_size=0.2, random_state=42)

            #This saves the training and testing sets to the paths specified
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False, header=True)

            logging.info("Ingestion of the data is completed")

            return(self.ingestion_config.train_data_path,
                   self.ingestion_config.test_data_path
            )
        

        except Exception as e:
            raise CustomException(e,sys)
            
if __name__=="__main__":
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()

    data_tranformation=DataTransformation()
    data_tranformation.initiate_data_transformation(train_data, test_data)


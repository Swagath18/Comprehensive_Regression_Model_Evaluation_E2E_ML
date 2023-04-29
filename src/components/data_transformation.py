import os
import sys
from dataclasses import dataclass
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer # Combines categorical and numerical in array (onehot and standardscaler)
from sklearn.impute import SimpleImputer # This is to fill the missing value with mean/median of that column
from sklearn.pipeline import Pipeline # To scale and train dataset
from sklearn.preprocessing import OneHotEncoder, StandardScaler 

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object

#In Dataingestion we created a class to read input and save in artifacts, similar process
@dataclass
class DataTransformationConfig:
    #creating preprocessor object file path with pickle file in artifacts folder
    preprocessor_obj_file_path=os.path.join('artifacts', "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        #Now we need initialize the path class
        #data_transformation_config will have variable of above class
        self.data_transformation_config=DataTransformationConfig() 

    #Now to perform all the transformation , from EDA we know what are numerical and categorical feature we have to use those
    def get_data_transformer_object(self):
        '''
        This function is for data transformation
        '''
        try:
            numerical_columns=["writing score", "reading score"]
            categorical_columns=["gender", "race/ethnicity", "parental level of education", "lunch", "test preparation course"]

            # To create a pipeline to train and to do standardscale
            #we are using median as we have seen there was outliers
            num_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")), 
                    ("scaler", StandardScaler())
                ]
            )
            #most_frequent=Missing values are replaced by most frequent values
            #Setting with_mean=False will only scale the data by dividing each feature by its SD(i did this because of error)
            cat_pipeline=Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")), 
                    ("one_hot_encoder",OneHotEncoder()),
                    ("scaler",StandardScaler(with_mean=False))
                ]
            )

            #logging("Numerical columns standard scaling is completed")
            #logging("Categorical columns onehotencoding is completed")

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            #Now we need to combine both categirical and numerical features 
            #Column Transforemr
            preprocessor=ColumnTransformer(
                [
                    ("num_pipline",num_pipeline, numerical_columns),
                    ("cat_piplines",cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)
    
    #Starting data transformation inside this fuction
    def initiate_data_transformation(self,train_path,test_path): #train and test path we get it from data ingestion

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            logging.info("Read train and test data completed")

            # To read all preprocessor object
            logging.info("obtaining preprocessing object")
            preprocessing_obj=self.get_data_transformer_object()

            #creating target column to predict math score
            target_column_name="math score"
            numerical_columns= ["writing score","reading score"]

            #droping target column from input feature for training
            input_feature_train_df=train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df=train_df[target_column_name]

            #droping target column from input feature for test
            input_feature_test_df=test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df=test_df[target_column_name]

            #we need to apply these preprocessing 
            logging.info("Applying preprocessing object on training and testing dataframe")

            #___Here using fit transofer method, we fit the transformed data to input size__
            #training data is scaled based on its own mean and standard deviation
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            #Test data is scaled based on the mean and standard deviation of the training data
            #This ensures that the test data is scaled in the same way as the training data
            input_feature_test_arr=preprocessing_obj.fit_transform(input_feature_test_df)


            train_arr=np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            #np.c_ = concatenates along 2nd axis even for different dimension

            test_arr=np.c_[input_feature_test_arr,np.array(target_feature_test_df)]

            logging.info("saved preprocessing object.")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            # To save this object we will be writing it in utils.py

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e,sys)
            
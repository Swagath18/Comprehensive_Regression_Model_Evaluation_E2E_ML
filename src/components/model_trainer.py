import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor)
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model

#In Dataingestion we created a class to read input and save in artifacts, similar process
@dataclass
class ModelTrainerConfig:
    #This variable is created to save the model to pkl file once it has been created
    trained_model_file_path=os.path.join("artifacts","model.pkl")

#This class is defined for training the model
class ModelTrainer:
    def __init__(self):
        #inside this variable we gonna get the path name of ModelTrainerCofig class
        self.model_trainer_config=ModelTrainerConfig()

    #Here the inputs are output of data transformer
    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("split training and test input data") #This is returned by data transformation
            #Splitting the last column which was concatenated in data transformation
            X_train,y_train,X_test, y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            #once we are done with splitting training and testing data, we create dictionary of model
            models={
                "Random Forest": RandomForestRegressor(),
                "Decision tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-neighbors Regressor": KNeighborsRegressor(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(),
                "AdaBoost Regressor": AdaBoostRegressor()
            }

            #now to do hyper parameter tuning
            params={
                "Decision tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    # 'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "K-neighbors Regressor":{
                    'n_neighbors':[5,7,9,11],
                    # 'weights':['uniform','distance'],
                    # 'algorithm':['ball_tree','kd_tree','brute']
                },
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoosting Regressor":{
                    'depth': [6,8,10],
                    # 'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }

            #creating dictionary report, evalue function is in utils 
            #params is added for hyperparameter tuning
            model_report:dict=evaluate_model(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models,params=params)

            ## To get the best model score from dictionary, sorting is done
            best_model_score=max(sorted(model_report.values()))
            ## To get the best model name from dict, 
            ## the best score is passed as index of which the values of model and that is passed as index for model name
            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model=models[best_model_name]

            if best_model_score <0.6:
                raise CustomException("No best model found")
            logging.info("Best found model on both training and testing dataset")

            #file path and obj is input for save_object in utils which dumps into pickle file
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)
            r2_square=r2_score(y_test, predicted)

            return r2_square
            
        except Exception as e:
            raise CustomException(e,sys)


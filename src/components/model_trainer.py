import os
import sys
from dataclasses import dataclass

import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.ensemble import AdaBoostRegressor,GradientBoostingRegressor,RandomForestRegressor
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_model

@dataclass
class ModeTrainerConfig:
    trained_model_file = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModeTrainerConfig()

    def initialte_model_training(self,train_array,test_array):
        try:
            logging.info('Splitting training and test inputs')
            xtrain,ytrain,xtest,ytest = (
                train_array[:,:-1],train_array[:,-1],
                test_array[:,:-1],test_array[:,-1]
            )
            models = {'RandomForest':RandomForestRegressor(),
                      'Decision Tree': DecisionTreeRegressor(),
                      'Gradient Boostomg':GradientBoostingRegressor(),
                      'Linear Regression': LinearRegression(),
                      'K-Neighbor Regressor' : KNeighborsRegressor(),
                      'XGBoost' : XGBRegressor(),
                      'CatBoostRegressor': CatBoostRegressor(),
                      'AdaBosst Regressor' : AdaBoostRegressor()
                      }
            model_report : dict= evaluate_model(xtrain=xtrain,ytrain=ytrain,xtest=xtest,ytest=ytest,models = models)

            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if(best_model_score<0.6):
                raise CustomException("No best model found")
            logging.info('Cest model found on both test and train')
            save_object(
                file_path = self.model_trainer_config.trained_model_file,
                obj = best_model
            )
            predicted = best_model.predict(xtest)
            r2_square = r2_score(ytest,predicted)
            return r2_square

        except Exception as e:
            raise CustomException(e,sys)
            


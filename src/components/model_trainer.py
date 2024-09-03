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
                      'Gradient Boosting':GradientBoostingRegressor(),
                      'Linear Regression': LinearRegression(),
                      'K-Neighbor Regressor' : KNeighborsRegressor(),
                      'XGBoost' : XGBRegressor(),
                      'CatBoostRegressor': CatBoostRegressor(),
                      'AdaBoost Regressor' : AdaBoostRegressor()
                      }
            params = {
                'RandomForest' : {'max_depth' : [3,4,5,6],'min_samples_split' : [3,5,7],'n_estimators' : [8,16,32,64,128]},
                'Decision Tree' : {'max_depth' : [3,4,5,6],'min_samples_split' : [3,5,7]},
                'Gradient Boosting' : {'n_estimators' : [8,16,32,64,128],'max_depth'  : [3,5,7]},
                'Linear Regression' : {},
                'K-Neighbor Regressor' : {'weights' : ['uniform','distance'], 'n_neighbors' : [3,4,5] },
                'XGBoost' : {'eta' : [0.001,0.005,0.01,0.05,0.1],'n_estimators' : [8,16,32,64,128]},
                'CatBoostRegressor' : {'depth' : [6,8,10], 'learning_rate' : [.1,.05,.001]},
                'AdaBoost Regressor' : {'learning_rate' : [.1,.05,.01,.05,.001],'n_estimators': [8,16,32,64,128]}

            }
            model_report : dict= evaluate_model(xtrain=xtrain,ytrain=ytrain,xtest=xtest,ytest=ytest,models = models,params = params)

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
            


import os
import sys
import pandas as pd
import numpy as np  
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from src.exception import CustomException
import dill
from src.logger import logging

def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,'wb') as file_obj:
            dill.dump(obj,file_obj)
    except Exception as e:
        raise CustomException(e,sys)
def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e,sys)

def evaluate_model(xtrain,ytrain,xtest,ytest,models,params):
    try:
        report = {}

        trained_model = []
        eval_score = []
        model_names = []
        for i in range(len(list(models))):
            model = list(models.values())[i]
            param = list(params.values())[i]
            gs = GridSearchCV(model,param,cv=3)
            gs.fit(xtrain,ytrain)
            model.set_params(**gs.best_params_)
            logging.info('The best parameters for %s is %s',str(model),str(gs.best_params_))
            model.fit(xtrain,ytrain)
            y_train_pred = model.predict(xtrain)
            y_test_pred = model.predict(xtest)

            train_model_score = r2_score(ytrain,y_train_pred)
            test_model_score = r2_score(ytest,y_test_pred)

            # trained_model.append(m)
            # eval_score.append(m.score(xtest,ytest))
            # model_names.append(model)
            report[list(models.keys())[i]] = test_model_score
        return report

    except Exception as e:
        raise(CustomException(e,sys))
    
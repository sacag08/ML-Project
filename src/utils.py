import os
import sys
import pandas as pd
import numpy as np  
import dill
from src.exception import CustomException
from sklearn.metrics import r2_score

def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,'wb') as file_obj:
            dill.dump(obj,file_obj)
    except Exception as e:
        raise CustomException(e,sys)
def evaluate_model(xtrain,ytrain,xtest,ytest,models):
    try:
        report = {}

        trained_model = []
        eval_score = []
        model_names = []
        for i in range(len(list(models))):
            model = list(models.values())[i]

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
    
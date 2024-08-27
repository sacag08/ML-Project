import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object
@dataclass
class DataTransformationConfig:
    preprocessor_file_path = os.path.join('artifacts','prepocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        ''' This function is responcible for dat transformation'''
        try:
            numerical_columns = ["writing_score","reading_score"]
            categorical_features = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"]
            num_pipeline = Pipeline(
                steps= [
                    ('fillna',SimpleImputer(strategy='median')),
                    ('sc',StandardScaler(with_mean=False))
                ]
            )
            cat_pipeline = Pipeline(
                [
                    ('fillna',SimpleImputer(strategy = 'most_frequent')),
                    ('ohe',OneHotEncoder()),
                    ('sc',StandardScaler(with_mean=False))
                ]
            )

            logging.info('Categorical encoding done')
            logging.info('Numerical scaling done')
            preprocessor = ColumnTransformer(
                [
                    ("numerical scaling",num_pipeline,numerical_columns),
                    ("cat encoding",cat_pipeline,categorical_features)
                ]
            )
            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info('read train and test data completed')
            logging.info('obtaining preprocessing object')
            preprocessing_object = self.get_data_transformer_object()
            target_column_name = 'math_score'
            numerical_columns = ["writing_score","reading_score"]
            
            input_feature_train_df = train_df.drop(columns=[target_column_name],axis=1) 
            target_feature_train_df = train_df[target_column_name]
            
            input_feature_test_df = test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info('Applying preprocessor object on training dataframeand testing dataframe')
            
            input_feature_train_arr = preprocessing_object.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_object.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr,np.array(target_feature_train_df)
            ]

            test_arr = np.c_[
                input_feature_test_arr,np.array(target_feature_test_df)
            ]

            logging.info("Saved preprocessing object")

            save_object(
                file_path = self.data_transformation_config.preprocessor_file_path,
                obj = preprocessing_object
            )
            return (
                train_arr,test_arr,self.data_transformation_config.preprocessor_file_path
                
            )

        except Exception as e:
            raise CustomException(e,sys)
            
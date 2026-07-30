import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class datatransformationconfig:
    preprocessor_obj_file_path = os.path.join('artifical' , "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformer_config= datatransformationconfig()

    def get_data_transformer_objects(self):
        try:
            numerical_columns =["writing_score" , "reading_score"]
            categorical_columns =[
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",

            ]
            num_pipeline=Pipeline(
                    steps=[
                        ("imputer" , SimpleImputer(strategy="median")), #used for misssing values
                        ("scaler" , StandardScaler()),
                    ]
            )
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer" ,SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder" , OneHotEncoder()),
                    ("scaler",StandardScaler(with_mean=False)),
                ]
            )
            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor=ColumnTransformer(
                [
                    ("num_pipeline" ,num_pipeline,numerical_columns),
                    ("cat_pipeline" ,cat_pipeline,categorical_columns)
                ]
            )

            return preprocessor
        except Exception as e:
               raise CustomException(e,sys)


    def initiate_data_tarnsformation(self,train_path,test_path):
         try:
              train_df=pd.read_csv(train_path)
              test_df=pd.read_csv(test_path)
              logging.info("Read train test data completed")
              logging.info("obtaing preprocessor object")


              preprocessing_obj=self.get_data_transformer_objects()

              target_column_name="math_score"
              numerical_columns = ["writing_score","reading_score"]

              #dropping the target column from train dataset
              input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
              target_feature_train_df=train_df[target_column_name]

                #dropping the target dataset from test dataset
              input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
              target_feature_test_df=test_df[target_column_name]

              logging.info(f"applying preprocessing for test and train dataframe")

              input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
              input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

              train_arr=np.c_[
                   input_feature_train_arr,np.array(target_feature_train_df)
              ]
              test_arr=np.c_[
                   input_feature_test_arr,np.array(target_feature_test_df)
              ]
              logging.info(f"saving preprocessing file")


                #saving pickle file which is created in utils.py
              save_object(
                   file_path=self.data_transformer_config.preprocessor_obj_file_path,
                   obj=preprocessing_obj
              )

              return (
                   train_arr,
                   test_arr,
                   self.data_transformer_config.preprocessor_obj_file_path,
              )
         except Exception as e:
              raise CustomException(e,sys)
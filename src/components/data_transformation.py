import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from dataclasses import dataclass
from src.exception import CustomException
from src.utils import saveObj
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

class DataTransformation:
    def __init__(self):
        pass

    def initiateDataTransformation(self, train_path, test_path):
        try:
            training_data = pd.read_csv(train_path)
            testing_data = pd.read_csv(test_path)

            target_column = 'math score'
            X_train = training_data.drop(columns=[target_column])
            y_train = training_data[target_column]

            X_test = testing_data.drop(columns=[target_column])
            y_test = testing_data[target_column]

            num_features = X_train.select_dtypes(exclude='object').columns
            cat_features = X_train.select_dtypes(include='object').columns

            preprocessor = ColumnTransformer([
                ('OneHotEncoding', OneHotEncoder(), cat_features),
                ('StandardScaling', StandardScaler(), num_features)
            ])

            X_train_transformed = preprocessor.fit_transform(X_train)
            X_test_transformed = preprocessor.transform(X_test)

            
            saveObj("artifacts/preprocessor.pkl", preprocessor)

            
            train_arr = np.c_[X_train_transformed, np.array(y_train)]
            test_arr = np.c_[X_test_transformed, np.array(y_test)]

            return train_arr, test_arr

        except Exception as e:
            raise CustomException(e, sys)

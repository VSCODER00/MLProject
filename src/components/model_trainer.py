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
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from xgboost.sklearn import XGBRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
class ModelTrainer:
    def __init__(self):
        pass
    def TraintheModel(self,training_data,testing_data):
        x_train=training_data[:,:-1]
        x_test=testing_data[:,:-1]
        y_train=training_data[:,-1]
        y_test=testing_data[:,-1]
        models = {
            "LinearRegression": (LinearRegression(), {}),
            "Ridge": (Ridge(), {"alpha": [0.1, 1.0, 10.0]}),
            "Lasso": (Lasso(), {"alpha": [0.001, 0.01, 0.1, 1.0]}),
            "ElasticNet": (ElasticNet(), {"alpha": [0.1, 1.0], "l1_ratio": [0.2, 0.5, 0.8]}),
            "SGD": (SGDRegressor(), {"penalty": ["l2", "l1"], "alpha": [0.0001, 0.001]}),
            "SVR": (SVR(), {"kernel": ["linear", "rbf"], "C": [0.1, 1, 10]}),
            "XGBoost": (XGBRegressor(), {"n_estimators": [50, 100], "max_depth": [3, 5]})
        }
        overview={}
        for name, (model, param_grid) in models.items():
            if param_grid:
                grid_search = GridSearchCV(model, param_grid, cv=3, n_jobs=-1, scoring='r2')
                grid_search.fit(x_train, y_train)
                best_model = grid_search.best_estimator_
                best_params = grid_search.best_params_
            else:
                model.fit(x_train, y_train)
                best_model = model
                best_params = "Default params"

            y_pred = best_model.predict(x_test)
            score = r2_score(y_test, y_pred)

            overview[name] = score
                
            
        
        return overview


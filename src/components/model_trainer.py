import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object
from src.utils import evaluate_model
from sklearn.model_selection import GridSearchCV

@dataclass
class ModelTrainerConfig:
    trained_model_fit_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("split training and test input data")        

            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1],
            )

            models= {
                "random forest":RandomForestRegressor(),
                "Decion tree":DecisionTreeRegressor(),
                "Gradient boosting": GradientBoostingRegressor(),
                "Linear regression": LinearRegression(),
                "K-nearest neighbout":KNeighborsRegressor(),
                "XGBclassifier":XGBRegressor(),
                "CatBoosting Classfier":CatBoostRegressor(verbose=False),
                "adaboosst":AdaBoostRegressor(),
            }
            # Define the hyperparameters for each algorithm
            param_grid = {
                "random forest": {
                    "n_estimators": [100, 200, 300],
                    "max_depth": [None, 5, 10],
                    "min_samples_split": [2, 5, 10],
                },
                "Decion tree": {
                    "max_depth": [None, 5, 10],
                    "min_samples_split": [2, 5, 10],
                },
                "Gradient boosting": {
                    "n_estimators": [100, 200, 300],
                    "learning_rate": [0.1, 0.01, 0.001],
                    "max_depth": [3, 5, 10],
                },
                "Linear regression": {},
                "K-nearest neighbout": {
                    "n_neighbors": [3, 5, 7],
                    "weights": ["uniform", "distance"],
                },
                "XGBclassifier": {
                    "n_estimators": [100, 200, 300],
                    "learning_rate": [0.1, 0.01, 0.001],
                    "max_depth": [3, 5, 10],
                },
                "CatBoosting Classfier": {
                    "iterations": [100, 200, 300],
                    "learning_rate": [0.1, 0.01, 0.001],
                    "depth": [3, 5, 10],
                },
                "adaboosst": {
                    "n_estimators": [50, 100, 200],
                    "learning_rate": [0.1, 0.01, 0.001],
                    "loss": ["linear", "square", "exponential"],
                },
            }
           

            model_report=evaluate_model(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models,param_grid=param_grid)

            best_model_score=max(sorted(model_report.values()))

            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model=models[best_model_name]

            if best_model_score <0.6:
                raise CustomException("no best model found")
            
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_fit_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)

            r2_s=r2_score(y_test,predicted)
            
            return r2_s,best_model_name
        
        except Exception as e:
            raise CustomException(e,sys)
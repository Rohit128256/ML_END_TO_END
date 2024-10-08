import os
import sys
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import AdaBoostRegressor,GradientBoostingRegressor,RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
import yaml

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifact","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("splitting train and test data")
            x_train,y_train,x_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
                "Random Forest" : RandomForestRegressor(),
                "Decision Tree" : DecisionTreeRegressor(),
                "Gradient Boosting" : GradientBoostingRegressor(),
                "Linear Regression" : LinearRegression(),
                "KNN Regressor" : KNeighborsRegressor(),
                "XGB Regressor" : XGBRegressor(),
                "AdaBoost Regressor" : AdaBoostRegressor(),
                "Catboost Regressor" : CatBoostRegressor()
            }

            with open("hyperparameter.yaml","r") as file:
                param = yaml.safe_load(file)

            model_report : dict = evaluate_models(x_train = x_train,y_train = y_train,x_test = x_test,y_test = y_test,models = models,params=param) 
            
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found",sys)
            
            logging.info(f"Best model found on traing and testing data")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )

            predicted = best_model.predict(x_test)

            r2_scr = r2_score(y_test, predicted)

            return r2_scr

        except Exception as e:
            raise CustomException(e,sys)
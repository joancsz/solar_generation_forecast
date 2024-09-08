from config import config
from .visualization import plot_generation_predicted_vs_observed
from .operational import negative_to_zero

import pandas as pd
import numpy as np
import os

from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression

from skopt import gp_minimize
import joblib

cfg = config['train']

class Trainer:

    def __init__(self, data: pd.DataFrame, target_name:str=None):
        self.build_train_test_dataframes(data, target_name)
    
    def build_train_test_dataframes(self, data, target_name) -> None:
        
        if not target_name:
            target_name = cfg['default_target_name']

        X = data.drop(target_name, axis=1)
        y = data[target_name]

        end_train = int(len(data)*0.6) #60% of the data from training
        end_test = int(len(data)*0.9)  #30% of the data from testing and 10% for validation

        self.X_train, self.X_test, self.X_val = X.iloc[:end_train], X.iloc[end_train:end_test], X.iloc[end_test:] 
        self.y_train, self.y_test, self.y_val = y.iloc[:end_train], y.iloc[end_train:end_test], y.iloc[end_test:] 
        self.X_train_for_val = X.iloc[:end_test]
        self.y_train_for_val = y.iloc[:end_test]

    def _random_forest_params_list_to_dict(self, params):
        n_estimators = params[0]
        max_depth = params[1]
        min_samples_split = params[2]
        min_samples_leaf = params[3]
        max_features = params[4]
        bootstrap = params[5]
        
        params_dict = {'n_estimators': n_estimators,
                        'max_depth': max_depth,
                        'min_samples_split': min_samples_split,
                        'min_samples_leaf': min_samples_leaf,
                        'max_features': max_features,
                        'bootstrap': bootstrap
                        }
                    
        return params_dict

    def _xgboost_params_list_to_dict(self, params):
        learning_rate = params[0]
        max_depth  = params[1]
        min_child_weight  = params[2]
        subsample = params[3]
        colsample_bytree = params[4]
        n_estimators = params[5]
        gamma = params[6]
        reg_lambda = params[7]
        reg_alpha = params[8]

        params_dict = {'learning_rate': learning_rate,
                        'max_depth': max_depth,
                        'min_child_weight': min_child_weight,
                        'subsample': subsample,
                        'colsample_bytree': colsample_bytree,
                        'n_estimators': n_estimators,
                        'gamma': gamma,
                        'reg_lambda': reg_lambda,
                        'reg_alpha': reg_alpha
                        }
                    
        return params_dict

    def _train_xgboost(self, params):


        learning_rate = params[0]
        max_depth  = params[1]
        min_child_weight  = params[2]
        subsample = params[3]
        colsample_bytree = params[4]
        n_estimators = params[5]
        gamma = params[6]
        reg_lambda = params[7]
        reg_alpha = params[8]

        dict_params = f"""
        'learning_rate': {learning_rate},
        'max_depth': {max_depth},
        'min_child_weight': {min_child_weight},
        'subsample': {subsample},
        'colsample_bytree': {colsample_bytree},
        'n_estimators': {n_estimators},
        'gamma': {gamma},
        'reg_lambda': {reg_lambda},
        'reg_alpha': {reg_alpha}
        """
        print(dict_params, '\n')

        xgb = XGBRegressor(learning_rate=learning_rate, max_depth = max_depth,
                            min_child_weight = min_child_weight,
                            subsample = subsample, colsample_bytree = colsample_bytree,
                            gamma = gamma, reg_lambda = reg_lambda, reg_alpha = reg_alpha,
                            random_state=0, n_estimators=n_estimators)

        xgb.fit(self.X_train, self.y_train)

        p = xgb.predict(self.X_test)
        p = negative_to_zero(p)

        return mean_absolute_error(self.y_test, p)

    def _train_random_forest(self, params):

        n_estimators = params[0]
        max_depth = params[1]
        min_samples_split = params[2]
        min_samples_leaf = params[3]
        max_features = params[4]
        bootstrap = params[5]
        
        dict_params = f"""
        'n_estimators': {n_estimators},
        'max_depth': {max_depth},
        'min_samples_split': {min_samples_split},
        'min_samples_leaf': {min_samples_leaf},
        'max_features': {max_features},
        'bootstrap': {bootstrap},
        """

        print(dict_params, '\n')

        rf = RandomForestRegressor(
            n_estimators=n_estimators, 
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            bootstrap=bootstrap,
            random_state=0
        )
        
        rf.fit(self.X_train, self.y_train)
        
        p = rf.predict(self.X_test)
        p = negative_to_zero(p)

        return mean_absolute_error(self.y_test, p)

    def train(self, model_name, space=None, n_calls=10):
        """
        Function that efectively does the Optimzation and return the best parameters for the weak learners
        """
        if model_name == 'xgboost':
            if not space:
                space = [tuple(item) for item in cfg['xgboost_space']]
            results = gp_minimize(self._train_xgboost, space, random_state=1, verbose=1, n_calls=n_calls, n_random_starts=10)
        elif model_name == 'randomforest':
            if not space:
                space = [tuple(item) for item in cfg['random_forest_space']]
            results = gp_minimize(self._train_random_forest, space, random_state=1, verbose=1, n_calls=n_calls, n_random_starts=10)
        else:
            print('Not supported model')

        best_params_list = results.x_iters[np.argmin(results.func_vals)]

        if model_name == 'xgboost':
            best_params = self._xgboost_params_list_to_dict(best_params_list)
        elif model_name == 'randomforest':
            best_params = self._random_forest_params_list_to_dict(best_params_list)
        
        return best_params

    def save_model(self, model, path=''):
        """
        Saves model to the model folder
        """
        if not path:
            path = os.path.join('model/', 'model.joblib')
        joblib.dump(model, path)

    def train_xgboost_and_dump(self, save=True):
        """
        Train and save xgboost model
        """ 
        xg_params = self.train(model_name='xgboost')
        model = XGBRegressor(**xg_params)
        model.fit(self.X_train_for_val, self.y_train_for_val)
        p = model.predict(self.X_val)
        p = negative_to_zero(p)

        print('MAE (MW):', mean_absolute_error(self.y_val, p))

        if save:
            self.save_model(model)

        print('Plotting at imgs/')
        plot_generation_predicted_vs_observed(pd.Series(p,self.y_val.index),
                                            self.y_val,
                                            title='Generation Predicted vs Observed',
                                            path_to='imgs/pred_vs_obs_final_model.jpg')

    def train_stacked_model(self, save=True) -> None:
        """
        Train and save the main model
        """ 
        print('Finding parameters for each model')

        xg_params = self.train(model_name='xgboost')
        rf_params = self.train(model_name='randomforest')

        estimators = [('xgb', XGBRegressor(**xg_params)),
                    ('rf', RandomForestRegressor(**rf_params))]

        print('Evaluating the final model')
        final_model = StackingRegressor(estimators=estimators, final_estimator=LinearRegression(), n_jobs=-1, cv=5)
        final_model.fit(self.X_train_for_val, self.y_train_for_val)
        p = final_model.predict(self.X_val)
        p = negative_to_zero(p)

        print('MAE (MW):', mean_absolute_error(self.y_val, p))

        if save:
            self.save_model(final_model)

        print('Plotting at imgs/')
        plot_generation_predicted_vs_observed(pd.Series(p,self.y_val.index),
                                            self.y_val,
                                            title='Generation Predicted vs Observed',
                                            path_to='imgs/pred_vs_obs_final_model.jpg')

        return final_model




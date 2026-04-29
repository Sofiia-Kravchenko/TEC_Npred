import json

import optuna
import tensorflow as tf

from catboost import CatBoostRegressor
from keras.optimizers import Adam
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.linear_model import LinearRegression

from utils import optuna_cbr_search, optuna_rfr_search, optuna_lstm_search, scale_combined, custom_loss

tf.random.set_seed(42)


def get_catboost(X_train_s, X_test_s, y_train_s, y_test_s, y_train, y_test, scaler_y):

    study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner())
    study.optimize(lambda trial: optuna_cbr_search(trial, X_train_s, y_train_s,
                                               X_test_s, y_test_s, scaler_y),
                   n_trials=20)

    best_params = study.best_params

    model = CatBoostRegressor(iterations=2000,
                              depth=best_params['depth'],
                              learning_rate=best_params['lr'],
                              loss_function='MAE',
                              verbose=0)
    return model

def get_lstm(input_shape, X_train_s, X_test_s, y_train_s, y_test_s, y_train, y_test, scaler_y, y_train_s_combined, y_test_s_combined):
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: optuna_lstm_search(trial, X_train_s, y_train_s,
                                               X_test_s, y_test_s, scaler_y, input_shape, y_train_s_combined, y_test_s_combined), n_trials=20)
    best_params = study.best_params
    model = Sequential([
        LSTM(best_params['n_units_lstm'], input_shape=input_shape),
        Dense(best_params['n_units_dense']),
        Dense(1)
    ])
    optimizer = Adam(learning_rate=best_params['lr'])
    model.compile(optimizer=optimizer, loss=custom_loss)
    return model

def get_linear():
    return LinearRegression()

def get_simple_mlp(input_shape):
    model = Sequential([
            Dense(128, activation='relu',input_shape=input_shape),
            Dense(1)
        ])
    model.compile(optimizer='adam', loss=custom_loss)
    return model

def get_mlp(input_shape):
    model = Sequential([
        Dense(64, activation='relu', kernel_initializer='he_normal'),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(1)
        ])
    model.compile(optimizer='adam', loss=custom_loss)
    return model

def get_rfr(X_train_s, X_test_s, y_train_s, y_test_s, y_train, y_test, scaler_y):
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: optuna_rfr_search(trial, X_train_s, y_train_s, X_test_s, y_test_s, scaler_y),
                   n_trials=20)

    best_params = study.best_params
    params = {
        "n_estimators": best_params['n_estimators'],
        "max_depth": best_params['max_depth'],
        "min_samples_split": best_params['min_samples_split'],
        "min_samples_leaf": best_params['min_samples_leaf'],
        "max_features": best_params['max_features'],
        "n_jobs": -1
    }
    return RandomForestRegressor(**params)
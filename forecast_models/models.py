import optuna
import tensorflow as tf

from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.linear_model import LinearRegression

from utils import optuna_cbr_search, optuna_rfr_search

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

def get_lstm(input_shape):
    model = Sequential([
        LSTM(50, input_shape=input_shape),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mae')
    return model

def get_linear():
    return LinearRegression()

def get_simple_mlp(input_shape):
    model = Sequential([
            Dense(128, activation='relu',input_shape=input_shape),
            Dense(1)
        ])
    model.compile(optimizer='adam', loss='mae')
    return model

def get_mlp(input_shape):
    model = Sequential([
        Dense(64, activation='relu', kernel_initializer='he_normal'),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(1)
        ])
    model.compile(optimizer='adam', loss='mae')
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
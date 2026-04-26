import tensorflow as tf

from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.linear_model import LinearRegression
from sklearn import metrics

tf.random.set_seed(42)

def get_catboost(X_train_s, X_test_s, y_train_s, y_test_s, y_train, y_test, scaler_y):
    learning_rates = [0.01, 0.05, 0.1, 0.2]
    depths = [4, 6, 8, 10]
    best_mae = float('inf')
    best_params = {'depth': 6, 'lr': 0.01}  # значения по умолчанию

    for d in depths:
        for lr in learning_rates:
            test_model = CatBoostRegressor(iterations=500,
                                           depth=d,
                                           learning_rate=lr,
                                           loss_function='MAE',
                                           verbose=0)
            test_model.fit(X_train_s, y_train_s, eval_set=(X_test_s, y_test_s),
                           early_stopping_rounds=50, use_best_model=True)
            preds = test_model.predict(X_test_s)
            yhat = scaler_y.inverse_transform(preds.reshape(-1, 1))
            mae = metrics.mean_absolute_error(y_test, yhat)
            if mae < best_mae:
                best_mae = mae
                best_params = {'depth': d, 'lr': lr}

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

def get_rfr():
    params = {
        "n_estimators": 35,
        "max_features": 3,
        "random_state": 1
    }
    return RandomForestRegressor(**params)
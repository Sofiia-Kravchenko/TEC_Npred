import json
import os

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostRegressor
from sklearn import metrics
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler

from utils import prepare_step_data, prepare_direct_data, optuna_cbr_search


def train_cbr_direct_multistep(data, checkpoint_dir,checkpoint_unit_type, n_out, train_size):
    results_list = []
    metrics_list = []

    PARAMS_FILE = checkpoint_dir+checkpoint_unit_type+'best_catboost_params.json'

    if os.path.exists(PARAMS_FILE):
        with open(PARAMS_FILE, 'r') as f:
            best_params_storage = json.load(f)
    else:
        best_params_storage = {}

    data_with_lag, base_features, weights_train = prepare_step_data(data, train_size, n_out)

    for i in range(0, n_out, 1):
        step = str(i + 1)
        print(f"\n=== Step training {step} ===")

        X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, scaler_y, y_test = prepare_direct_data(data, data_with_lag, step, base_features, train_size)

        if step in best_params_storage:
            print(f"--- Step {step}: Using saved parameters: {best_params_storage[step]}")
            best_params = best_params_storage[step]
        else:
            print(f"--- Step {step}: Grid Search...")
            study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner())
            study.optimize(lambda trial: optuna_cbr_search(trial, X_train_scaled, y_train_scaled,
                                                   X_test_scaled, y_test_scaled, scaler_y),
                           n_trials=20)

            best_params = study.best_params

            best_params_storage[step] = best_params
            with open(PARAMS_FILE, 'w') as f:
                json.dump(best_params_storage, f, indent=4)

        model = CatBoostRegressor(iterations=2000,
                                  depth=best_params['depth'],
                                  learning_rate=best_params['lr'],
                                  loss_function='MAE',
                                  verbose=0)

        model.fit(X_train_scaled, y_train_scaled, eval_set=(X_test_scaled, y_test_scaled), sample_weight=weights_train,
                  early_stopping_rounds=250, use_best_model=True)

        yhat_scaled = model.predict(X_test_scaled)
        yhat = scaler_y.inverse_transform(yhat_scaled.reshape(-1, 1))
        yhat = np.maximum(yhat, 0)

        MAE_test = metrics.mean_absolute_error(y_test, yhat)
        MSE_test = metrics.mean_squared_error(y_test, yhat)
        WAPE_test = np.sum(np.abs(y_test - yhat)) / np.sum(np.abs(y_test)) * 100
        R2_test = metrics.r2_score(y_test, yhat)

        results_list.append(yhat)
        metrics_list.append([MAE_test,MSE_test,WAPE_test,R2_test])
    return np.hstack(results_list), np.hstack(metrics_list)


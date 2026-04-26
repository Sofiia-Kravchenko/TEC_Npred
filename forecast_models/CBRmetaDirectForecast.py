import json
import os

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn import metrics
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler

from utils import prepare_step_data, prepare_direct_data, prepare_meta_step_data, prepare_meta_direct_data


def train_cbr_meta_direct_multistep(data, checkpoint_dir, n_out, train_size, checkpoint_unit_type, results,
                                     best_window_model_name, test_idx, best_step_model, best_stat_model_name, calc_goal):
    results_list = []
    metrics_list = []

    PARAMS_FILE = checkpoint_dir+checkpoint_unit_type+'best_catboost_params.json'

    if os.path.exists(PARAMS_FILE):
        with open(PARAMS_FILE, 'r') as f:
            best_params_storage = json.load(f)
    else:
        best_params_storage = {}

    data_with_lag, base_features, weights_train = prepare_meta_step_data(data, train_size, n_out, results,
                                                                         best_window_model_name, best_stat_model_name,
                                                                         best_step_model,
                                                                         test_idx)

    learning_rates = [0.01, 0.05, 0.1, 0.2]
    depths = [4, 6, 8, 10]

    for i in range(0, n_out, 1):
        step = str(i + 1)
        print(f"\n=== Step training {step} ===")

        X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, scaler_y, y_test = prepare_meta_direct_data(data_with_lag, step, base_features, train_size, calc_goal)

        best_mae = float('inf')
        best_params = {'depth': 6, 'lr': 0.01}  # значения по умолчанию

        if step in best_params_storage:
            print(f"--- Step {step}: Using saved parameters: {best_params_storage[step]}")
            best_params = best_params_storage[step]
        else:
            print(f"--- Step {step}: Grid Search...")
            for d in depths:
                for lr in learning_rates:
                    test_model = CatBoostRegressor(iterations=500,
                                                   depth=d,
                                                   learning_rate=lr,
                                                   loss_function='MAE',
                                                   verbose=0)
                    test_model.fit(X_train_scaled, y_train_scaled, eval_set=(X_test_scaled, y_test_scaled),
                                   sample_weight=weights_train, early_stopping_rounds=250, use_best_model=True)
                    preds = test_model.predict(X_test_scaled)
                    yhat = scaler_y.inverse_transform(preds.reshape(-1, 1))
                    mae = metrics.mean_absolute_error(y_test, yhat)
                    if mae < best_mae:
                        best_mae = mae
                        best_params = {'depth': d, 'lr': lr}

                    best_params_storage[step] = {'depth': d, 'lr': lr}

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

        MAE_test = metrics.mean_absolute_error(y_test, yhat)
        MSE_test = metrics.mean_squared_error(y_test, yhat)
        WAPE_test = np.sum(np.abs(y_test - yhat)) / np.sum(np.abs(y_test)) * 100
        R2_test = metrics.r2_score(y_test, yhat)

        results_list.append(yhat)
        metrics_list.append([MAE_test,MSE_test,WAPE_test,R2_test])
    return np.hstack(results_list), np.hstack(metrics_list)


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
import tensorflow as tf

from forecast_models.CBRmetaDirectForecast import train_cbr_meta_direct_multistep
from forecast_models.LSTMmetaDirectForecast import train_lstm_meta_direct_multistep
from utils import prepare_stat_data, prepare_window_data, prepare_meta_data, prepare_meta_window_data
from print_results import print_stat_results, print_step_results, plot_compare_models_violations, \
    plot_single_model_violations
from sklearn.preprocessing import PolynomialFeatures
from forecast_models.CBRDirectForecast import train_cbr_direct_multistep
from forecast_models.LSTMDirectForecast import train_lstm_direct_multistep
from forecast_models.models import (
    get_catboost, get_linear, get_lstm,
    get_mlp, get_rfr
)
from tensorflow.keras.callbacks import ModelCheckpoint

tf.random.set_seed(42)

def calc_power_generation(data_path, test_start_index, checkpoint_dir, n_out, calc_goal, results, test_idx, best_window_model_name, best_stat_model_name, best_step_model, reports_dir, hierarchical_features, cliping_and_customLoss):
    if calc_goal == 'TEC':
        x_train_s, x_test_s, y_train_s, y_test_s, y_train, y_test, scaler_y, datas, test_idx,  y_train_s_combined, y_test_s_combined, y_test_combined = prepare_stat_data(data_path, test_start_index, cliping_and_customLoss)
        xw_train_s, xw_test_s, yw_train_s, yw_test_s, yw_test, scaler_yw, datasw, testw_idx,  yw_train_s_combined, yw_test_s_combined, yw_test_combined = prepare_window_data(data_path, test_start_index, cliping_and_customLoss)
    else:
        x_train_s, x_test_s, y_train_s, y_test_s, y_train, y_test, scaler_y, datas, test_idx,  y_train_s_combined, y_test_s_combined, y_test_combined = prepare_meta_data(results, data_path, test_idx, test_start_index, calc_goal, best_window_model_name, best_stat_model_name, best_step_model, hierarchical_features, cliping_and_customLoss)
        xw_train_s, xw_test_s, yw_train_s, yw_test_s, yw_test, scaler_yw, datasw, testw_idx,  yw_train_s_combined, yw_test_s_combined, yw_test_combined = prepare_meta_window_data(results, data_path, test_idx, test_start_index, calc_goal, best_window_model_name, best_stat_model_name, best_step_model, hierarchical_features, cliping_and_customLoss)
    results = {}
    constraints = {calc_goal + '_Available_Nmin': scaler_y.inverse_transform(y_test_s_combined[:, 2][:, None]),
                   calc_goal + '_Available_Nmax': scaler_y.inverse_transform(y_test_s_combined[:, 1][:, None])}

    # ---  LSTM ---
    print('LSTM_Stat_Model')
    x_train_lstm = x_train_s.reshape((x_train_s.shape[0], 1, x_train_s.shape[1]))
    x_test_lstm = x_test_s.reshape((x_test_s.shape[0], 1, x_test_s.shape[1]))

    for loss_type in ['custom', 'mae']:
        print(f"\n--- Processing LSTM with {loss_type.upper()} loss ---")

        model_lstm, early_stop_callback, current_epochs, checkpoint_filepath = get_lstm(
            (1, x_train_s.shape[1]), x_train_lstm, x_test_lstm, y_train_s, y_test_s, y_train, y_test,
            scaler_y, y_train_s_combined, y_test_s_combined, checkpoint_dir, calc_goal, loss_type=loss_type
        )

        model_checkpoint_callback = ModelCheckpoint(
            filepath=checkpoint_filepath, save_weights_only=False,
            monitor='val_loss', mode='min', save_best_only=True, verbose=0
        )

        model_lstm.fit(
            x_train_lstm, y_train_s_combined,
            epochs=current_epochs,
            batch_size=1024,
            validation_data=(x_test_lstm, y_test_s_combined),
            callbacks=[early_stop_callback, model_checkpoint_callback],
            verbose=0,
            shuffle=False
        )

        yhat_s = model_lstm.predict(x_test_lstm)
        yhat = scaler_y.inverse_transform(yhat_s)

        n_max = y_test_combined[:, 1][:, None]
        n_min = y_test_combined[:, 2][:, None]

        if loss_type == 'custom':
            yhat[yhat < n_min] = 0
            yhat = np.where(yhat > n_max, n_max, yhat)
            yhat = np.maximum(yhat, 0)

        over_max_mask = yhat > n_max
        count_over_max = np.sum(over_max_mask)
        mean_over_max = np.mean(yhat[over_max_mask] - n_max[over_max_mask]) if count_over_max > 0 else 0

        under_min_mask = yhat < n_min
        count_under_min = np.sum(under_min_mask)
        mean_under_min = np.mean(n_min[under_min_mask] - yhat[under_min_mask]) if count_under_min > 0 else 0

        print(f"\n[{loss_type.upper()} Loss] Bounds violations summary:")
        print(f"  -> Upper bound violations: {count_over_max} times (Mean excess: {mean_over_max:.4f})")
        print(f"  -> Lower bound violations: {count_under_min} times (Mean deficit: {mean_under_min:.4f})")
        # ------------------------------------------


        results[f'LSTM_{loss_type}'] = yhat

    # ---  Linear Regression ---
    print('LR_Stat_Model')
    model_lr = get_linear()
    model_lr.fit(x_train_s, y_train)
    yhat = model_lr.predict(x_test_s)
    if cliping_and_customLoss==0:
        yhat[yhat < y_test_combined[:, 2][:, None]] = 0
        n_max = y_test_combined[:, 1][:, None]
        yhat = np.where(yhat > n_max, n_max, yhat)
        yhat = np.maximum(yhat, 0)
    results['Linear'] = np.maximum(yhat, 0)

    # ---  poly Regression ---
    print('LRpoly_Stat_Model')
    poly = PolynomialFeatures(2, include_bias=False)
    x_train_poly = poly.fit_transform(x_train_s)
    x_test_poly = poly.transform(x_test_s)

    model_poly = get_linear()
    model_poly.fit(x_train_poly, y_train)
    if cliping_and_customLoss==0:
        yhat = model_poly.predict(x_test_poly)
        yhat[yhat < y_test_combined[:, 2][:, None]] = 0
        n_max = y_test_combined[:, 1][:, None]
        yhat = np.where(yhat > n_max, n_max, yhat)
        yhat = np.maximum(yhat, 0)
    results['Polynomial (d=2)'] = yhat

    # ---  Gradient Boosting ---
    print('CBR_Stat_Model')
    model_cb = get_catboost(x_train_s, x_test_s, y_train_s, y_test_s, y_train, y_test, scaler_y)
    model_cb.fit(x_train_s, y_train_s, eval_set=(x_test_s, y_test_s), early_stopping_rounds=250, use_best_model=True)
    yhat_s = model_cb.predict(x_test_s).reshape(-1, 1)
    yhat = scaler_y.inverse_transform(yhat_s)
    if cliping_and_customLoss==0:
        yhat[yhat < y_test_combined[:, 2][:, None]] = 0
        n_max = y_test_combined[:, 1][:, None]
        yhat = np.where(yhat > n_max, n_max, yhat)
        yhat = np.maximum(yhat, 0)
    results['Boosting'] = yhat

    # ---  MLP ---
    print('MLP_Stat_Model')
    checkpoint_filepath = checkpoint_dir + calc_goal + '_MLP.keras'
    params_filepath = checkpoint_dir + calc_goal + '_MLP_params.json'
    model_checkpoint_callback = ModelCheckpoint(filepath=checkpoint_filepath, save_weights_only=False,
                                                monitor='val_loss', mode='min', save_best_only=True, verbose=0)

    model_mlp, early_stop_callback, current_epochs = get_mlp((x_train_s.shape[1],), x_train_s, x_test_s, y_train_s, y_test_s, y_train, y_test, scaler_y, y_train_s_combined, y_test_s_combined, checkpoint_filepath, params_filepath)

    model_mlp.fit(x_train_s ,
                y_train_s_combined,
                epochs=current_epochs,
                batch_size=1024,
                validation_data=(x_test_s, y_test_s_combined),
                validation_batch_size=1024,
                callbacks=[early_stop_callback,model_checkpoint_callback],
                verbose=0,
                shuffle=False)

    yhat_s = model_mlp.predict(x_test_s)
    yhat = scaler_y.inverse_transform(yhat_s)
    if cliping_and_customLoss==0:
        yhat[yhat < y_test_combined[:, 2][:, None]] = 0
        n_max = y_test_combined[:, 1][:, None]
        yhat = np.where(yhat > n_max, n_max, yhat)
        yhat = np.maximum(yhat, 0)
    results['MLP'] = yhat

    # ---  Random Forest Regression---
    print('RFR_Stat_Model')
    model_rfr = get_rfr(x_train_s, x_test_s, y_train_s, y_test_s, y_train, y_test, scaler_y)
    model_rfr.fit(x_train_s, y_train_s)
    yhat_s = model_rfr.predict(x_test_s).reshape(-1, 1)
    yhat = scaler_y.inverse_transform(yhat_s)
    if cliping_and_customLoss==0:
        yhat[yhat < y_test_combined[:, 2][:, None]] = 0
        n_max = y_test_combined[:, 1][:, None]
        yhat = np.where(yhat > n_max, n_max, yhat)
        yhat = np.maximum(yhat, 0)
    results['RandomForest'] = yhat

    # ---  Gradient booster with window ---
    print('CBR_with_window_Stat_Model')
    model_cbw = get_catboost(xw_train_s, xw_test_s, yw_train_s, yw_test_s, y_train, y_test, scaler_yw)
    model_cbw.fit(xw_train_s, yw_train_s, eval_set=(xw_test_s, yw_test_s), early_stopping_rounds=250, use_best_model=True)
    yhat_s = model_cbw.predict(xw_test_s).reshape(-1, 1)
    yhat = scaler_y.inverse_transform(yhat_s)
    if cliping_and_customLoss==0:
        yhat[yhat < yw_test_combined[:, 2][:, None]] = 0
        n_max = yw_test_combined[:, 1][:, None]
        yhat = np.where(yhat > n_max, n_max, yhat)
        yhat = np.maximum(yhat, 0)
    results['Boosting_custom_with_window'] = yhat

    # ---  LSTM with window ---
    print('LSTM_with_window_Stat_Model Evaluation')
    xw_train_lstm = xw_train_s.reshape((xw_train_s.shape[0], 1, xw_train_s.shape[1]))
    xw_test_lstm = xw_test_s.reshape((xw_test_s.shape[0], 1, xw_test_s.shape[1]))
    for loss_type in ['custom', 'mae']:
        print(f"\n--- Processing LSTM with window ({loss_type.upper()} loss) ---")

        model_lstmrw, early_stop_callback, current_epochs, checkpoint_filepath = get_lstm(
            (1, xw_train_s.shape[1]), xw_train_lstm, xw_test_lstm, yw_train_s, yw_test_s, y_train, y_test,
            scaler_yw, yw_train_s_combined, yw_test_s_combined, checkpoint_dir, calc_goal + '_with_RW',
            loss_type=loss_type
        )

        model_checkpoint_callback = ModelCheckpoint(
            filepath=checkpoint_filepath, save_weights_only=False,
            monitor='val_loss', mode='min', save_best_only=True, verbose=0
        )

        model_lstmrw.fit(
            xw_train_lstm, yw_train_s_combined,
            epochs=current_epochs,
            batch_size=1024,
            validation_data=(xw_test_lstm, yw_test_s_combined),
            validation_batch_size=1024,
            callbacks=[early_stop_callback, model_checkpoint_callback],
            verbose=0,
            shuffle=False
        )

        yhat_s = model_lstmrw.predict(xw_test_lstm)
        yhat = scaler_yw.inverse_transform(yhat_s)

        n_max = yw_test_combined[:, 1][:, None]
        n_min = yw_test_combined[:, 2][:, None]

        if loss_type == 'custom':
            yhat[yhat < n_min] = 0
            yhat = np.where(yhat > n_max, n_max, yhat)
            yhat = np.maximum(yhat, 0)

        over_max_mask = yhat > n_max
        count_over_max = np.sum(over_max_mask)
        mean_over_max = np.mean(yhat[over_max_mask] - n_max[over_max_mask]) if count_over_max > 0 else 0

        under_min_mask = yhat < n_min
        count_under_min = np.sum(under_min_mask)
        mean_under_min = np.mean(n_min[under_min_mask] - yhat[under_min_mask]) if count_under_min > 0 else 0

        print(f"[{loss_type.upper()} Loss - Window] Bounds violations summary:")
        print(f"  -> Upper bound violations: {count_over_max} times (Mean excess: {mean_over_max:.4f})")
        print(f"  -> Lower bound violations: {count_under_min} times (Mean deficit: {mean_under_min:.4f})")

        results[f'LSTM_{loss_type}_with_window'] = yhat

    # ---  Random Forest Regression with window---
    print('RFR_with_window_Stat_Model')
    model_rfr = get_rfr(xw_train_s, xw_test_s, yw_train_s, yw_test_s, y_train, y_test, scaler_yw)
    model_rfr.fit(xw_train_s, yw_train_s)
    yhat_s = model_rfr.predict(xw_test_s).reshape(-1, 1)
    yhat = scaler_y.inverse_transform(yhat_s)
    results['RandomForest_mae_with_window'] = yhat
    if cliping_and_customLoss==0:
        yhat[yhat < yw_test_combined[:, 2][:, None]] = 0
        n_max = yw_test_combined[:, 1][:, None]
        yhat = np.where(yhat > n_max, n_max, yhat)
        yhat = np.maximum(yhat, 0)
    results['RandomForest_custom_with_window'] = yhat

    best_window_model_name, best_stat_model_name  = print_stat_results(reports_dir, results, y_test, calc_goal)

    plot_single_model_violations(
        y_true_real=y_test,
        y_pred=results[best_stat_model_name],
        n_max=y_test_combined[:, 1][:, None],
        n_min=y_test_combined[:, 2][:, None],
        model_name=best_stat_model_name,
    )

    plot_single_model_violations(
        y_true_real=yw_test,
        y_pred=results[best_window_model_name],
        n_max=yw_test_combined[:, 1][:, None],
        n_min=yw_test_combined[:, 2][:, None],
        model_name=best_window_model_name,
    )

    # ---  LSTM direct forecast---
    print('LSTM_direct_Stat_Model')
    if calc_goal == 'TEC':
        lstm_multi_results, lstm_multi_metrics = train_lstm_direct_multistep(data_path, checkpoint_dir, n_out, test_start_index, calc_goal, cliping_and_customLoss)
    else :
        lstm_multi_results, lstm_multi_metrics = train_lstm_meta_direct_multistep(data_path, checkpoint_dir, n_out, test_start_index, results,
                                     best_window_model_name, test_idx, best_step_model, best_stat_model_name, calc_goal, hierarchical_features ,cliping_and_customLoss)
    lstm_multi_results = np.array(lstm_multi_results).reshape(-1, 14)
    lstm_multi_metrics = np.array(lstm_multi_metrics).reshape(-1, 4)
    # --- Boosting with window direct forecast---
    print('CBR_direct_Stat_Model')
    if calc_goal == 'TEC':
        cbr_multi_results, cbr_multi_metrics = train_cbr_direct_multistep(data_path, checkpoint_dir, calc_goal, n_out, test_start_index, cliping_and_customLoss)
    else:
        cbr_multi_results, cbr_multi_metrics = train_cbr_meta_direct_multistep(data_path, checkpoint_dir, n_out,
                                                                         test_start_index, results,
                                     best_window_model_name, test_idx, best_step_model, best_stat_model_name, calc_goal, hierarchical_features, cliping_and_customLoss)
    cbr_multi_results = np.array(cbr_multi_results).reshape(-1, 14)
    cbr_multi_metrics = np.array(cbr_multi_metrics).reshape(-1, 4)

    best_step_model_name = print_step_results(lstm_multi_metrics, cbr_multi_metrics)
    if best_step_model_name == 'lstm': best_step_model = lstm_multi_results
    if best_step_model_name == 'CBR': best_step_model = cbr_multi_results

    return results, constraints, test_idx, best_window_model_name, best_stat_model_name, best_step_model
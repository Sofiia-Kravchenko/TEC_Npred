import os

import numpy as np
import pandas as pd
import tensorflow as tf

from forecast_models.CBRmetaDirectForecast import train_cbr_meta_direct_multistep
from forecast_models.LSTMmetaDirectForecast import train_lstm_meta_direct_multistep
from utils import prepare_stat_data, prepare_window_data, prepare_meta_data, prepare_meta_window_data
from print_results import print_stat_results, print_step_results
from sklearn.preprocessing import PolynomialFeatures
from forecast_models.CBRDirectForecast import train_cbr_direct_multistep
from forecast_models.LSTMDirectForecast import train_lstm_direct_multistep
from forecast_models.models import (
    get_catboost, get_linear, get_lstm,
    get_mlp, get_rfr, get_simple_mlp
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model

tf.random.set_seed(42)

def calc_power_generation(data_path, test_start_index, checkpoint_dir, checkpoint_unit_type, n_out, calc_goal, results, test_idx, best_window_model_name, best_stat_model_name, best_step_model):
    print("best_window_model_name:", best_window_model_name)
    print("best_stat_model_name:", best_stat_model_name)
    if calc_goal == 'TEC_N_Aver':
        X_train_s, X_test_s, y_train_s, y_test_s, y_train, y_test, scaler_y, datas, test_idx = prepare_stat_data(data_path, test_start_index=test_start_index)
        Xw_train_s, Xw_test_s, yw_train_s, yw_test_s, scaler_yw, datasw, testw_idx = prepare_window_data(data_path, test_start_index=test_start_index)
    else:
        X_train_s, X_test_s, y_train_s, y_test_s, y_train, y_test, scaler_y, datas, test_idx = prepare_meta_data(results, data_path, test_idx, test_start_index, calc_goal, best_window_model_name, best_stat_model_name, best_step_model)
        Xw_train_s, Xw_test_s, yw_train_s, yw_test_s, scaler_yw, datasw, testw_idx = prepare_meta_window_data(results, data_path, test_idx, test_start_index, calc_goal, best_window_model_name, best_stat_model_name, best_step_model)
    results = {}
    # ---  LSTM ---
    print('LSTM_Stat_Model')
    X_train_lstm = X_train_s.reshape((X_train_s.shape[0], 1, X_train_s.shape[1]))
    X_test_lstm = X_test_s.reshape((X_test_s.shape[0], 1, X_test_s.shape[1]))

    checkpoint_filepath = checkpoint_dir+checkpoint_unit_type+'_LSTM.keras'
    model_checkpoint_callback = ModelCheckpoint(filepath=checkpoint_filepath, save_weights_only=False,
                                                monitor='val_loss', mode='min', save_best_only=True, verbose=0)

    model_lstm = get_lstm((1, X_train_s.shape[1]))

    if os.path.exists(checkpoint_filepath):
        print("Loading model for further training...")
        model_lstm = load_model(checkpoint_filepath)
        model_lstm.optimizer.learning_rate.assign(1e-4)
        current_epochs = 50
        early_stop_callback = EarlyStopping(monitor='val_loss', patience=10, verbose=0, restore_best_weights=True,
                                            min_delta=0.0001)
    else:
        print("Checkpoint not found, starting training...")
        current_epochs = 1000
        early_stop_callback = EarlyStopping(monitor='val_loss', patience=100, verbose=1, restore_best_weights=True,
                                            min_delta=0.0001)

    model_lstm.fit(
        X_train_lstm, y_train_s,
        epochs=current_epochs,
        batch_size=30,
        validation_data=(X_test_lstm, y_test_s),
        callbacks=[early_stop_callback, model_checkpoint_callback],
        verbose=0,
        shuffle=False
    )

    pred_lstm_s = model_lstm.predict(X_test_lstm)
    y_pred_unscaled = scaler_y.inverse_transform(pred_lstm_s)
    results['LSTM'] = np.maximum(y_pred_unscaled, 0)

    # ---  Linear Regression ---
    print('LR_Stat_Model')
    model_lr = get_linear()
    model_lr.fit(X_train_s, y_train)
    y_pred_unscaled = model_lr.predict(X_test_s)
    results['Linear'] = np.maximum(y_pred_unscaled, 0)

    # ---  poly Regression ---
    print('LRpoly_Stat_Model')
    poly = PolynomialFeatures(2, include_bias=False)
    X_train_poly = poly.fit_transform(X_train_s)
    X_test_poly = poly.transform(X_test_s)

    model_poly = get_linear()
    model_poly.fit(X_train_poly, y_train)
    y_pred_unscaled = model_poly.predict(X_test_poly)
    results['Polynomial (d=2)'] = np.maximum(y_pred_unscaled, 0)

    # ---  Gradient Boosting ---
    print('CBR_Stat_Model')
    model_cb = get_catboost(X_train_s, X_test_s, y_train_s, y_test_s, y_train, y_test, scaler_y)
    model_cb.fit(X_train_s, y_train_s, eval_set=(X_test_s, y_test_s), early_stopping_rounds=250, use_best_model=True)
    y_test_pred = model_cb.predict(X_test_s).reshape(-1, 1)
    y_pred_unscaled = scaler_y.inverse_transform(y_test_pred)
    results['Boosting'] = np.maximum(y_pred_unscaled, 0)

    # ---  simple MLP ---
    print('sMLP_Stat_Model')
    model_smlp = get_simple_mlp((X_train_s.shape[1],))

    checkpoint_filepath = checkpoint_dir+checkpoint_unit_type+'_sMLP.keras'
    model_checkpoint_callback = ModelCheckpoint(filepath=checkpoint_filepath, save_weights_only=False,
                                                monitor='val_loss', mode='min', save_best_only=True, verbose=0)
    if os.path.exists(checkpoint_filepath):
        print("Loading model for further training...")
        model_smlp = load_model(checkpoint_filepath)
        model_smlp.optimizer.learning_rate.assign(1e-4)
        current_epochs = 50
        early_stop_callback = EarlyStopping(monitor='val_loss', patience=10, verbose=0, restore_best_weights=True,
                                            min_delta=0.0001)
    else:
        print("Checkpoint not found, starting training...")
        current_epochs = 1000
        early_stop_callback = EarlyStopping(monitor='val_loss', patience=100, verbose=1, restore_best_weights=True,
                                            min_delta=0.0001)

    model_smlp.fit(X_train_s ,
                y_train_s,
                epochs=current_epochs,
                batch_size=30,
                validation_data=(X_test_s, y_test_s),
                validation_batch_size=30,
                callbacks=[early_stop_callback,model_checkpoint_callback],
                verbose=0,
                shuffle=False)
    y_test_pred_scaled = model_smlp.predict(X_test_s)
    y_pred_unscaled = scaler_y.inverse_transform(y_test_pred_scaled)
    results['simple_MLP'] = np.maximum(y_pred_unscaled, 0)

    # ---  MLP ---
    print('MLP_Stat_Model')
    model_mlp = get_mlp((X_train_s.shape[1],))

    checkpoint_filepath = checkpoint_dir+checkpoint_unit_type+'_MLP.keras'
    model_checkpoint_callback = ModelCheckpoint(filepath=checkpoint_filepath, save_weights_only=False,
                                                monitor='val_loss', mode='min', save_best_only=True, verbose=0)
    if os.path.exists(checkpoint_filepath):
        print("Loading model for further training...")
        model_mlp = load_model(checkpoint_filepath)
        model_mlp.optimizer.learning_rate.assign(1e-4)
        current_epochs = 50
        early_stop_callback = EarlyStopping(monitor='val_loss', patience=10, verbose=0, restore_best_weights=True,
                                            min_delta=0.0001)
    else:
        print("Checkpoint not found, starting training...")
        current_epochs = 1000
        early_stop_callback = EarlyStopping(monitor='val_loss', patience=100, verbose=1, restore_best_weights=True,
                                            min_delta=0.0001)

    model_mlp.fit(X_train_s ,
                y_train_s,
                epochs=current_epochs,
                batch_size=30,
                validation_data=(X_test_s, y_test_s),
                validation_batch_size=30,
                callbacks=[early_stop_callback,model_checkpoint_callback],
                verbose=0,
                shuffle=False)

    y_test_pred_scaled = model_mlp.predict(X_test_s)
    y_pred_unscaled = scaler_y.inverse_transform(y_test_pred_scaled)
    results['MLP'] = np.maximum(y_pred_unscaled, 0)

    # ---  Random Forest Regression---
    print('RFR_Stat_Model')
    model_rfr = get_rfr()
    model_rfr.fit(X_train_s, y_train_s)
    y_test_pred = model_rfr.predict(X_test_s).reshape(-1, 1)
    y_pred_unscaled = scaler_y.inverse_transform(y_test_pred)
    results['RandomForest'] = np.maximum(y_pred_unscaled, 0)

    # ---  Gradient booster with window ---
    print('CBR_with_window_Stat_Model')
    model_cbw = get_catboost(X_train_s, X_test_s, y_train_s, y_test_s, y_train, y_test, scaler_y)
    model_cbw.fit(Xw_train_s, yw_train_s, eval_set=(Xw_test_s, yw_test_s), early_stopping_rounds=250, use_best_model=True)
    yw_test_pred = model_cbw.predict(Xw_test_s).reshape(-1, 1)
    y_pred_unscaled = scaler_yw.inverse_transform(yw_test_pred)
    results['Boosting_with_window'] = np.maximum(y_pred_unscaled, 0)

    # ---  LSTM with window ---
    print('LSTM_with_window_Stat_Model')
    Xw_train_lstm = Xw_train_s.reshape((Xw_train_s.shape[0], 1, Xw_train_s.shape[1]))
    Xw_test_lstm = Xw_test_s.reshape((Xw_test_s.shape[0], 1, Xw_test_s.shape[1]))

    checkpoint_filepath = checkpoint_dir+checkpoint_unit_type+'_LSTM_with_RW.keras'
    model_checkpoint_callback = ModelCheckpoint(filepath=checkpoint_filepath, save_weights_only=False,
                                                monitor='val_loss', mode='min', save_best_only=True, verbose=0)

    model_lstmrw = get_lstm((1, Xw_train_s.shape[1]))

    if os.path.exists(checkpoint_filepath):
        print("Loading model for further training...")
        model_lstmrw = load_model(checkpoint_filepath)
        model_lstmrw.optimizer.learning_rate.assign(1e-4)
        current_epochs = 50
        early_stop_callback = EarlyStopping(monitor='val_loss', patience=10, verbose=0, restore_best_weights=True,
                                            min_delta=0.0001)
    else:
        print("Checkpoint not found, starting training...")
        current_epochs = 1000
        early_stop_callback = EarlyStopping(monitor='val_loss', patience=100, verbose=1, restore_best_weights=True,
                                            min_delta=0.0001)
    model_lstmrw.fit(Xw_train_lstm, yw_train_s,
                epochs=current_epochs,
                batch_size=30,
                validation_data=(Xw_test_lstm, yw_test_s),
                validation_batch_size=30,
                callbacks=[early_stop_callback,model_checkpoint_callback],
                verbose=0,
                shuffle=False)

    pred_lstm_s = model_lstmrw.predict(Xw_test_lstm)
    y_pred_unscaled = scaler_yw.inverse_transform(pred_lstm_s)
    results['LSTM_with_window'] = np.maximum(y_pred_unscaled, 0)

    # ---  simple MLP with window---
    print('sMLP_Stat_Model')
    model_smlpw = get_simple_mlp((Xw_train_s.shape[1],))

    checkpoint_filepath = checkpoint_dir+checkpoint_unit_type+'_sMLP_with_RW.keras'
    model_checkpoint_callback = ModelCheckpoint(filepath=checkpoint_filepath, save_weights_only=False,
                                                monitor='val_loss', mode='min', save_best_only=True, verbose=0)
    if os.path.exists(checkpoint_filepath):
        print("Loading model for further training...")
        model_smlpw = load_model(checkpoint_filepath)
        model_smlpw.optimizer.learning_rate.assign(1e-4)
        current_epochs = 50
        early_stop_callback = EarlyStopping(monitor='val_loss', patience=10, verbose=0, restore_best_weights=True,
                                            min_delta=0.0001)
    else:
        print("Checkpoint not found, starting training...")
        current_epochs = 1000
        early_stop_callback = EarlyStopping(monitor='val_loss', patience=100, verbose=1, restore_best_weights=True,
                                            min_delta=0.0001)

    model_smlpw.fit(Xw_train_s,
                yw_train_s,
                epochs=current_epochs,
                batch_size=30,
                validation_data=(Xw_test_s, yw_test_s),
                validation_batch_size=30,
                callbacks=[early_stop_callback,model_checkpoint_callback],
                verbose=0,
                shuffle=False)
    yw_test_pred_scaled = model_smlpw.predict(Xw_test_s)
    y_pred_unscaled = scaler_yw.inverse_transform(yw_test_pred_scaled)
    results['simple_MLP_with_window'] = np.maximum(y_pred_unscaled, 0)

    # ---  MLP with window---
    print('MLP_with_window_Stat_Model')
    model_mlpw = get_mlp((Xw_train_s.shape[1],))

    checkpoint_filepath = checkpoint_dir + checkpoint_unit_type + '_MLP_with_RW.keras'
    model_checkpoint_callback = ModelCheckpoint(filepath=checkpoint_filepath, save_weights_only=False,
                                                monitor='val_loss', mode='min', save_best_only=True, verbose=0)
    if os.path.exists(checkpoint_filepath):
        print("Loading model for further training...")
        model_mlpw = load_model(checkpoint_filepath)
        model_mlpw.optimizer.learning_rate.assign(1e-4)
        current_epochs = 50
        early_stop_callback = EarlyStopping(monitor='val_loss', patience=10, verbose=0, restore_best_weights=True,
                                            min_delta=0.0001)
    else:
        print("Checkpoint not found, starting training...")
        current_epochs = 1000
        early_stop_callback = EarlyStopping(monitor='val_loss', patience=100, verbose=1, restore_best_weights=True,
                                            min_delta=0.0001)

    model_mlpw.fit(Xw_train_s,
                  yw_train_s,
                  epochs=current_epochs,
                  batch_size=30,
                  validation_data=(Xw_test_s, yw_test_s),
                  validation_batch_size=30,
                  callbacks=[early_stop_callback, model_checkpoint_callback],
                  verbose=0,
                  shuffle=False)

    yw_test_pred_scaled = model_mlpw.predict(Xw_test_s)
    y_pred_unscaled = scaler_yw.inverse_transform(yw_test_pred_scaled)
    results['MLP_with_window'] = np.maximum(y_pred_unscaled, 0)

    # ---  LSTM direct forecast---
    print('LSTM_direct_Stat_Model')
    if calc_goal == 'TEC_N_Aver':
        lstm_multi_results, lstm_multi_metrics = train_lstm_direct_multistep(data_path, checkpoint_dir, n_out, test_start_index, checkpoint_unit_type)
    else :
        lstm_multi_results, lstm_multi_metrics = train_lstm_meta_direct_multistep(data_path, checkpoint_dir, n_out, test_start_index, checkpoint_unit_type, results,
                                     best_window_model_name, test_idx, best_step_model, best_stat_model_name, calc_goal)
    lstm_multi_results = np.array(lstm_multi_results).reshape(-1, 14)
    lstm_multi_metrics = np.array(lstm_multi_metrics).reshape(-1, 4)
    # --- Boosting with window direct forecast---
    print('CBR_direct_Stat_Model')
    if calc_goal == 'TEC_N_Aver':
        cbr_multi_results, cbr_multi_metrics = train_cbr_direct_multistep(data_path, checkpoint_dir,checkpoint_unit_type, n_out, test_start_index)
    else:
        cbr_multi_results, cbr_multi_metrics = train_cbr_meta_direct_multistep(data_path, checkpoint_dir, n_out,
                                                                         test_start_index, checkpoint_unit_type, results,
                                     best_window_model_name, test_idx, best_step_model, best_stat_model_name, calc_goal)
    cbr_multi_results = np.array(cbr_multi_results).reshape(-1, 14)
    cbr_multi_metrics = np.array(cbr_multi_metrics).reshape(-1, 4)

    print(results)
    best_window_model_name, best_stat_model_name  = print_stat_results(results, y_test, calc_goal)
    best_step_model_name = print_step_results(lstm_multi_metrics, cbr_multi_metrics)
    if best_step_model_name == 'lstm': best_step_model = lstm_multi_results
    if best_step_model_name == 'CBR': best_step_model = cbr_multi_results

    return results, test_idx, best_window_model_name, best_stat_model_name, best_step_model
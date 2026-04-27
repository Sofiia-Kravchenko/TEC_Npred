import os
import tensorflow as tf

import numpy as np
import pandas as pd
import tensorflow.keras.backend as K
from keras.models import load_model
from keras import Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Dense, LSTM
from pandas import DataFrame, concat
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics

from utils import prepare_meta_step_data, prepare_meta_direct_data
tf.random.set_seed(42)

def train_lstm_meta_direct_multistep(data, checkpoint_dir, n_out, train_size, checkpoint_unit_type, results, best_window_model_name, test_idx, best_step_model, best_stat_model_name, calc_goal):
    results_list = []
    metrics_list = []
    checkpoint_dir = checkpoint_dir + 'multistep/'
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_base = checkpoint_dir + checkpoint_unit_type + '_LSTM_LAG_model_step_'
    data_with_lag, base_features, weights_train = prepare_meta_step_data(data, train_size, n_out, results, best_window_model_name, best_stat_model_name, best_step_model,
     test_idx)

    def physical_constraints_loss(y_true, y_pred):
        base_error = K.abs(y_true - y_pred)
        penalty_gate = K.maximum(0.0, threshold_scaled - y_pred)
        return K.mean(base_error + (penalty_gate * 20.0))
    def build_model(input_shape):
        model = Sequential()
        model.add(LSTM(100, input_shape=input_shape))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(1))
        model.compile(loss=physical_constraints_loss, optimizer='adam', metrics=['mse', 'mape', 'r2_score'])
        return model
    for i in range(0, n_out, 1):
        step = i + 1
        print(f"\n=== Step training {step} ===")

        X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, scaler_y, y_test = prepare_meta_direct_data(data, data_with_lag, step, base_features, train_size, calc_goal)

        X_train_scaled = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
        X_test_scaled = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

        threshold_scaled = scaler_y.transform(np.array([[90.0]]))[0, 0]
        model = build_model((X_train_scaled.shape[1], X_train_scaled.shape[2]))

        current_checkpoint = checkpoint_base + str(i) + '.keras'

        if os.path.exists(current_checkpoint):
            print(f"--- Step {step}: File found. Loading and RESUMING training... ---")
            model = load_model(current_checkpoint)
            model.optimizer.learning_rate.assign(1e-4)
            current_epochs = 50
            early_stop_callback = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        else:
            print(f"--- Step {step}: File not found. Starting NEW training... ---")
            current_epochs = 1000
            early_stop_callback = EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True)

            if i > 0:
                prev_model_path = checkpoint_base + str(i - 1) + '.keras'
                if os.path.exists(prev_model_path):
                    model.load_weights(prev_model_path)
                    print(f"Weights initialized from step {i}")

        checkpoint_callback = ModelCheckpoint(filepath=current_checkpoint, save_best_only=True, monitor='val_loss')


        model.fit(X_train_scaled, y_train_scaled,
                  epochs=current_epochs,
                  batch_size=32,
                  validation_data=(X_test_scaled, y_test_scaled),
                  callbacks=[early_stop_callback, checkpoint_callback],
                  verbose=0,
                  shuffle=False)

        yhat_scaled = model.predict(X_test_scaled)
        yhat = scaler_y.inverse_transform(yhat_scaled)
        yhat = np.maximum(yhat, 0)

        MAE_test = metrics.mean_absolute_error(y_test, yhat)
        MSE_test = metrics.mean_squared_error(y_test, yhat)
        WAPE_test = np.sum(np.abs(y_test - yhat)) / np.sum(np.abs(y_test)) * 100
        R2_test = metrics.r2_score(y_test, yhat)
        results_list.append(yhat)
        metrics_list.append([MAE_test,MSE_test,WAPE_test,R2_test])

    return np.hstack(results_list), np.hstack(metrics_list)
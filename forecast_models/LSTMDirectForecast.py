import gc
from tensorflow.keras import backend as keras_backend
import os
import optuna
import json
import tensorflow as tf

import numpy as np
from keras.models import load_model
from keras import Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Dense, LSTM
from keras.src.optimizers import Adam
from sklearn import metrics

from utils import prepare_step_data, prepare_direct_data
tf.random.set_seed(42)

def train_lstm_direct_multistep(data, checkpoint_dir, n_out, train_size, checkpoint_unit_type):

    results_list = []
    metrics_list = []

    checkpoint_dir = os.path.join(checkpoint_dir, 'multistep/')
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_base = os.path.join(checkpoint_dir, f"{checkpoint_unit_type}_LSTM_LAG_model_step_")
    params_path = os.path.join(checkpoint_dir, f"{checkpoint_unit_type}_best_params.json")

    if os.path.exists(params_path):
        with open(params_path, 'r') as f:
            all_steps_params = json.load(f)
    else:
        all_steps_params = {}

    data_with_lag, base_features, weights_train = prepare_step_data(data, train_size, n_out)

    for i in range(n_out):
        step = i + 1
        step_key = str(i)
        current_checkpoint = f"{checkpoint_base}{i}.keras"
        print(f"\n=== Step training {step} ===")

        X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, scaler_y, y_test = prepare_direct_data(data, data_with_lag, step, base_features, train_size)

        X_train_scaled = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
        X_test_scaled = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

        if os.path.exists(current_checkpoint):
            print(f"--- Step {step}: Checkpoint found. Loading... ---")
            model = load_model(current_checkpoint)
            model.optimizer.learning_rate.assign(1e-4)
            current_epochs, patience = 50, 10
        else:
            print(f"--- Step {step}: No checkpoint.")
            current_epochs, patience = 1000, 100

            if step_key not in all_steps_params:
                print(f"--- Step {step}: Tuning hyperparameters with Optuna... ---")

                def objective(trial):
                    n_lstm = trial.suggest_int('n_lstm', 50, 200)
                    n_dense = trial.suggest_int('n_dense', 20, 100)
                    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
                    m = Sequential([
                        LSTM(n_lstm, input_shape=(X_train_scaled.shape[1], X_train_scaled.shape[2])),
                        Dense(n_dense, activation='relu'),
                        Dense(1)
                    ])
                    m.compile(loss='mae', optimizer=Adam(learning_rate=lr))
                    m.fit(X_train_scaled, y_train_scaled, epochs=5, batch_size=32, verbose=0)
                    return m.evaluate(X_test_scaled, y_test_scaled, verbose=0)

                study = optuna.create_study(direction='minimize')
                study.optimize(objective, n_trials=20)
                all_steps_params[step_key] = study.best_params
                with open(params_path, 'w') as f:
                    json.dump(all_steps_params, f)

            params = all_steps_params[step_key]
            model = Sequential([
                LSTM(params['n_lstm'], input_shape=(X_train_scaled.shape[1], X_train_scaled.shape[2])),
                Dense(params['n_dense'], activation='relu'),
                Dense(1)
            ])
            model.compile(loss='mae', optimizer=Adam(learning_rate=params['lr']), metrics=['mse'])

            if i > 0:
                prev_path = f"{checkpoint_base}{i - 1}.keras"
                if os.path.exists(prev_path):
                    try:
                        model.load_weights(prev_path, by_name=True, skip_mismatch=True)
                        print(f"--- Step {step}: Weights initialized from step {i} ---")
                    except:
                        pass

        checkpoint_callback = ModelCheckpoint(filepath=current_checkpoint, save_best_only=True, monitor='val_loss')
        early_stop_callback = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)

        model.fit(X_train_scaled, y_train_scaled,
                  epochs=current_epochs,
                  batch_size=32,
                  validation_data=(X_test_scaled, y_test_scaled),
                  callbacks=[early_stop_callback, checkpoint_callback],
                  verbose=0, shuffle=False)

        yhat = scaler_y.inverse_transform(model.predict(X_test_scaled))
        yhat = np.maximum(yhat, 0)

        results_list.append(yhat)
        metrics_list.append([
            metrics.mean_absolute_error(y_test, yhat),
            metrics.mean_squared_error(y_test, yhat),
            np.sum(np.abs(y_test - yhat)) / np.sum(np.abs(y_test)) * 100,
            metrics.r2_score(y_test, yhat)
        ])

        del model
        keras_backend.clear_session()
        gc.collect()

    return np.hstack(results_list), np.array(metrics_list)
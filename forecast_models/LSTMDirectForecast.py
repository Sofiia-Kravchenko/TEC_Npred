import os

import numpy as np
import pandas as pd
from keras.models import load_model
from keras import Sequential
from keras.src.callbacks import ModelCheckpoint, EarlyStopping
from keras.src.layers import Dense, LSTM
from pandas import DataFrame, concat
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics


def train_lstm_direct_multistep(data, checkpoint_dir, n_out, train_size):
    results_list = []
    metrics_list = []
    checkpoint_dir = checkpoint_dir + 'multistep'
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_base = checkpoint_dir + '/Npred_LSTM_LAG_model_step_'
    data = pd.read_csv(data, delimiter=';', parse_dates=['Date'], dayfirst=True)

    data['Month'] = data['Date'].dt.month
    data['Year'] = data['Date'].dt.year
    data['Month_sin'] = np.sin(2 * np.pi * data['Month'] / 12)
    data['Month_cos'] = np.cos(2 * np.pi * data['Month'] / 12)

    data['B1_N_Aver'] = data['B1_N'] / 24
    data['B2_N_Aver'] = data['B2_N'] / 24
    data['B3_N_Aver'] = data['B3_N'] / 24
    data['B4_GT41_N_Aver'] = data['B4_GT41_N'] / 24
    data['B4_GT42_N_Aver'] = data['B4_GT42_N'] / 24
    data['B4_PT40_N_Aver'] = data['B4_N_Aver'] - data['B4_GT41_N_Aver'] - data['B4_GT42_N_Aver']

    data['B1_inWork'] = np.where(data['B1_N_Aver'] > 125, 1, 0)
    data['B2_inWork'] = np.where(data['B2_N_Aver'] > 125, 1, 0)
    data['B3_inWork'] = np.where(data['B3_N_Aver'] > 125, 1, 0)
    data['B4_GT41_inWork'] = np.where(data['B4_GT41_N_Aver'] > 50, 1, 0)
    data['B4_GT42_inWork'] = np.where(data['B4_GT42_N_Aver'] > 50, 1, 0)
    data['B4_inWork'] = np.where(data['B4_N_Aver'] > 50, 1, 0)

    data['B1_Available_N'] = data['B1_inWork'] * 250
    data['B2_Available_N'] = data['B2_inWork'] * 250
    data['B3_Available_N'] = data['B3_inWork'] * 250
    data['B4_Available_N'] = (data['B4_GT41_inWork'] * 150 + data['B4_GT42_inWork'] * 150 + 75 *
                              (data['B4_GT41_inWork'] + data['B4_GT42_inWork']))

    data['T_Prev'] = data['T'].shift(1)
    data['Month_sin_Prev'] = data['Month_sin'].shift(1)
    data['Month_cos_Prev'] = data['Month_cos'].shift(1)
    data['Year_Prev'] = data['Year'].shift(1)
    data['B4_GT41_N_Aver_Prev'] = data['B4_GT41_N_Aver'].shift(1)
    data['B4_GT42_N_Aver_Prev'] = data['B4_GT42_N_Aver'].shift(1)
    data['B4_PT40_N_Aver_Prev'] = data['B4_PT40_N_Aver'].shift(1)
    data['B1_N_Aver_Prev'] = data['B1_N_Aver'].shift(1)
    data['B2_N_Aver_Prev'] = data['B2_N_Aver'].shift(1)
    data['B3_N_Aver_Prev'] = data['B3_N_Aver'].shift(1)
    data['B4_N_Aver_Prev'] = data['B4_N_Aver'].shift(1)
    data['B1_Available_N_Prev'] = data['B1_Available_N'].shift(1)
    data['B2_Available_N_Prev'] = data['B2_Available_N'].shift(1)
    data['B3_Available_N_Prev'] = data['B3_Available_N'].shift(1)
    data['B4_Available_N_Prev'] = data['B4_Available_N'].shift(1)
    data['TEC_N_Aver_Prev'] = data['TEC_N_Aver'].shift(1)

    data = data[400:]
    data.set_index('Date', inplace=True)

    base_features = ['T', 'Month_sin', 'Month_cos', 'Year',
                'B1_inWork', 'B2_inWork', 'B3_inWork' ,'B4_GT41_inWork', 'B4_GT42_inWork', 'B4_inWork',
                'B1_Available_N', 'B2_Available_N', 'B3_Available_N', 'B4_Available_N',
                'T_Prev', 'Month_sin_Prev', 'Month_cos_Prev', 'Year_Prev',
                'B4_GT41_N_Aver_Prev', 'B4_GT42_N_Aver_Prev' ,'B4_PT40_N_Aver_Prev', 'TEC_N_Aver_Prev',
                'B1_N_Aver_Prev', 'B2_N_Aver_Prev' ,'B3_N_Aver_Prev', 'B4_N_Aver_Prev',
                'B1_Available_N_Prev', 'B2_Available_N_Prev' ,'B3_Available_N_Prev', 'B4_Available_N_Prev']

    df = DataFrame(data)
    df['T_rolling_3'] = df['T'].rolling(window=3).mean()

    cols, names ,agg = list(), list(), list()
    for i in range(1, n_out + 1):
        cols.append(df[['T']].shift(-i))
        cols.append(df[['Month_sin']].shift(-i))
        cols.append(df[['Month_cos']].shift(-i))
        cols.append(df[['Year']].shift(-i))
        cols.append(df[['TEC_N_Aver']].shift(-i))
        cols.append(df[['B1_inWork']].shift(-i))
        cols.append(df[['B2_inWork']].shift(-i))
        cols.append(df[['B3_inWork']].shift(-i))
        cols.append(df[['B4_GT41_inWork']].shift(-i))
        cols.append(df[['B4_GT42_inWork']].shift(-i))
        cols.append(df[['B4_inWork']].shift(-i))
        cols.append(df[['B1_Available_N']].shift(-i))
        cols.append(df[['B2_Available_N']].shift(-i))
        cols.append(df[['B3_Available_N']].shift(-i))
        cols.append(df[['B4_Available_N']].shift(-i))
        names += [f'T_lag{i}', f'Month_sin_lag{i}', f'Month_cos_lag{i}', f'Year_lag{i}',
                  f'TEC_N_Aver_lag{i}', f'B1_inWork_lag{i}', f'B2_inWork_lag{i}', f'B3_inWork_lag{i}',
                  f'B4_GT41_inWork_lag{i}', f'B4_GT42_inWork_lag{i}', f'B4_inWork_lag{i}',
                  f'B1_Available_N_lag{i}', f'B2_Available_N_lag{i}', f'B3_Available_N_lag{i}', f'B4_Available_N_lag{i}']

    agg = concat(cols, axis=1)
    agg.columns = names
    agg.dropna(inplace=True)

    data_with_lag = pd.concat([data, agg], axis=1)
    data_with_lag.dropna(inplace=True)
    def build_model(input_shape):
        model = Sequential()
        model.add(LSTM(100, input_shape=input_shape))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(1))
        model.compile(loss='mae', optimizer='adam', metrics=['mse', 'mape', 'r2_score'])
        return model

    for i in range(0, n_out, 1):
        step = i + 1
        print(f"\n=== Step training {step} ===")

        current_features = ['T', 'Month_sin', 'Month_cos', 'Year', 'B1_inWork', 'B2_inWork',
                          'B3_inWork', 'B4_GT41_inWork', 'B4_GT42_inWork', 'B4_inWork',
                          'B1_Available_N', 'B2_Available_N', 'B3_Available_N', 'B4_Available_N']
        current_features = ['T', 'Month_sin', 'Month_cos', 'Year', 'B1_inWork', 'B2_inWork',
                          'B3_inWork', 'B4_GT41_inWork', 'B4_GT42_inWork', 'B4_inWork',
                          'B1_Available_N', 'B2_Available_N', 'B3_Available_N', 'B4_Available_N']

        past_features  = ['T_Prev', 'Month_sin_Prev', 'Month_cos_Prev', 'Year_Prev', 'B1_inWork_Prev',
                         'B2_inWork_Prev', 'B3_inWork_Prev', 'B4_GT41_inWork_Prev', 'B4_GT42_inWork_Prev',
                         'B4_inWork_Prev', 'B1_Available_N_Prev', 'B2_Available_N_Prev',
                         'B3_Available_N_Prev', 'B4_Available_N_Prev']

        future_features = [
            f'T_lag{step}',
            f'Month_sin_lag{step}',
            f'Month_cos_lag{step}',
            f'Year_lag{step}',
            f'B1_inWork_lag{step}',
            f'B2_inWork_lag{step}',
            f'B3_inWork_lag{step}',
            f'B4_GT41_inWork_lag{step}',
            f'B4_GT42_inWork_lag{step}',
            f'B4_inWork_lag{step}',
            f'B1_Available_N_lag{step}',
            f'B2_Available_N_lag{step}',
            f'B3_Available_N_lag{step}',
            f'B4_Available_N_lag{step}'
        ]
        future_features = [
                f'T_lag{step}',
                f'B1_Available_N_lag{step}',
                f'B2_Available_N_lag{step}',
                f'B3_Available_N_lag{step}',
                f'B4_Available_N_lag{step}'
            ]
        x_step_today  = data_with_lag[current_features].values
        x_step_today  = np.column_stack([x_step_today, data_with_lag['TEC_N_Aver_Prev'].values])

        x_step_future  = data_with_lag[future_features].values
        x_step_future  = np.column_stack([x_step_future, np.zeros(len(x_step_future))])

        xt_train, xt_test = x_step_today[:train_size], x_step_today[train_size:]
        xf_train, xf_test = x_step_future[:train_size], x_step_future[train_size:]

        scaler = MinMaxScaler()
        scaler.fit(np.vstack([xt_train, xf_train]))

        xt_train_s = scaler.transform(xt_train)
        xf_train_s = scaler.transform(xf_train)
        xt_test_s = scaler.transform(xt_test)
        xf_test_s = scaler.transform(xf_test)

        X_train_3d = np.stack([xt_train_s, xf_train_s], axis=1)
        X_test_3d = np.stack([xt_test_s, xf_test_s], axis=1)


        y = data_with_lag.loc[:, [f'TEC_N_Aver_lag{step}']].values
        y_train, y_test = y[:train_size], y[train_size:]
        scaler_y = MinMaxScaler()
        y_train_scaled = scaler_y.fit_transform(y_train)
        y_test_scaled = scaler_y.transform(y_test)

        model = build_model((X_train_3d.shape[1], X_train_3d.shape[2]))

        current_checkpoint = checkpoint_base + str(i) + '.keras'

        if os.path.exists(current_checkpoint):
            print(f"--- Step {step}: File found. Loading and RESUMING training... ---")
            model = load_model(current_checkpoint)
            model.optimizer.learning_rate.assign(1e-4)
            current_epochs = 50
        else:
            print(f"--- Step {step}: File not found. Starting NEW training... ---")
            current_epochs = 1000

            if i > 0:
                prev_model_path = checkpoint_base + str(i - 1) + '.keras'
                if os.path.exists(prev_model_path):
                    model.load_weights(prev_model_path)
                    print(f"Weights initialized from step {i}")

        checkpoint_callback = ModelCheckpoint(filepath=current_checkpoint, save_best_only=True, monitor='val_loss')
        early_stop_callback = EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True)

        model.fit(X_train_3d , y_train_scaled,
                  epochs=current_epochs,
                  batch_size=32,
                  validation_data=(X_test_3d , y_test_scaled),
                  callbacks=[early_stop_callback, checkpoint_callback],
                  verbose=2,
                  shuffle=False)

        yhat_scaled = model.predict(X_test_3d)
        yhat = scaler_y.inverse_transform(yhat_scaled)
        MAE_test = metrics.mean_absolute_error(y_test, yhat)
        MSE_test = metrics.mean_squared_error(y_test, yhat)
        MAPE_test = metrics.mean_absolute_percentage_error(y_test, yhat)*100
        R2_test = metrics.r2_score(y_test, yhat)
        results_list.append(yhat)
        metrics_list.append([MAE_test,MSE_test,MAPE_test,R2_test])
    return np.hstack(metrics_list)
import numpy as np
import pandas as pd
from keras import Sequential
from keras.src.callbacks import ModelCheckpoint, EarlyStopping
from keras.src.layers import LSTM, Dense
from matplotlib import pyplot as plt
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn import datasets, linear_model, metrics, model_selection, __all__, ensemble

import warnings

warnings.filterwarnings("ignore")

data = pd.read_csv(r'C:\Users\Andrey\PyCharmMiscProject\data\TEC22_Data_N.csv', delimiter=';', parse_dates=['Date'], dayfirst=True)
data['Month'] = data['Date'].dt.month
data['Year'] = data['Date'].dt.year
data['Month_sin'] = np.sin(2 * np.pi * data['Month'] / 12)
data['Month_cos'] = np.cos(2 * np.pi * data['Month'] / 12)
data['T_rolling_3'] = data['T'].rolling(window=3).mean()
data = data[400:]


data.set_index('Date', inplace=True)

n_out = 14

base_features = ['Month_sin', 'Month_cos', 'Year', 'T',
                 'B1_inWork', 'B2_inWork', 'B3_inWork',
                 'B4_GT41_inWork', 'B4_GT42_inWork', 'B4_inWork']

df = DataFrame(data)
cols, names ,agg = list(), list(), list()
for i in range(1, n_out + 1):
    cols.append(df[['T']].shift(-i))
    cols.append(df[['TEC_N_Aver']].shift(-i))
    cols.append(df[['B1_inWork']].shift(-i))
    cols.append(df[['B2_inWork']].shift(-i))
    cols.append(df[['B3_inWork']].shift(-i))
    cols.append(df[['B4_GT41_inWork']].shift(-i))
    cols.append(df[['B4_GT42_inWork']].shift(-i))
    cols.append(df[['B4_inWork']].shift(-i))
    cols.append(df[['T_rolling_3']].shift(-i))
    names += [('T_lag%d' % i)]
    names += [('N_lag%d' % i)]
    names += [('B1_inWork_lag%d' % i)]
    names += [('B2_inWork_lag%d' % i)]
    names += [('B3_inWork_lag%d' % i)]
    names += [('B4_GT41_inWork_lag%d' % i)]
    names += [('B4_GT42_inWork_lag%d' % i)]
    names += [('B4_inWork_lag%d' % i)]
    names += [f'T_mean{i}']

agg = concat(cols, axis=1)
agg.columns = names
agg.dropna(inplace=True)

data_with_lag = pd.concat([data, agg], axis=1)
data_with_lag.dropna(inplace=True)

metrics_list = []
results_list = np.array(())
for i in range(0, n_out, 1):
    step = i + 1
    print(f'\n=== Step {step} Prediction ===')

    step_features = base_features + [
        'TEC_N_Aver',
        'B1_inWork', 'B2_inWork', 'B3_inWork',
        'B4_GT41_inWork', 'B4_GT42_inWork', 'B4_inWork',
        f'T_lag{step}',
        f'B1_inWork_lag{step}',
        f'B2_inWork_lag{step}',
        f'B3_inWork_lag{step}',
        f'B4_GT41_inWork_lag{step}',
        f'B4_GT42_inWork_lag{step}',
        f'B4_inWork_lag{step}',
        #f'T_mean{step}'
    ]

    x = data_with_lag[step_features].values
    y = data_with_lag[[f'N_lag{step}']].values

    n = 2860

    X_train = x[:n]
    X_test = x[n:]
    y_train = y[:n]
    y_test = y[n:]

    plt.plot(X_test)
    plt.show()

    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_scaled = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
    X_test_scaled = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

    scaler_y = MinMaxScaler()
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_test_scaled = scaler_y.transform(y_test)

    checkpoint_filepath = r'C:\Users\Andrey\PyCharmMiscProject\checkpoint\TEC22_Npred_LSTM_LAG' + str(i) + '.keras'

    model_checkpoint_callback = ModelCheckpoint(filepath=checkpoint_filepath, save_weights_only=False, monitor='val_loss', mode='min', save_best_only=True, verbose=1)

    early_stop_callback = EarlyStopping(monitor='val_loss', patience=250, verbose=1, restore_best_weights=True, min_delta=0.0001)

    model=Sequential()
    model.add(LSTM(100, input_shape=(X_train_scaled.shape[1], X_train_scaled.shape[2])))
    model.add(Dense(50))
    model.add(Dense(1))

    # TRANSFER LEARNING: Загружаем веса от шага i-1
    if i > 0:
        prev_model = r'C:\Users\Andrey\PyCharmMiscProject\checkpoint\TEC22_Npred_LSTM_LAG' + str(i - 1) + '.keras'
        try:
            model.load_weights(prev_model)
            print(f"Weights loaded from step {i}")
        except:
            pass

    #model.load_weights('checkpoint/TEC22_Npred_LSTM_LAG' + str(i) + '.keras')

    model.compile(loss='mae', optimizer='adam', metrics=['mse', 'mape', 'r2_score'])

    # fit network
    history = model.fit(X_train_scaled, y_train_scaled, epochs=1000, batch_size=32, validation_data=(X_test_scaled, y_test_scaled), verbose=2, shuffle=False, callbacks = [early_stop_callback, model_checkpoint_callback])

    y_test_pred_scaled = model.predict(X_test_scaled)
    yhat = scaler_y.inverse_transform(y_test_pred_scaled)

    MAE_test = metrics.mean_absolute_error(y_test, yhat)
    MSE_test = metrics.mean_squared_error(y_test, yhat)
    MAPE_test = (metrics.mean_absolute_percentage_error(y_test, yhat))*100
    R2_test = metrics.r2_score(y_test, yhat)
    print(f"MAE_test: {MAE_test:.2f}")
    print(f"MSE_test: {MSE_test:.2f}")
    print(f"MAPE_test: {MAPE_test:.2f}")
    print(f"R2_test: {R2_test:.2f}")
    metrics_list.append([MAE_test,MSE_test,MAPE_test,R2_test])

    # 1. Настройка шрифтов (English, Large size)
    plt.rcParams.update({'font.size': 16, 'axes.labelsize': 18, 'axes.titlesize': 22})

    # --- ГРАФИК 1: Actual vs Prediction с интервалом ---
    plt.figure(figsize=(18, 9))
    show_points = 150
    y_true = y_test[-show_points:].flatten()
    y_pred = yhat[-show_points:].flatten()

    # Расчет интервала
    std_err = np.std(y_true - y_pred)
    margin = 1.96 * std_err

    plt.plot(y_true, label='Actual Values', color='#1a1a1a', linewidth=3)
    plt.plot(y_true, label='Actual Values', color='#1a1a1a', linewidth=3)
    plt.plot(y_pred, label='LSTM Prediction', color='#d63031', linewidth=3, linestyle='--')
    plt.fill_between(range(len(y_pred)), (y_pred - margin), (y_pred + margin),
                     color='#fab1a0', alpha=0.3, label='95% Confidence Interval')

    plt.title(f'Power Generation: Forecast Step {step}', pad=20)
    plt.xlabel('Timeline (Test Samples)')
    plt.ylabel('Power Output (N)')
    plt.legend(shadow=True)
    plt.grid(True, alpha=0.3)
    plt.show()

    # --- ГРАФИК 2: Error Analysis (Исправленный) ---
    plt.figure(figsize=(18, 8))

    # Превращаем список в DataFrame для удобства
    metrics_df = pd.DataFrame(metrics_list, columns=['MAE', 'MSE', 'MAPE', 'R2'])

    # Проверка: берем количество строк, которое реально успело накопиться
    actual_steps = range(1, len(metrics_df) + 1)

    plt.bar(actual_steps, metrics_df['MAPE'], color='#0984e3', alpha=0.7, label='MAPE (%)')
    plt.plot(actual_steps, metrics_df['MAPE'], marker='o', color='#00008b', linewidth=3)

    plt.title('Forecasting Error Growth by Horizon', pad=20)
    plt.xlabel('Prediction Step (Days Ahead)')
    plt.ylabel('Mean Absolute Percentage Error (%)')
    plt.xticks(actual_steps)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend()

    plt.tight_layout()
    plt.show()

    import matplotlib.pyplot as plt
    import seaborn as sns

    # 1. Настройка стиля
    plt.rcParams.update({'font.size': 14, 'axes.labelsize': 16, 'axes.titlesize': 20})
    plt.figure(figsize=(12, 10))

    # 2. Подготовка данных (используем последний step_features из цикла)
    # Считаем корреляцию всех признаков с целевым значением N
    features_df = data_with_lag[step_features + [f'N_lag{step}']]
    correlations = features_df.corr()[f'N_lag{step}'].drop(f'N_lag{step}').sort_values(ascending=True)

    # 3. Визуализация
    colors = ['#74b9ff' if x > 0 else '#ff7675' for x in correlations]
    correlations.plot(kind='barh', color=colors, edgecolor='black', alpha=0.8)

    # Добавляем подписи
    plt.title(f'Feature Impact on Prediction (Step {step})', pad=20)
    plt.xlabel('Correlation Coefficient with Target (N)')
    plt.ylabel('Input Features')
    plt.grid(axis='x', linestyle='--', alpha=0.6)

    # Оптимизация расположения, чтобы названия фич не обрезались
    plt.tight_layout()
    plt.show()
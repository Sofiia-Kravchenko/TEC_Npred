import warnings

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
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error
import seaborn as sns

warnings.filterwarnings("ignore")

data = pd.read_csv(r'C:\Users\guryanov\PyCharmMiscProject\pythonProject\data\TEC22_Data_N.csv', delimiter=';', parse_dates=['Date'], dayfirst=True)
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
data['B4_GT41_inWork'] = np.where(data['B4_GT41_N_Aver'] > 25, 1, 0)
data['B4_GT42_inWork'] = np.where(data['B4_GT42_N_Aver'] > 25, 1, 0)
data['B4_inWork'] = np.where(data['B4_N_Aver'] > 25, 1, 0)

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

metrics_list = []
results_list = np.array(())

x = data.loc[:,['T', 'Month_sin', 'Month_cos', 'Year',
                 'B1_inWork', 'B2_inWork', 'B3_inWork', 'B4_GT41_inWork', 'B4_GT42_inWork', 'B4_inWork',
                 'B1_Available_N', 'B2_Available_N', 'B3_Available_N', 'B4_Available_N',
                 'T_Prev', 'Month_sin_Prev', 'Month_cos_Prev', 'Year_Prev',
                 'B4_GT41_N_Aver_Prev', 'B4_GT42_N_Aver_Prev', 'B4_PT40_N_Aver_Prev', 'TEC_N_Aver_Prev',
                 'B1_N_Aver_Prev', 'B2_N_Aver_Prev', 'B3_N_Aver_Prev', 'B4_N_Aver_Prev',
                 'B1_Available_N_Prev', 'B2_Available_N_Prev', 'B3_Available_N_Prev', 'B4_Available_N_Prev', 'TEC_Q_Aver']]
x = x.values
y = data.loc[:,['TEC_N_Aver']]
y = y.values

n = 2860

X_train = x[:n]
X_test = x[n: ]
y_train = y[:n]
y_test = y[n: ]

def create_windows(x_data, y_data, window_size=14):
    X, y = [], []
    for i in range(len(x_data) - window_size):
        X.append(x_data[i : i + window_size])
        y.append(y_data[i + window_size])
    return np.array(X), np.array(y)

scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

scaler_y = MinMaxScaler()
y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

window_size = 7
X_train_win, y_train_win = create_windows(X_train_scaled, y_train_scaled, window_size)
X_test_win, y_test_win = create_windows(X_test_scaled, y_test_scaled, window_size)

X_train_flat = X_train_win.reshape(X_train_win.shape[0], -1)
X_test_flat = X_test_win.reshape(X_test_win.shape[0], -1)

model = CatBoostRegressor(iterations=2000,
                          depth=6,
                          learning_rate=0.01,
                          loss_function='MAE',
                          verbose=0)

model.fit(X_train_flat, y_train_win, eval_set=(X_test_flat, y_test_win), early_stopping_rounds=50)

y_test_pred = model.predict(X_test_flat).reshape(-1, 1)
yhat = scaler_y.inverse_transform(y_test_pred)
y_test_actual = scaler_y.inverse_transform(y_test_win)

MAE_test = metrics.mean_absolute_error(y_test_actual, yhat)
MSE_test = metrics.mean_squared_error(y_test_actual, yhat)
MAPE_test = metrics.mean_absolute_percentage_error(y_test_actual, yhat)
R2_test = metrics.r2_score(y_test_actual, yhat)
print(f"MAE_test: {MAE_test:.2f}")
print(f"MSE_test: {MSE_test:.2f}")
print(f"MAPE_test: {MAPE_test:.2f}")
print(f"R2_test: {R2_test:.2f}")

plt.plot(yhat)
plt.plot(y_test_actual)
plt.show()

learning_rates = [0.01, 0.05, 0.1, 0.2]
depths = [4, 6, 8, 10]
'''
results = []

window_sizes = [1, 3, 7, 14, 30]  # Варианты глубины истории
learning_rates = [0.05, 0.1]
depths = [6, 8]

results = []
best_mae = float('inf')
best_params = {}

# 1. Основной цикл по размеру окна
for ws in window_sizes:
    # Пересоздаем окна для текущего размера
    X_train_win, y_train_win = create_windows(X_train_scaled, y_train_scaled, ws)
    X_test_win, y_test_win = create_windows(X_test_scaled, y_test_scaled, ws)

    X_train_flat = X_train_win.reshape(X_train_win.shape[0], -1)
    X_test_flat = X_test_win.reshape(X_test_win.shape[0], -1)

    y_actual_unscaled = scaler_y.inverse_transform(y_test_win)

    for d in depths:
        for lr in learning_rates:
            test_model = CatBoostRegressor(iterations=1000,
                                           depth=d,
                                           learning_rate=lr,
                                           loss_function='MAE',
                                           random_seed=1,
                                           verbose=0,
                                           early_stopping_rounds=50)

            test_model.fit(X_train_flat, y_train_win, eval_set=(X_test_flat, y_test_win))

            y_test_pred_scaled = test_model.predict(X_test_flat).reshape(-1, 1)
            yhat_unscaled = scaler_y.inverse_transform(y_test_pred_scaled)

            mae = metrics.mean_absolute_error(y_actual_unscaled, yhat_unscaled)

            # Сохраняем результат
            results.append({
                'Window': ws,
                'Depth': d,
                'LR': lr,
                'MAE': mae
            })

            # Отслеживаем лучшую модель
            if mae < best_mae:
                best_mae = mae
                best_params = {'Window': ws, 'Depth': d, 'LR': lr}

# 2. Вывод результатов
df_res = pd.DataFrame(results)
print("results:", results)
print("\nBest Parameters Found:")
print(best_params, "with MAE:", round(best_mae, 2))

# 3. Визуализация (для одного из срезов, например, лучшего Window)
best_ws = best_params['Window']
pivot_data = df_res[df_res['Window'] == best_ws].pivot(index="Depth", columns="LR", values="MAE")

plt.figure(figsize=(12, 10))
sns.set(font_scale=1.5)
sns.heatmap(pivot_data, annot=True, fmt=".2f", cmap='viridis', annot_kws={"size": 20, "weight": "bold"})
plt.title(f'Sensitivity Analysis (Window Size = {best_ws})', fontsize=26, fontweight='bold', pad=25)
plt.show()'''
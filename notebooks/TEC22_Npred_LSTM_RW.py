import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb
from keras.src.layers import Dense, LSTM, Normalization
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from statsmodels.graphics.tukeyplot import results
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn import datasets, linear_model, metrics, model_selection, __all__, ensemble
from tensorflow.keras.callbacks import ModelCheckpoint
from keras import backend as K
import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv(r'C:\Users\Andrey\PyCharmMiscProject\data\TEC22_Data_N.csv', delimiter=';', parse_dates=['Date'], dayfirst=True)
data['Month'] = data['Date'].dt.month
data['Year'] = data['Date'].dt.year
data['Month_sin'] = np.sin(2 * np.pi * data['Month'] / 12)
data['Month_cos'] = np.cos(2 * np.pi * data['Month'] / 12)
data = data[400:]

data.set_index('Date', inplace=True)

n_out = 14

metrics_list = []
results_list = np.array(())

x = data.loc[:,['T', 'B1_inWork', 'B2_inWork', 'B3_inWork' ,'B4_GT41_inWork', 'B4_GT42_inWork', 'B4_inWork', 'Month_sin', 'Month_cos', 'Year']]
x = x.values
y = data.loc[:,['TEC_N_Aver']]
y = y.values

n = 2860

X_train = x[:n]
X_test = x[n: ]
y_train = y[:n]
y_test = y[n: ]

scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

scaler_y = MinMaxScaler()
y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

#X_train_scaled = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
#X_test_scaled = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

def create_windows(x_data, y_data, window_size=14):
    X, y = [], []
    for i in range(len(x_data) - window_size):
        # Берем окно признаков (например, за 24 часа)
        X.append(x_data[i : i + window_size])
        # Целевое значение — следующее значение после окна
        y.append(y_data[i + window_size])
    return np.array(X), np.array(y)

window_size = 7
X_train_win, y_train_win = create_windows(X_train_scaled, y_train_scaled, window_size)
X_test_win, y_test_win = create_windows(X_test_scaled, y_test_scaled, window_size)

checkpoint_filepath = r'C:\Users\Andrey\PyCharmMiscProject\checkpoint\TEC22_Npred_LSTM_RW.keras'
model_checkpoint_callback = ModelCheckpoint(filepath=checkpoint_filepath, save_weights_only=False, monitor='val_loss', mode='min', save_best_only=True, verbose=1)

early_stop_callback = EarlyStopping(monitor='val_loss', patience=250, verbose=1, restore_best_weights=True, min_delta=0.0001)

model = Sequential()
model.add(LSTM(50, input_shape=(X_train_win.shape[1], X_train_win.shape[2])))
model.add(Dense(25))
model.add(Dense(1))

model.compile(loss='mae', optimizer='adam')

#model.load_weights(r'C:\Users\Andrey\PyCharmMiscProject\checkpoint\TEC22_Npred_LSTM_RW.keras')

model.compile(loss='mae', optimizer='adam', metrics=['mse', 'mape', 'r2_score'])

# fit network
history = model.fit(X_train_win, y_train_win, epochs=2500, batch_size=32, validation_data=(X_test_win, y_test_win), verbose=2, shuffle=False, callbacks = [early_stop_callback, model_checkpoint_callback])

y_test_pred_scaled = model.predict(X_test_win)
yhat = scaler_y.inverse_transform(y_test_pred_scaled)
y_test_actual = scaler_y.inverse_transform(y_test_win)

MAE_test = metrics.mean_absolute_error(y_test_actual, yhat)
MSE_test = metrics.mean_squared_error(y_test_actual, yhat)
MAPE_test = metrics.mean_absolute_percentage_error(y_test_actual, yhat)
R2_test = metrics.r2_score(y_test_actual, yhat)
print(f"MAE_test: {MAE_test:.2f}")
print(f"MSE_test: {MSE_test:.2f}")
print(f"MAPE_test: {MAPE_test:.2f}")
print(f"R2_test: {R2_test:.2f}")

def evaluate_window_size(w_size, X_tr_scaled, y_tr, X_te_scaled, y_te):
    # Создаем окна
    X_train_w, y_train_w = create_windows(X_tr_scaled, y_tr, w_size)
    X_test_w, y_test_w = create_windows(X_te_scaled, y_te, w_size)

    K.clear_session()

    # Строим модель
    model = Sequential([
        LSTM(50, input_shape=(X_train_w.shape[1], X_train_w.shape[2])),
        Dense(25),
        Dense(1)
    ])
    model.compile(loss='mae', optimizer='adam')

    # Обучаем (меньше эпох для теста)
    model.fit(X_train_w, y_train_w, epochs=500, batch_size=64, verbose=0, shuffle=False, callbacks = [early_stop_callback])

    # Оценка
    preds = model.predict(X_test_w)
    mae = metrics.mean_absolute_error(y_test_w, preds)
    return mae


window_sizes = [1, 3, 7, 21, 30]
mae_results = []

for w in window_sizes:
    mae = evaluate_window_size(w, X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled)
    mae_results.append(mae)
    print(f"Window: {w} hours | MAE: {mae:.4f}")

plt.figure(figsize=(10, 6))
plt.plot(window_sizes, mae_results, marker='o', linestyle='--', color='b')
plt.title('Влияние размера окна на ошибку модели (MAE)')
plt.xlabel('Размер окна (часы)')
plt.ylabel('MAE на тестовой выборке')
plt.grid(True)
plt.show()
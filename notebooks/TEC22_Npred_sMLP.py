import warnings
import numpy as np
import pandas as pd
from keras.src.callbacks import EarlyStopping
from keras.src.layers import Flatten, Dense
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from tensorflow.keras.models import Sequential
from sklearn import datasets, linear_model, metrics, model_selection

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

X_train_scaled = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_scaled = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

model = Sequential()
model.add(Flatten(input_shape=(X_train_scaled.shape[1], X_train_scaled.shape[2])))

model.add(Dense(128, activation='linear'))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mae', metrics=['mse', 'mape', 'r2_score'])

early_stop_callback = EarlyStopping(monitor='val_loss', patience=250, verbose=1, restore_best_weights=True,
                                    min_delta=0.0001)

history_MNN = model.fit(X_train_scaled,
                    y_train_scaled,
                    epochs=1000,
                    batch_size=30,
                    validation_split=0.2,
                    validation_batch_size=30,
                    callbacks=early_stop_callback,
                    verbose="auto")

y_test_pred_scaled  = model.predict(X_test_scaled)
y_test_pred = scaler_y.inverse_transform(y_test_pred_scaled)

MAE_test = metrics.mean_absolute_error(y_test, y_test_pred)
MSE_test = metrics.mean_squared_error(y_test, y_test_pred)
MAPE_test = metrics.mean_absolute_percentage_error(y_test, y_test_pred)
R2_test = metrics.r2_score(y_test, y_test_pred)
print(f"MAE_test: {MAE_test:.2f}")
print(f"MSE_test: {MSE_test:.2f}")
print(f"MAPE_test: {MAPE_test:.2f}")
print(f"R2_test: {R2_test:.2f}")
import numpy as np
import pandas as pd
from keras import Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import *
from keras.models import *
from matplotlib import pyplot as plt
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn import datasets, linear_model, metrics, model_selection, __all__, ensemble

import warnings
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

data['B1_Available_N'] = data['B1_inWork'] * 250
data['B2_Available_N'] = data['B2_inWork'] * 250
data['B3_Available_N'] = data['B3_inWork'] * 250
data['B4_Available_N'] = data['B4_GT41_inWork'] * 150 + data['B4_GT42_inWork'] * 150 + 75*(data['B4_GT41_inWork']+data['B4_GT42_inWork'])

data['B4_GT41_N_Aver_lag_1'] = data['B4_GT41_N_Aver'].shift(1)
data['B4_GT42_N_Aver_lag_1'] = data['B4_GT42_N_Aver'].shift(1)
data['B4_PT40_N_Aver_lag_1'] = data['B4_PT40_N_Aver'].shift(1)
data['B1_N_Aver_lag_1'] = data['B1_N_Aver'].shift(1)
data['B2_N_Aver_lag_1'] = data['B2_N_Aver'].shift(1)
data['B3_N_Aver_lag_1'] = data['B3_N_Aver'].shift(1)
data['B4_N_Aver_lag_1'] = data['B4_N_Aver'].shift(1)
data['TEC_N_Aver_lag_1'] = data['TEC_N_Aver'].shift(1)
data.dropna(inplace=True)

data.set_index('Date', inplace=True)

metrics_list = []
results_list = np.array(())

x = data.loc[:,['T', 'B1_inWork', 'B2_inWork', 'B3_inWork' ,'B4_GT41_inWork', 'B4_GT42_inWork', 'B4_inWork', 'Month_sin', 'Month_cos', 'Year',
               'B1_Available_N', 'B2_Available_N', 'B3_Available_N', 'B4_Available_N',
                'B4_GT41_N_Aver_lag_1', 'B4_GT42_N_Aver_lag_1', 'B4_PT40_N_Aver_lag_1', 'B4_N_Aver_lag_1',
                'B1_N_Aver_lag_1', 'B2_N_Aver_lag_1' ,'B3_N_Aver_lag_1', 'TEC_N_Aver_lag_1']]
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
checkpoint_filepath = r'C:\Users\guryanov\PyCharmMiscProject\pythonProject\checkpoint\TEC22_Npred_LSTM.keras'

model_checkpoint_callback = ModelCheckpoint(filepath=checkpoint_filepath, save_weights_only=False, monitor='val_loss', mode='min', save_best_only=True, verbose=1)

early_stop_callback = EarlyStopping(monitor='val_loss', patience=250, verbose=1, restore_best_weights=True, min_delta=0.0001)

model=Sequential()
model.add(LSTM(100, input_shape=(X_train_scaled.shape[1], X_train_scaled.shape[2])))
model.add(Dense(50))
model.add(Dense(1))

#model.load_weights(r'C:\Users\Andrey\PyCharmMiscProject\checkpoint\TEC22_Npred_LSTM.keras')

model.compile(loss='mae', optimizer='adam', metrics=['mse', 'mape', 'r2_score'])

# fit network
history_LSTM = model.fit(X_train_scaled,
                    y_train_scaled,
                    epochs=1000,
                    batch_size=30,
                    validation_data=(X_test_scaled, y_test_scaled),
                    validation_batch_size=30,
                    callbacks=[early_stop_callback,model_checkpoint_callback],
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
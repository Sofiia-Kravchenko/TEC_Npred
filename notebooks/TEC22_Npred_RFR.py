import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from keras import Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import *
from keras.models import *
from matplotlib import pyplot as plt
from pandas import DataFrame
from pandas import concat
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn import datasets, linear_model, metrics, model_selection, __all__, ensemble
import seaborn as sns

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

params = {
    "n_estimators": 35,
    "max_features": 3,
    "random_state": 1,
}

model_RFR = RandomForestRegressor(**params).fit(X_train_scaled,y_train)

y_test_pred = model_RFR.predict(X_test_scaled)
MAE_test = metrics.mean_absolute_error(y_test, y_test_pred)
MSE_test = metrics.mean_squared_error(y_test, y_test_pred)
MAPE_test = metrics.mean_absolute_percentage_error(y_test, y_test_pred)
R2_test = metrics.r2_score(y_test, y_test_pred)
print(f"MAE_test: {MAE_test:.2f}")
print(f"MSE_test: {MSE_test:.2f}")
print(f"MAPE_test: {MAPE_test:.2f}")
print(f"R2_test: {R2_test:.2f}")

scores_train =[]
scores_test =[]
calc_range = 250
for k in range(1, calc_range):
    print(k)
    rfc = RandomForestRegressor(n_estimators=k, max_features = 3, random_state=1, min_weight_fraction_leaf=0.0)
    rfc.fit(X_train_scaled, y_train)
    y_pred_test = rfc.predict(X_test_scaled)
    scores_test.append(mean_absolute_error(y_test, y_pred_test))
    y_pred_train = rfc.predict(X_train_scaled)
    scores_train.append(mean_absolute_error(y_train, y_pred_train))
plt.plot(range(1, calc_range), scores_test, label="Test Set MSE")
plt.plot(range(1, calc_range), scores_train, label="Train Set MSE")
plt.xlabel('Value of n_estimators for RandomForestRegressor')
plt.ylabel('Testing Accuracy')
plt.legend(loc="upper right")
plt.grid(True)
plt.show()
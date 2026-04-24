import numpy as np
import pandas as pd
from keras import Sequential
from keras.src.callbacks import ModelCheckpoint, EarlyStopping
from keras.src.layers import LSTM, Dense
from matplotlib import pyplot as plt
from pandas import DataFrame
from pandas import concat
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sklearn import datasets, linear_model, metrics, model_selection, __all__, ensemble

import warnings

warnings.filterwarnings("ignore")

data = pd.read_csv(r'C:\Users\Andrey\PyCharmMiscProject\data\TEC22_Data_N.csv', delimiter=';', parse_dates=['Date'], dayfirst=True)
data['Month'] = data['Date'].dt.month
data['Year'] = data['Date'].dt.year
data = data[400:]

data.set_index('Date', inplace=True)

n_out = 14

metrics_list = []
results_list = np.array(())

x = data.loc[:,['T', 'B1_inWork', 'B2_inWork', 'B3_inWork' ,'B4_GT41_inWork', 'B4_GT42_inWork', 'B4_inWork', 'Month', 'Year']]
x = x.values
y = data.loc[:,['TEC_N_Aver']]
y = y.values

n = 2860

X_train = x[:n]
X_test = x[n:]
y_train = y[:n]
y_test = y[n:]

scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

params = {
    "random_state": 0,
    "n_estimators": 5000,
    "max_depth": 4,
    "min_samples_split": 3,
    "learning_rate": 0.01,
    "loss": "absolute_error",
}

model_GBR = ensemble.GradientBoostingRegressor(**params).fit(X_train_scaled,y_train)

y_train_pred = model_GBR.predict(X_train_scaled)
MAE_train = metrics.mean_absolute_error(y_train, y_train_pred)
MSE_train = metrics.mean_squared_error(y_train, y_train_pred)
R2_train = metrics.r2_score(y_train, y_train_pred)
print(f"MAE_LR_train: {MAE_train:.2f}")
print(f"MSE_LR_train: {MSE_train:.2f}")
print(f"R2_LR_train: {R2_train:.2f}")

y_test_pred = model_GBR.predict(X_test_scaled)
MAE_test = metrics.mean_absolute_error(y_test, y_test_pred)
MSE_test = metrics.mean_squared_error(y_test, y_test_pred)
MAPE_test = (metrics.mean_absolute_percentage_error(y_test, y_test_pred))*100
R2_test = metrics.r2_score(y_test, y_test_pred)
print(f"MAE_LR_test: {MAE_test:.2f}")
print(f"MSE_LR_test: {MSE_test:.2f}")
print(f"MAPE_test: {MAPE_test:.2f}")
print(f"R2_LR_test: {R2_test:.2f}")


test_score = np.zeros((params["n_estimators"],), dtype=np.float64)
for i, y_pred in enumerate(model_GBR.staged_predict(X_test_scaled)):
    test_score[i] = mean_absolute_error(y_test, y_pred)

plt.title("MAE")
plt.plot(np.arange(params["n_estimators"]) + 1, model_GBR.train_score_, "g-",label="Training Set MAE")
plt.plot(np.arange(params["n_estimators"]) + 1, test_score, "b-", label="Test Set MAE")
plt.legend(loc="upper right")
plt.xlabel("Boosting Iterations")
plt.ylabel("MAE")
plt.grid(True)
plt.show()

df = pd.DataFrame(metrics_list, columns=['MAE', 'MSE', 'MAPE', 'R2'])
df.to_excel(r'C:\Users\Andrey\PyCharmMiscProject\data\TEC22_Npred_XGB_metrics.xlsx')

df = pd.DataFrame(results_list)
df.to_excel(r'C:\Users\Andrey\PyCharmMiscProject\data\TEC22_Npred_XGB_rez.xlsx')

'''selected_columns = ['TEC_Q_Aver','TEC_N_Aver', 'T', 'Month', 'Year']
correlation_matrix = data[selected_columns].corr()
plt.figure(figsize=(15, 15))
sb.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Корреляционная матрица')
plt.show()'''
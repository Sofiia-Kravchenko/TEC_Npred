import numpy as np
import pandas as pd
import shap
from catboost import CatBoostRegressor
from matplotlib import pyplot as plt
from sklearn import metrics
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns

data = pd.read_csv(r'C:\Users\guryanov\PycharmProjects\TEC_Npred\data\TEC22_Data.csv', delimiter=';', parse_dates=['Date'], dayfirst=True)

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
data['B4_inWork'] = np.where(data['B4_N_Aver'] > 40, 1, 0)

data = data.drop(data[(data['B1_N_Aver'] < 125) & (data['B1_N_Aver']  > 0)].index)
data = data.drop(data[(data['B2_N_Aver'] < 125) & (data['B2_N_Aver']  > 0)].index)
data = data.drop(data[(data['B3_N_Aver'] < 125) & (data['B3_N_Aver']  > 0)].index)
data = data.drop(data[(data['B4_GT41_N_Aver'] < 50) & (data['B4_GT41_N_Aver']  > 0)].index)
data = data.drop(data[(data['B4_GT42_N_Aver'] < 50) & (data['B4_GT42_N_Aver']  > 0)].index)
data = data.drop(data[(data['B4_PT40_N_Aver'] < 40) & (data['B4_PT40_N_Aver']  > 0)].index)
data = data.drop(data[(data['TEC_N_Aver'] == 0)].index)

data['B1_Available_N'] = data['B1_inWork'] * 250
data['B2_Available_N'] = data['B2_inWork'] * 250
data['B3_Available_N'] = data['B3_inWork'] * 250
#data['B4_Available_N'] = np.where(data['T'] < -2,
#    (data['B4_GT41_inWork'] * 160 + data['B4_GT42_inWork'] * 160 + 72 *(data['B4_GT41_inWork'] + data['B4_GT42_inWork'])),
#    (data['B4_GT41_inWork'] * 150 + data['B4_GT42_inWork'] * 150 + 72 *(data['B4_GT41_inWork'] + data['B4_GT42_inWork'])))
data['B4_Available_N'] = (data['B4_GT41_inWork'] * 150 + data['B4_GT42_inWork'] * 150 + 72 *(data['B4_GT41_inWork'] + data['B4_GT42_inWork']))

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

is_half_block = (data['B4_GT41_inWork'] + data['B4_GT42_inWork'] == 1)
data['sample_weight'] = np.where(is_half_block, 2.0, 1.0)

train_size = 2562
n_out = 14

data.set_index('Date', inplace=True)

base_features = ['T', 'Month_sin', 'Month_cos',
                'B1_inWork', 'B2_inWork', 'B3_inWork' ,'B4_GT41_inWork', 'B4_GT42_inWork', 'B4_inWork',
                'B1_Available_N', 'B2_Available_N', 'B3_Available_N', 'B4_Available_N',
                'T_Prev', 'TEC_N_Aver_Prev']

df = DataFrame(data)
df['T_rolling_3'] = df['T'].rolling(window=3).mean()

cols, names, agg = list(), list(), list()
for i in range(1, n_out + 1):

    weights = df['sample_weight'].values
    weights_train = weights[:train_size]

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

learning_rates = [0.01, 0.05, 0.1, 0.2]
depths = [4, 6, 8, 10]

for i in range(0, n_out, 1):
    step = i + 1
    print(f"\n=== Step training {step} ===")

    step_features = base_features + [
        f'T_lag{step}',
        f'B1_inWork_lag{step}', f'B2_inWork_lag{step}', f'B3_inWork_lag{step}',
        f'B4_GT41_inWork_lag{step}', f'B4_GT42_inWork_lag{step}', f'B4_inWork_lag{step}',
        f'B1_Available_N_lag{step}',
        f'B2_Available_N_lag{step}',
        f'B3_Available_N_lag{step}',
        f'B4_Available_N_lag{step}'
    ]

    x = data_with_lag[step_features].values
    y = data_with_lag.loc[:, [f'TEC_N_Aver_lag{step}']].values
    X_train, X_test = x[:train_size], x[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    scaler_y = MinMaxScaler()
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_test_scaled = scaler_y.transform(y_test)

    best_mae = float('inf')
    best_params = {'depth': 6, 'lr': 0.01} # значения по умолчанию

    for d in depths:
        for lr in learning_rates:
            test_model = CatBoostRegressor(iterations=500,
                                           depth=d,
                                           learning_rate=lr,
                                           loss_function='MAE',
                                           verbose=0)
            test_model.fit(X_train_scaled, y_train_scaled, eval_set=(X_test_scaled, y_test_scaled), sample_weight=weights_train, early_stopping_rounds=250, use_best_model=True)
            preds = test_model.predict(X_test_scaled)
            yhat = scaler_y.inverse_transform(preds.reshape(-1, 1))
            mae = metrics.mean_absolute_error(y_test, yhat)
            if mae < best_mae:
                best_mae = mae
                best_params = {'depth': d, 'lr': lr}

    print(f"Лучшие параметры для шага {step}: {best_params}, MAE: {best_mae:.2f}")

    model = CatBoostRegressor(iterations=2000,
                              depth=best_params['depth'],
                              learning_rate=best_params['lr'],
                              loss_function='MAE',
                              verbose=0)

    model.fit(X_train_scaled, y_train_scaled, eval_set=(X_test_scaled, y_test_scaled), sample_weight=weights_train, early_stopping_rounds=250, use_best_model=True)

    yhat_scaled = model.predict(X_test_scaled)
    yhat = scaler_y.inverse_transform(yhat_scaled.reshape(-1, 1))

    MAE_test = metrics.mean_absolute_error(y_test, yhat)
    MSE_test = metrics.mean_squared_error(y_test, yhat)
    MAPE_test = metrics.mean_absolute_percentage_error(y_test, yhat) * 100
    R2_test = metrics.r2_score(y_test, yhat)
    print(MAE_test)

    results = []

# Цикл перебора
for d in depths:
    row = []
    for lr in learning_rates:
        test_model = CatBoostRegressor(iterations=1000,  # 1000 достаточно для теста
                                       depth=d,
                                       learning_rate=lr,
                                       loss_function='MAE',
                                       verbose=0,
                                       early_stopping_rounds=50)
        test_model.fit(X_train_scaled, y_train, eval_set=(X_test_scaled, y_test))
        preds = test_model.predict(X_test_scaled)
        mae = metrics.mean_absolute_error(y_test, preds)
        row.append(mae)
    results.append(row)

# Визуализация через Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(results, annot=True, fmt=".2f",
            xticklabels=learning_rates,
            yticklabels=depths,
            cmap='YlGnBu')
plt.xlabel('Learning Rate')
plt.ylabel('Depth')
plt.title('Влияние параметров на MAE (чем меньше значение, тем лучше)')
plt.show()
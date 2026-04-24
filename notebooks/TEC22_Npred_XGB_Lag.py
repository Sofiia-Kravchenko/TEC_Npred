import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from keras import Sequential
from keras.src.callbacks import ModelCheckpoint, EarlyStopping
from keras.src.layers import LSTM, Dense
from matplotlib import pyplot as plt
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn import datasets, linear_model, metrics, model_selection, __all__, ensemble
import seaborn as sns

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
        f'T_mean{step}',
    ]

    x = data_with_lag[step_features].values
    y = data_with_lag[[f'N_lag{step}']].values

    n = 2860

    X_train = x[:n]
    X_test = x[n:]
    y_train = y[:n]
    y_test = y[n:]

    scaler = MinMaxScaler()
    #scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    scaler_y = MinMaxScaler()
    #scaler_y = RobustScaler()
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_test_scaled = scaler_y.transform(y_test)

    X_train_flat = X_train_scaled
    X_test_flat = X_test_scaled
    y_train_win = y_train_scaled
    y_test_win = y_test_scaled

    model = CatBoostRegressor(iterations=2000,
                              depth=6,
                              learning_rate=0.01,
                              loss_function='MAE',
                              random_seed=42,
                              verbose=0)

    model.fit(X_train_flat, y_train_win, eval_set=(X_test_flat, y_test_win), early_stopping_rounds=50)

    y_test_pred = model.predict(X_test_flat).reshape(-1, 1)
    yhat = scaler_y.inverse_transform(y_test_pred)
    y_test_actual = scaler_y.inverse_transform(y_test_win)

    MAE_test = metrics.mean_absolute_error(y_test, yhat)
    MSE_test = metrics.mean_squared_error(y_test, yhat)
    MAPE_test = metrics.mean_absolute_percentage_error(y_test, yhat)
    R2_test = metrics.r2_score(y_test, yhat)
    print(f"MAE_test: {MAE_test:.2f}")
    print(f"MSE_test: {MSE_test:.2f}")
    print(f"MAPE_test: {MAPE_test:.2f}")
    print(f"R2_test: {R2_test:.2f}")
    metrics_list.append([MAE_test,MSE_test,MAPE_test,R2_test])

    if i == 0:
        results_list = yhat
    else:
        results_list = np.hstack((results_list, yhat))

    # plot history

metrics_np = np.array(metrics_list)

sns.set_style("whitegrid")

# Увеличиваем базовый шрифт на треть (был 16, стал ~22)
plt.rcParams.update({'font.size': 22})

fig, ax1 = plt.subplots(figsize=(20, 11)) # Немного увеличили размер окна для крупных шрифтов

days = np.arange(1, n_out + 1)
mae_values = metrics_np[:, 0]
r2_values = metrics_np[:, 3]

# График MAE (Bars)
color_mae = '#3498db'
ax1.set_xlabel('Forecast Horizon (Days)', fontsize=28, fontweight='bold', labelpad=20)
ax1.set_ylabel('MAE (Mean Absolute Error)', color=color_mae, fontsize=28, fontweight='bold', labelpad=20)
bars = ax1.bar(days, mae_values, color=color_mae, alpha=0.7, label='MAE', edgecolor='black', linewidth=1.5)

# Настройка делений (Ticks)
ax1.tick_params(axis='y', labelcolor=color_mae, labelsize=22)
ax1.tick_params(axis='x', labelsize=22)

# Вторая ось для R2 (Line)
ax2 = ax1.twinx()
color_r2 = '#e74c3c'
ax2.set_ylabel('R2 Score (Accuracy)', color=color_r2, fontsize=28, fontweight='bold', labelpad=20)
ax2.plot(days, r2_values, color=color_r2, marker='o', markersize=14, linewidth=6, label='R2 Score')
ax2.tick_params(axis='y', labelcolor=color_r2, labelsize=22)

# Лимиты осей
ax1.set_ylim(0, max(mae_values) * 1.3) # Динамический запас сверху
ax2.set_ylim(min(r2_values) - 0.55, 1.1)

# Заголовок (увеличен на треть)
plt.title('Prediction Quality per Forecast Day\n(Step-by-Step Gradient booster Training)',
          fontsize=32, fontweight='bold', pad=40)

plt.xticks(days)

# Легенда (крупнее и ниже)
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='upper center',
           bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=True, fontsize=24, shadow=True)

# Подписи MAE внутри столбцов (увеличены)
for bar in bars:
    yval = bar.get_height()
    ax1.text(
        bar.get_x() + bar.get_width()/2,
        yval * 0.85,
        f'{yval:.1f}',
        ha='center', va='top',
        fontsize=18, fontweight='bold', color='white'
    )

# Подписи R2 над точками (увеличены)
for x, y in zip(days, r2_values):
    ax2.text(
        x, y + 0.02,
        f'{y:.2f}',
        ha='center', va='bottom',
        fontsize=18, fontweight='bold', color=color_r2,
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=2)
    )

plt.tight_layout()
plt.show()
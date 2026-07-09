import optuna
import pandas as pd
import numpy as np
import tensorflow as tf
import pulp
import seaborn as sns
from catboost import CatBoostRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from pandas import DataFrame, concat
from optuna.integration import CatBoostPruningCallback
from scipy.optimize import minimize

goal_mapping = {
    "data/TEC22_Data.csv": ["B1", "B2", "B3", "B4", "TEC"],
    "data/TEC14_Data.csv": ["B1", "B2", "TEC"]
}

def prepare_stat_data(filepath, test_start_index):
    data = pd.read_csv(filepath, delimiter=';', parse_dates=['Date'], dayfirst=True)

    data['Month'] = data['Date'].dt.month
    data['Year'] = data['Date'].dt.year
    data['Month_sin'] = np.sin(2 * np.pi * data['Month'] / 12)
    data['Month_cos'] = np.cos(2 * np.pi * data['Month'] / 12)

    first_work_day = data[data['TEC_N_Aver'] > 0].index[0]
    data = data.loc[first_work_day:]

    if filepath == 'data/TEC22_Data.csv':
        data['B1_N_Aver'] = data['B1_N'] / 24
        data['B2_N_Aver'] = data['B2_N'] / 24
        data['B3_N_Aver'] = data['B3_N'] / 24
        data['B4_GT41_N_Aver'] = data['B4_GT41_N'] / 24
        data['B4_GT42_N_Aver'] = data['B4_GT42_N'] / 24
        data['B4_PT40_N_Aver'] = data['B4_N_Aver'] - data['B4_GT41_N_Aver'] - data['B4_GT42_N_Aver']

        data['B1_inWork'] = np.where(data['B1_N_Aver'] >= 125, 1, 0)
        data['B2_inWork'] = np.where(data['B2_N_Aver'] >= 125, 1, 0)
        data['B3_inWork'] = np.where(data['B3_N_Aver'] >= 125, 1, 0)
        data['B4_GT41_inWork'] = np.where(data['B4_GT41_N_Aver'] >= 50, 1, 0)
        data['B4_GT42_inWork'] = np.where(data['B4_GT42_N_Aver'] >= 50, 1, 0)
        data['B4_inWork'] = np.where(data['B4_N_Aver'] >= 76, 1, 0)

        data.loc[data['B1_N_Aver'] < 125, 'B1_N_Aver'] = 0
        data.loc[data['B2_N_Aver'] < 125, 'B2_N_Aver'] = 0
        data.loc[data['B3_N_Aver'] < 125, 'B3_N_Aver'] = 0
        data.loc[data['B4_GT41_N_Aver'] <= 50, 'B4_GT41_N_Aver'] = 0
        data.loc[data['B4_GT42_N_Aver'] <= 50, 'B4_GT42_N_Aver'] = 0
        data.loc[data['B4_N_Aver'] <= 76, 'B4_N_Aver'] = 0
        data.loc[data['TEC_N_Aver'] <= 50, 'TEC_N_Aver'] = 0

        data['B4_GT41_N_Aver'] = np.where(data['B4_N_Aver'] < 76, 0, data['B4_GT41_N_Aver'])
        data['B4_GT42_N_Aver'] = np.where(data['B4_N_Aver'] < 76, 0, data['B4_GT42_N_Aver'])

        data['B1_Available_Nmax'] = data['B1_inWork'] * 250
        data['B2_Available_Nmax'] = data['B2_inWork'] * 250
        data['B3_Available_Nmax'] = data['B3_inWork'] * 250
        data['B4_GT41_Available_Nmax'] = np.where(data['T'] <= -2.3, data['B4_GT41_inWork']*172.7, data['B4_GT41_inWork']*(-0.9484 * data['T'] + 170.78))
        data['B4_GT42_Available_Nmax'] = np.where(data['T'] <= -2.3, data['B4_GT42_inWork']*172.7, data['B4_GT42_inWork']*(-0.9484 * data['T'] + 170.78))
        gt41_conditions = [((data['Year'] >= 2024) & (data['B4_GT41_Available_Nmax'] > 160.0)), (data['Year'] < 2024), ((data['Year'] >= 2024) & (data['B4_GT41_Available_Nmax'] <= 160.0))]
        gt42_conditions = [((data['Year'] >= 2024) & (data['B4_GT42_Available_Nmax'] > 160.0)), (data['Year'] < 2024), ((data['Year'] >= 2024) & (data['B4_GT42_Available_Nmax'] <= 160.0))]
        gt41_choices = [160.0, (data['B4_GT41_Available_Nmax']), (data['B4_GT41_Available_Nmax'])]
        gt42_choices = [160.0, (data['B4_GT42_Available_Nmax']), (data['B4_GT42_Available_Nmax'])]
        data['B4_GT41_Available_Nmax'] = np.select(gt41_conditions , gt41_choices, 160)
        data['B4_GT42_Available_Nmax'] = np.select(gt42_conditions , gt42_choices, 160)
        data['B4_Available_Nmax'] = data['B4_GT41_Available_Nmax'] + data['B4_GT42_Available_Nmax'] + 72*(data['B4_GT41_inWork'] + data['B4_GT42_inWork'])
        data['TEC_Available_Nmax'] = data['B1_Available_Nmax'] + data['B2_Available_Nmax'] + data['B3_Available_Nmax']+ data['B4_Available_Nmax']

        data['B1_Available_Nmin'] = data['B1_inWork'] * 125
        data['B2_Available_Nmin'] = data['B2_inWork'] * 125
        data['B3_Available_Nmin'] = data['B3_inWork'] * 125
        data['B4_Available_Nmin'] = np.where(data['T'] <= -2.3,
                                             0.5*data['B4_GT41_inWork']*172.7 + 0.5*data['B4_GT42_inWork']*172.7 + 26*(data['B4_GT41_inWork'] + data['B4_GT42_inWork']),
                                             0.5*data['B4_GT41_inWork']*(-0.9484 * data['T'] + 170.78) + 0.5*data['B4_GT42_inWork']*(-0.9484 * data['T'] + 170.78) + 26*(data['B4_GT41_inWork'] + data['B4_GT42_inWork']))
        data['TEC_Available_Nmin'] = data['B1_Available_Nmin'] + data['B2_Available_Nmin'] + data['B3_Available_Nmin']+ data['B4_Available_Nmin']

        data.dropna(inplace=True)

        x = data.loc[:, ['T',  'Month_sin', 'Month_cos',
                        'B1_Available_Nmax', 'B2_Available_Nmax', 'B3_Available_Nmax', 'B4_Available_Nmax', 'TEC_Available_Nmax',
                        'B1_Available_Nmin', 'B2_Available_Nmin', 'B3_Available_Nmin', 'B4_Available_Nmin', 'TEC_Available_Nmin']].values
        y = data.loc[:, ['TEC_N_Aver']].values

        corr_features = [
            'T',  'Month_sin', 'Month_cos',
                        'B1_Nmax', 'B2_Nmax', 'B3_Nmax', 'B4_Nmax', 'TEC_Nmax',
                        'B1_Nmin', 'B2_Nmin', 'B3_Nmin', 'B4_Nmin', 'TEC_Nmin', 'TEC_N_Aver'
        ]

        # Создаем временный DataFrame из массивов x и y
        df_corr = pd.DataFrame(
            data=np.hstack([x, y]),
            columns=corr_features
        )

        corr_matrix = df_corr.corr()
        plt.figure(figsize=(14, 11), dpi=100)
        sns.set_theme(style='white')
        plt.rcParams.update({
            'font.size': 14,
            'axes.labelsize': 16,
            'xtick.labelsize': 13,
            'ytick.labelsize': 13
        })

        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            linewidths=1.5,
            vmin=-1, vmax=1,
            annot_kws={"size": 14, "weight": "bold"},
            cbar_kws={"shrink": 0.8}
        )

        plt.xticks(rotation=45, ha='right', weight='bold')
        plt.yticks(rotation=0, weight='bold')

        plt.title('Correlation matrix of CHP and electricity generation features', fontsize=18, pad=20, weight='bold')
        plt.tight_layout()

        plt.show()

        y_true_combined = data.loc[:, ['TEC_N_Aver', 'TEC_Available_Nmax',
                                       'TEC_Available_Nmin']].values

    if filepath == 'data/TEC14_Data.csv':
        data['B1_GT11_N_Aver'] = data['B1_GT11_N'] / 24
        data['B1_GT12_N_Aver'] = data['B1_GT12_N'] / 24
        data['B1_PT10_N_Aver'] = data['B1_PT10_N'] / 24

        data['B2_GT21_N_Aver'] = data['B2_GT21_N'] / 24
        data['B2_GT22_N_Aver'] = data['B2_GT22_N'] / 24
        data['B2_PT20_N_Aver'] = data['B2_PT20_N'] / 24

        data['B1_GT11_inWork'] = np.where(data['B1_GT11_N_Aver'] >= 30, 1, 0)
        data['B1_GT12_inWork'] = np.where(data['B1_GT12_N_Aver'] >= 30, 1, 0)
        data['B2_GT21_inWork'] = np.where(data['B2_GT21_N_Aver'] >= 30, 1, 0)
        data['B2_GT22_inWork'] = np.where(data['B2_GT22_N_Aver'] >= 30, 1, 0)
        data['B1_inWork'] = np.where(data['B1_N_Aver'] > 45, 1, 0)
        data['B2_inWork'] = np.where(data['B2_N_Aver'] > 45, 1, 0)

        data.loc[data['B1_GT11_N_Aver'] < 30, 'B1_GT11_N_Aver'] = 0
        data.loc[data['B1_GT12_N_Aver'] < 30, 'B1_GT12_N_Aver'] = 0
        data.loc[data['B2_GT21_N_Aver'] < 30, 'B2_GT21_N_Aver'] = 0
        data.loc[data['B2_GT22_N_Aver'] < 30, 'B2_GT22_N_Aver'] = 0
        data.loc[data['B1_N_Aver'] < 45, 'B1_N_Aver'] = 0
        data.loc[data['B2_N_Aver'] < 45, 'B2_N_Aver'] = 0
        data.loc[data['TEC_N_Aver'] < 45, 'TEC_N_Aver'] = 0
        data['B1_GT11_N_Aver'] = np.where(data['B1_N_Aver'] < 45, 0,  data['B1_GT11_N_Aver'])
        data['B1_GT12_N_Aver'] = np.where(data['B1_N_Aver'] < 45, 0,  data['B1_GT12_N_Aver'])
        data['B2_GT21_N_Aver'] = np.where(data['B2_N_Aver'] < 45, 0,  data['B2_GT21_N_Aver'])
        data['B2_GT22_N_Aver'] = np.where(data['B2_N_Aver'] < 45, 0,  data['B2_GT22_N_Aver'])

        data['B1_Available_Nmax'] = (data['B1_GT11_inWork'] * (2 / 1000000000 * data['T']**6 + 7 / 1000000000 * data['T']**5 - 4 / 1000000 * data['T']**4 - 0.0001 * data['T']**3 + 0.0006 * data['T']**2 - 0.2475 * data['T'] + 69.719)
                                     + data['B1_GT12_inWork'] * (2 / 1000000000 * data['T']**6 + 7 / 1000000000 * data['T']**5 - 4 / 1000000 * data['T']**4 - 0.0001 * data['T']**3 + 0.0006 * data['T']**2 - 0.2475 * data['T'] + 69.719)
                                     + 26 * (data['B1_GT11_inWork'] + data['B1_GT12_inWork']))
        data['B2_Available_Nmax'] = (data['B2_GT21_inWork'] * (2 / 1000000000 * data['T']**6 + 7 / 1000000000 * data['T']**5 - 4 / 1000000 * data['T']**4 - 0.0001 * data['T']**3 + 0.0006 * data['T']**2 - 0.2475 * data['T'] + 69.719)
                                     + data['B2_GT22_inWork'] * (2 / 1000000000 * data['T']**6 + 7 / 1000000000 * data['T']**5 - 4 / 1000000 * data['T']**4 - 0.0001 * data['T']**3 + 0.0006 * data['T']**2 - 0.2475 * data['T'] + 69.719)
                                     + 26 * (data['B2_GT21_inWork'] + data['B2_GT22_inWork']))
        data['TEC_Available_Nmax'] = data['B1_Available_Nmax'] + data['B2_Available_Nmax']

        data['B1_Available_Nmin'] = (data['B1_GT11_inWork'] * 0.4*(2 / 1000000000 * data['T']**6 + 7 / 1000000000 * data['T']**5 - 4 / 1000000 * data['T']**4 - 0.0001 * data['T']**3 + 0.0006 * data['T']**2 - 0.2475 * data['T'] + 69.719)
                                     + data['B1_GT12_inWork'] * 0.4*(2 / 1000000000 * data['T']**6 + 7 / 1000000000 * data['T']**5 - 4 / 1000000 * data['T']**4 - 0.0001 * data['T']**3 + 0.0006 * data['T']**2 - 0.2475 * data['T'] + 69.719)
                                     + 15 * ( data['B1_GT11_inWork'] + data['B1_GT12_inWork']))
        data['B2_Available_Nmin'] = (data['B2_GT21_inWork'] * 0.4*(2 / 1000000000 * data['T']**6 + 7 / 1000000000 * data['T']**5 - 4 / 1000000 * data['T']**4 - 0.0001 * data['T']**3 + 0.0006 * data['T']**2 - 0.2475 * data['T'] + 69.719)
                                     + data['B2_GT22_inWork'] * 0.4*(2 / 1000000000 * data['T']**6 + 7 / 1000000000 * data['T']**5 - 4 / 1000000 * data['T']**4 - 0.0001 * data['T']**3 + 0.0006 * data['T']**2 - 0.2475 * data['T'] + 69.719)
                                     + 15 * (data['B2_GT21_inWork'] + data['B2_GT22_inWork']))
        data['TEC_Available_Nmin'] = data['B1_Available_Nmin'] + data['B2_Available_Nmin']

        data.dropna(inplace=True)

        x = data.loc[:, ['T', 'Month_sin', 'Month_cos',
                         'B1_Available_Nmax', 'B2_Available_Nmax',
                         'B1_Available_Nmin', 'B2_Available_Nmin',
                         'TEC_Available_Nmax','TEC_Available_Nmin']].values
        y = data.loc[:, ['TEC_N_Aver']].values
        y_true_combined = data.loc[:, ['TEC_N_Aver', 'TEC_Available_Nmax',
                                       'TEC_Available_Nmin']].values

        corr_features = [
            'T', 'Month_sin', 'Month_cos',
            'B1_Nmax', 'B2_Nmax',  # Сократил названия для лучшей читаемости на графике
            'B1_Nmin', 'B2_Nmin',
            'TEC_Nmax', 'TEC_Nmin',
            'TEC_N_Aver'  # Наша целевая переменная Y
        ]

        # Создаем временный DataFrame из массивов x и y
        df_corr = pd.DataFrame(
            data=np.hstack([x, y]),
            columns=corr_features
        )

        # 2. Считаем матрицу корреляции Пирсона
        corr_matrix = df_corr.corr()

        # 3. Настраиваем глобальные крупные шрифты для графика
        plt.figure(figsize=(14, 11), dpi=100)
        sns.set_theme(style='white')
        plt.rcParams.update({
            'font.size': 14,  # Крупный базовый шрифт
            'axes.labelsize': 16,  # Шрифт осей
            'xtick.labelsize': 13,  # Шрифт подписей колонок по X
            'ytick.labelsize': 13  # Шрифт подписей строк по Y
        })

        # 4. Строим тепловую карту (Heatmap)
        # Используем расходящуюся палитру 'coolwarm' (синий - холодно/отрицательно, красный - горячо/положительно)
        sns.heatmap(
            corr_matrix,
            annot=True,  # Включает отображение цифр внутри ячеек
            fmt=".2f",  # Округляет значения до 2 знаков после запятой
            cmap="coolwarm",  # Контрастная цветовая схема
            linewidths=1.5,  # Толщина белых линий-разделителей ячеек
            vmin=-1, vmax=1,  # Фиксируем границы корреляции от -1 до 1
            annot_kws={"size": 14, "weight": "bold"},  # КРУПНЫЙ и ЖИРНЫЙ шрифт цифр внутри ячеек
            cbar_kws={"shrink": 0.8}  # Немного уменьшаем боковую цветовую шкалу
        )

        # 5. Красиво поворачиваем подписи, чтобы они не налезали друг на друга
        plt.xticks(rotation=45, ha='right', weight='bold')
        plt.yticks(rotation=0, weight='bold')

        plt.title('Correlation matrix of CHP and electricity generation features', fontsize=18, pad=20, weight='bold')
        plt.tight_layout()

        # 6. Выводим на экран
        plt.show()

    data.set_index('Date', inplace=True)

    n = test_start_index
    x_train, x_test = x[:n], x[n:]
    y_train, y_test = y[:n], y[n:]
    y_train_comb_raw, y_test_comb_raw = y_true_combined[:n], y_true_combined[n:]

    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()

    x_train_s = scaler_x.fit_transform(x_train)
    x_test_s = scaler_x.transform(x_test)

    scaler_y.fit(y_train)
    y_train_s =scaler_y.transform(y_train)
    y_test_s = scaler_y.transform(y_test)

    y_train_s_combined = scale_combined(y_train_comb_raw, scaler_y)
    y_test_s_combined = scale_combined(y_test_comb_raw, scaler_y)

    test_idx = data.index[test_start_index:]

    return x_train_s, x_test_s, y_train_s, y_test_s, y_train, y_test, scaler_y, data.index, test_idx, y_train_s_combined, y_test_s_combined, y_test_comb_raw
def prepare_window_data(filepath, test_start_index):
    data = pd.read_csv(filepath, delimiter=';', parse_dates=['Date'], dayfirst=True)

    data['Month'] = data['Date'].dt.month
    data['Year'] = data['Date'].dt.year
    data['Month_sin'] = np.sin(2 * np.pi * data['Month'] / 12)
    data['Month_cos'] = np.cos(2 * np.pi * data['Month'] / 12)

    first_work_day = data[data['TEC_N_Aver'] > 0].index[0]
    data = data.loc[first_work_day:]

    if filepath == 'data/TEC22_Data.csv':
        data['B1_N_Aver'] = data['B1_N'] / 24
        data['B2_N_Aver'] = data['B2_N'] / 24
        data['B3_N_Aver'] = data['B3_N'] / 24
        data['B4_GT41_N_Aver'] = data['B4_GT41_N'] / 24
        data['B4_GT42_N_Aver'] = data['B4_GT42_N'] / 24
        data['B4_PT40_N_Aver'] = data['B4_N_Aver'] - data['B4_GT41_N_Aver'] - data['B4_GT42_N_Aver']

        data['B1_inWork'] = np.where(data['B1_N_Aver'] >= 125, 1, 0)
        data['B2_inWork'] = np.where(data['B2_N_Aver'] >= 125, 1, 0)
        data['B3_inWork'] = np.where(data['B3_N_Aver'] >= 125, 1, 0)
        data['B4_GT41_inWork'] = np.where(data['B4_GT41_N_Aver'] >= 50, 1, 0)
        data['B4_GT42_inWork'] = np.where(data['B4_GT42_N_Aver'] >= 50, 1, 0)
        data['B4_inWork'] = np.where(data['B4_N_Aver'] >= 76, 1, 0)

        data.loc[data['B1_N_Aver'] < 125, 'B1_N_Aver'] = 0
        data.loc[data['B2_N_Aver'] < 125, 'B2_N_Aver'] = 0
        data.loc[data['B3_N_Aver'] < 125, 'B3_N_Aver'] = 0
        data.loc[data['B4_GT41_N_Aver'] <= 50, 'B4_GT41_N_Aver'] = 0
        data.loc[data['B4_GT42_N_Aver'] <= 50, 'B4_GT42_N_Aver'] = 0
        data.loc[data['B4_N_Aver'] <= 76, 'B4_N_Aver'] = 0
        data.loc[data['TEC_N_Aver'] <= 50, 'TEC_N_Aver'] = 0

        data['B4_GT41_N_Aver'] = np.where(data['B4_N_Aver'] < 76, 0, data['B4_GT41_N_Aver'])
        data['B4_GT42_N_Aver'] = np.where(data['B4_N_Aver'] < 76, 0, data['B4_GT42_N_Aver'])

        data['B1_Available_Nmax'] = data['B1_inWork'] * 250
        data['B2_Available_Nmax'] = data['B2_inWork'] * 250
        data['B3_Available_Nmax'] = data['B3_inWork'] * 250
        data['B4_GT41_Available_Nmax'] = np.where(data['T'] <= -2.3, data['B4_GT41_inWork']*172.7, data['B4_GT41_inWork']*(-0.9484 * data['T'] + 170.78))
        data['B4_GT42_Available_Nmax'] = np.where(data['T'] <= -2.3, data['B4_GT42_inWork']*172.7, data['B4_GT42_inWork']*(-0.9484 * data['T'] + 170.78))
        gt41_conditions = [((data['Year'] >= 2024) & (data['B4_GT41_Available_Nmax'] > 160.0)), (data['Year'] < 2024), ((data['Year'] >= 2024) & (data['B4_GT41_Available_Nmax'] <= 160.0))]
        gt42_conditions = [((data['Year'] >= 2024) & (data['B4_GT42_Available_Nmax'] > 160.0)), (data['Year'] < 2024), ((data['Year'] >= 2024) & (data['B4_GT42_Available_Nmax'] <= 160.0))]
        gt41_choices = [160.0, (data['B4_GT41_Available_Nmax']), (data['B4_GT41_Available_Nmax'])]
        gt42_choices = [160.0, (data['B4_GT42_Available_Nmax']), (data['B4_GT42_Available_Nmax'])]
        data['B4_GT41_Available_Nmax'] = np.select(gt41_conditions , gt41_choices, 160)
        data['B4_GT42_Available_Nmax'] = np.select(gt42_conditions , gt42_choices, 160)
        data['B4_Available_Nmax'] = data['B4_GT41_Available_Nmax'] + data['B4_GT42_Available_Nmax'] + 72*(data['B4_GT41_inWork'] + data['B4_GT42_inWork'])
        data['TEC_Available_Nmax'] = data['B1_Available_Nmax'] + data['B2_Available_Nmax'] + data['B3_Available_Nmax'] + data['B4_Available_Nmax']

        data['B1_Available_Nmin'] = data['B1_inWork'] * 125
        data['B2_Available_Nmin'] = data['B2_inWork'] * 125
        data['B3_Available_Nmin'] = data['B3_inWork'] * 125
        data['B4_Available_Nmin'] = np.where(data['T'] <= -2.3,
                                             0.5 * data['B4_GT41_inWork'] * 172.7 + 0.5 * data[
                                                 'B4_GT42_inWork'] * 172.7 + 26 * (data['B4_GT41_inWork'] + data['B4_GT42_inWork']),
                                             0.5 * data['B4_GT41_inWork'] * (-0.9484 * data['T'] + 170.78) + 0.5 * data[
                                                 'B4_GT42_inWork'] * (-0.9484 * data['T'] + 170.78) + 26 * (data['B4_GT41_inWork'] + data['B4_GT42_inWork']))
        data['TEC_Available_Nmin'] = data['B1_Available_Nmin'] + data['B2_Available_Nmin'] + data['B3_Available_Nmin'] + \
                                     data['B4_Available_Nmin']

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
        data['B1_Available_Nmin_Prev'] = data['B1_Available_Nmin'].shift(1)
        data['B2_Available_Nmin_Prev'] = data['B2_Available_Nmin'].shift(1)
        data['B3_Available_Nmin_Prev'] = data['B3_Available_Nmin'].shift(1)
        data['B4_Available_Nmin_Prev'] = data['B4_Available_Nmin'].shift(1)
        data['TEC_Available_Nmin_Prev'] = data['TEC_Available_Nmin'].shift(1)
        data['B1_Available_Nmax_Prev'] = data['B1_Available_Nmax'].shift(1)
        data['B2_Available_Nmax_Prev'] = data['B2_Available_Nmax'].shift(1)
        data['B3_Available_Nmax_Prev'] = data['B3_Available_Nmax'].shift(1)
        data['B4_Available_Nmax_Prev'] = data['B4_Available_Nmax'].shift(1)
        data['TEC_Available_Nmax_Prev'] = data['TEC_Available_Nmax'].shift(1)
        data['TEC_N_Aver_Prev'] = data['TEC_N_Aver'].shift(1)

        data.dropna(inplace=True)

        x = data.loc[:,['T', 'Month_sin', 'Month_cos',
                    'B1_Available_Nmin', 'B2_Available_Nmin', 'B3_Available_Nmin', 'B4_Available_Nmin', 'TEC_Available_Nmin',
                    'B1_Available_Nmax', 'B2_Available_Nmax', 'B3_Available_Nmax', 'B4_Available_Nmax', 'TEC_Available_Nmax',
                    'T_Prev', 'Month_sin_Prev', 'Month_cos_Prev',
                    'B1_N_Aver_Prev', 'B2_N_Aver_Prev' ,'B3_N_Aver_Prev', 'B4_N_Aver_Prev', 'TEC_N_Aver_Prev',
                    'B1_Available_Nmin_Prev', 'B2_Available_Nmin_Prev', 'B3_Available_Nmin_Prev', 'B4_Available_Nmin_Prev', 'TEC_Available_Nmin_Prev',
                    'B1_Available_Nmax_Prev', 'B2_Available_Nmax_Prev', 'B3_Available_Nmax_Prev', 'B4_Available_Nmax_Prev', 'TEC_Available_Nmax_Prev']].values

        y = data.loc[:, ['TEC_N_Aver']].values

        y_true_combined = data.loc[:, ['TEC_N_Aver', 'TEC_Available_Nmax',
                                       'TEC_Available_Nmin']].values

    if filepath == 'data/TEC14_Data.csv':
        data['B1_GT11_N_Aver'] = data['B1_GT11_N'] / 24
        data['B1_GT12_N_Aver'] = data['B1_GT12_N'] / 24
        data['B1_PT10_N_Aver'] = data['B1_PT10_N'] / 24

        data['B2_GT21_N_Aver'] = data['B2_GT21_N'] / 24
        data['B2_GT22_N_Aver'] = data['B2_GT22_N'] / 24
        data['B2_PT20_N_Aver'] = data['B2_PT20_N'] / 24

        data['B1_GT11_inWork'] = np.where(data['B1_GT11_N_Aver'] >= 30, 1, 0)
        data['B1_GT12_inWork'] = np.where(data['B1_GT12_N_Aver'] >= 30, 1, 0)
        data['B2_GT21_inWork'] = np.where(data['B2_GT21_N_Aver'] >= 30, 1, 0)
        data['B2_GT22_inWork'] = np.where(data['B2_GT22_N_Aver'] >= 30, 1, 0)
        data['B1_inWork'] = np.where(data['B1_N_Aver'] > 45, 1, 0)
        data['B2_inWork'] = np.where(data['B2_N_Aver'] > 45, 1, 0)

        data.loc[data['B1_GT11_N_Aver'] < 30, 'B1_GT11_N_Aver'] = 0
        data.loc[data['B1_GT12_N_Aver'] < 30, 'B1_GT12_N_Aver'] = 0
        data.loc[data['B2_GT21_N_Aver'] < 30, 'B2_GT21_N_Aver'] = 0
        data.loc[data['B2_GT22_N_Aver'] < 30, 'B2_GT22_N_Aver'] = 0
        data.loc[data['B1_N_Aver'] < 45, 'B1_N_Aver'] = 0
        data.loc[data['B2_N_Aver'] < 45, 'B2_N_Aver'] = 0
        data.loc[data['TEC_N_Aver'] < 45, 'TEC_N_Aver'] = 0
        data['B1_GT11_N_Aver'] = np.where(data['B1_N_Aver'] < 45, 0,  data['B1_GT11_N_Aver'])
        data['B1_GT12_N_Aver'] = np.where(data['B1_N_Aver'] < 45, 0,  data['B1_GT12_N_Aver'])
        data['B2_GT21_N_Aver'] = np.where(data['B2_N_Aver'] < 45, 0,  data['B2_GT21_N_Aver'])
        data['B2_GT22_N_Aver'] = np.where(data['B2_N_Aver'] < 45, 0,  data['B2_GT22_N_Aver'])

        data['B1_Available_Nmax'] = (data['B1_GT11_inWork'] * 70 + data['B1_GT12_inWork'] * 70 + 26 * (
                data['B1_GT11_inWork'] + data['B1_GT12_inWork']))
        data['B2_Available_Nmax'] = (data['B2_GT21_inWork'] * 70 + data['B2_GT22_inWork'] * 70 + 26 * (
                data['B2_GT21_inWork'] + data['B2_GT22_inWork']))
        data['TEC_Available_Nmax'] = data['B1_Available_Nmax'] + data['B2_Available_Nmax']

        data['B1_Available_Nmin'] = (data['B1_GT11_inWork'] * 30 + data['B1_GT12_inWork'] * 30 + 15 * (
                data['B1_GT11_inWork'] + data['B1_GT12_inWork']))
        data['B2_Available_Nmin'] = (data['B2_GT21_inWork'] * 30 + data['B2_GT22_inWork'] * 30 + 15 * (
                data['B2_GT21_inWork'] + data['B2_GT22_inWork']))
        data['TEC_Available_Nmin'] = data['B1_Available_Nmin'] + data['B2_Available_Nmin']

        data['T_Prev'] = data['T'].shift(1)
        data['Month_sin_Prev'] = data['Month_sin'].shift(1)
        data['Month_cos_Prev'] = data['Month_cos'].shift(1)
        data['Year_Prev'] = data['Year'].shift(1)
        data['B1_N_Aver_Prev'] = data['B1_N_Aver'].shift(1)
        data['B2_N_Aver_Prev'] = data['B2_N_Aver'].shift(1)
        data['TEC_Available_Nmax_Prev'] = data['TEC_Available_Nmax'].shift(1)
        data['B1_Available_Nmax_Prev'] = data['B1_Available_Nmax'].shift(1)
        data['B2_Available_Nmax_Prev'] = data['B2_Available_Nmax'].shift(1)
        data['TEC_Available_Nmin_Prev'] = data['TEC_Available_Nmin'].shift(1)
        data['B1_Available_Nmin_Prev'] = data['B1_Available_Nmin'].shift(1)
        data['B2_Available_Nmin_Prev'] = data['B2_Available_Nmin'].shift(1)
        data['TEC_N_Aver_Prev'] = data['TEC_N_Aver'].shift(1)

        data.dropna(inplace=True)

        x = data.loc[:, ['T', 'Month_sin', 'Month_cos',
                         'B1_Available_Nmax', 'B2_Available_Nmax',
                         'B1_Available_Nmin', 'B2_Available_Nmin',
                         'TEC_Available_Nmax', 'TEC_Available_Nmin',
                         'T_Prev', 'Month_sin_Prev', 'Month_cos_Prev',
                         'B1_N_Aver_Prev', 'B2_N_Aver_Prev',
                         'B1_Available_Nmax_Prev', 'B2_Available_Nmax_Prev',
                         'B1_Available_Nmin_Prev', 'B2_Available_Nmin_Prev',
                         'TEC_Available_Nmax_Prev', 'TEC_Available_Nmin_Prev',
                         'TEC_N_Aver_Prev']].values
        y = data.loc[:, ['TEC_N_Aver']].values

        y_true_combined = data.loc[:, ['TEC_N_Aver', 'TEC_Available_Nmax',
                                       'TEC_Available_Nmin']].values

    data.set_index('Date', inplace=True)

    n = test_start_index
    x_train, x_test = x[:n], x[n:]
    y_train, y_test = y[:n], y[n:]
    y_train_comb_raw, y_test_comb_raw = y_true_combined[:n], y_true_combined[n:]

    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()

    x_train_s = scaler_x.fit_transform(x_train)
    x_test_s = scaler_x.transform(x_test)

    scaler_y.fit(y_train)
    y_train_s =scaler_y.transform(y_train)
    y_test_s = scaler_y.transform(y_test)

    y_train_s_combined = scale_combined(y_train_comb_raw, scaler_y)
    y_test_s_combined = scale_combined(y_test_comb_raw, scaler_y)

    test_idx = data.index[test_start_index:]

    return x_train_s, x_test_s, y_train_s, y_test_s, y_test, scaler_y, data.index, test_idx, y_train_s_combined, y_test_s_combined, y_test_comb_raw

def prepare_meta_data(results, filepath, test_idx, test_start_index, calc_goal, best_window_model_name, best_stat_model_name, best_step_model):
    data = pd.read_csv(filepath, delimiter=';', parse_dates=['Date'], dayfirst=True)

    data['TEC_N_Aver_pred'] = data['TEC_N_Aver']

    data['Month'] = data['Date'].dt.month
    data['Year'] = data['Date'].dt.year
    data['Month_sin'] = np.sin(2 * np.pi * data['Month'] / 12)
    data['Month_cos'] = np.cos(2 * np.pi * data['Month'] / 12)

    first_work_day = data[data['TEC_N_Aver'] > 0].index[0]
    data = data.loc[first_work_day:]

    if filepath == 'data/TEC22_Data.csv':
        data['B1_N_Aver'] = data['B1_N'] / 24
        data['B2_N_Aver'] = data['B2_N'] / 24
        data['B3_N_Aver'] = data['B3_N'] / 24
        data['B4_GT41_N_Aver'] = data['B4_GT41_N'] / 24
        data['B4_GT42_N_Aver'] = data['B4_GT42_N'] / 24
        data['B4_PT40_N_Aver'] = data['B4_N_Aver'] - data['B4_GT41_N_Aver'] - data['B4_GT42_N_Aver']

        data['B1_inWork'] = np.where(data['B1_N_Aver'] >= 125, 1, 0)
        data['B2_inWork'] = np.where(data['B2_N_Aver'] >= 125, 1, 0)
        data['B3_inWork'] = np.where(data['B3_N_Aver'] >= 125, 1, 0)
        data['B4_GT41_inWork'] = np.where(data['B4_GT41_N_Aver'] >= 50, 1, 0)
        data['B4_GT42_inWork'] = np.where(data['B4_GT42_N_Aver'] >= 50, 1, 0)
        data['B4_inWork'] = np.where(data['B4_N_Aver'] >= 76, 1, 0)

        data.loc[data['B1_N_Aver'] < 125, 'B1_N_Aver'] = 0
        data.loc[data['B2_N_Aver'] < 125, 'B2_N_Aver'] = 0
        data.loc[data['B3_N_Aver'] < 125, 'B3_N_Aver'] = 0
        data.loc[data['B4_GT41_N_Aver'] <= 50, 'B4_GT41_N_Aver'] = 0
        data.loc[data['B4_GT42_N_Aver'] <= 50, 'B4_GT42_N_Aver'] = 0
        data.loc[data['B4_N_Aver'] <= 76, 'B4_N_Aver'] = 0
        data.loc[data['TEC_N_Aver'] <= 50, 'TEC_N_Aver'] = 0

        data['B4_GT41_N_Aver'] = np.where(data['B4_N_Aver'] < 76, 0, data['B4_GT41_N_Aver'])
        data['B4_GT42_N_Aver'] = np.where(data['B4_N_Aver'] < 76, 0, data['B4_GT42_N_Aver'])

        data['B1_Available_Nmax'] = data['B1_inWork'] * 250
        data['B2_Available_Nmax'] = data['B2_inWork'] * 250
        data['B3_Available_Nmax'] = data['B3_inWork'] * 250
        data['B4_GT41_Available_Nmax'] = np.where(data['T'] <= -2.3, data['B4_GT41_inWork']*172.7, data['B4_GT41_inWork']*(-0.9484 * data['T'] + 170.78))
        data['B4_GT42_Available_Nmax'] = np.where(data['T'] <= -2.3, data['B4_GT42_inWork']*172.7, data['B4_GT42_inWork']*(-0.9484 * data['T'] + 170.78))
        gt41_conditions = [((data['Year'] >= 2024) & (data['B4_GT41_Available_Nmax'] > 160.0)), (data['Year'] < 2024), ((data['Year'] >= 2024) & (data['B4_GT41_Available_Nmax'] <= 160.0))]
        gt42_conditions = [((data['Year'] >= 2024) & (data['B4_GT42_Available_Nmax'] > 160.0)), (data['Year'] < 2024), ((data['Year'] >= 2024) & (data['B4_GT42_Available_Nmax'] <= 160.0))]
        gt41_choices = [160.0, (data['B4_GT41_Available_Nmax']), (data['B4_GT41_Available_Nmax'])]
        gt42_choices = [160.0, (data['B4_GT42_Available_Nmax']), (data['B4_GT42_Available_Nmax'])]
        data['B4_GT41_Available_Nmax'] = np.select(gt41_conditions , gt41_choices, 160)
        data['B4_GT42_Available_Nmax'] = np.select(gt42_conditions , gt42_choices, 160)
        data['B4_Available_Nmax'] = data['B4_GT41_Available_Nmax'] + data['B4_GT42_Available_Nmax'] + 72*(data['B4_GT41_inWork'] + data['B4_GT42_inWork'])
        data['TEC_Available_Nmax'] = data['B1_Available_Nmax'] + data['B2_Available_Nmax'] + data['B3_Available_Nmax'] + data['B4_Available_Nmax']

        data['B1_Available_Nmin'] = data['B1_inWork'] * 125
        data['B2_Available_Nmin'] = data['B2_inWork'] * 125
        data['B3_Available_Nmin'] = data['B3_inWork'] * 125
        data['B4_Available_Nmin'] = np.where(data['T'] <= -2.3,0.5 * data['B4_GT41_inWork'] * 172.7 + 0.5 * data['B4_GT42_inWork'] * 172.7 + 26 * (data['B4_GT41_inWork'] + data['B4_GT42_inWork']),
                                             0.5 * data['B4_GT41_inWork'] * (-0.9484 * data['T'] + 170.78) + 0.5 * data['B4_GT42_inWork'] * (-0.9484 * data['T'] + 170.78) + 26 * (data['B4_GT41_inWork'] + data['B4_GT42_inWork']))
        data['TEC_Available_Nmin'] = data['B1_Available_Nmin'] + data['B2_Available_Nmin'] + data['B3_Available_Nmin'] + data['B4_Available_Nmin']



    if filepath == 'data/TEC14_Data.csv':
        data['B1_GT11_N_Aver'] = data['B1_GT11_N'] / 24
        data['B1_GT12_N_Aver'] = data['B1_GT12_N'] / 24
        data['B1_PT10_N_Aver'] = data['B1_PT10_N'] / 24

        data['B2_GT21_N_Aver'] = data['B2_GT21_N'] / 24
        data['B2_GT22_N_Aver'] = data['B2_GT22_N'] / 24
        data['B2_PT20_N_Aver'] = data['B2_PT20_N'] / 24

        data['B1_GT11_inWork'] = np.where(data['B1_GT11_N_Aver'] >= 30, 1, 0)
        data['B1_GT12_inWork'] = np.where(data['B1_GT12_N_Aver'] >= 30, 1, 0)
        data['B2_GT21_inWork'] = np.where(data['B2_GT21_N_Aver'] >= 30, 1, 0)
        data['B2_GT22_inWork'] = np.where(data['B2_GT22_N_Aver'] >= 30, 1, 0)
        data['B1_inWork'] = np.where(data['B1_N_Aver'] >= 45, 1, 0)
        data['B2_inWork'] = np.where(data['B2_N_Aver'] >= 45, 1, 0)

        data.loc[data['B1_GT11_N_Aver'] < 30, 'B1_GT11_N_Aver'] = 0
        data.loc[data['B1_GT12_N_Aver'] < 30, 'B1_GT12_N_Aver'] = 0
        data.loc[data['B2_GT21_N_Aver'] < 30, 'B2_GT21_N_Aver'] = 0
        data.loc[data['B2_GT22_N_Aver'] < 30, 'B2_GT22_N_Aver'] = 0
        data.loc[data['B1_N_Aver'] < 45, 'B1_N_Aver'] = 0
        data.loc[data['B2_N_Aver'] < 45, 'B2_N_Aver'] = 0
        data.loc[data['TEC_N_Aver'] < 45, 'TEC_N_Aver'] = 0
        data['B1_GT11_N_Aver'] = np.where(data['B1_N_Aver'] < 45, 0,  data['B1_GT11_N_Aver'])
        data['B1_GT12_N_Aver'] = np.where(data['B1_N_Aver'] < 45, 0,  data['B1_GT12_N_Aver'])
        data['B2_GT21_N_Aver'] = np.where(data['B2_N_Aver'] < 45, 0,  data['B2_GT21_N_Aver'])
        data['B2_GT22_N_Aver'] = np.where(data['B2_N_Aver'] < 45, 0,  data['B2_GT22_N_Aver'])

        data['B1_Available_Nmax'] = (data['B1_GT11_inWork'] * 70 + data['B1_GT12_inWork'] * 70 + 26 * (
                    data['B1_GT11_inWork'] + data['B1_GT12_inWork']))
        data['B2_Available_Nmax'] = (data['B2_GT21_inWork'] * 70 + data['B2_GT22_inWork'] * 70 + 26 * (
                    data['B2_GT21_inWork'] + data['B2_GT22_inWork']))
        data['TEC_Available_Nmax'] = data['B1_Available_Nmax'] + data['B2_Available_Nmax']

        data['B1_Available_Nmin'] = (data['B1_GT11_inWork'] * 30 + data['B1_GT12_inWork'] * 30 + 15 * (
                    data['B1_GT11_inWork'] + data['B1_GT12_inWork']))
        data['B2_Available_Nmin'] = (data['B2_GT21_inWork'] * 30 + data['B2_GT22_inWork'] * 30 + 15 * (
                    data['B2_GT21_inWork'] + data['B2_GT22_inWork']))
        data['TEC_Available_Nmin'] = data['B1_Available_Nmin'] + data['B2_Available_Nmin']

    data.dropna(inplace=True)

    data.set_index('Date', inplace=True)

    first_pred_value = results[best_window_model_name].flatten()[0]
    first_test_index = test_idx[0]
    data.loc[first_test_index, 'TEC_N_Aver_pred'] = first_pred_value
    data.loc[test_idx[1:15], 'TEC_N_Aver_pred'] = best_step_model[0][:14]

    remaining_indices = test_idx[15:]
    remaining_preds = results[best_stat_model_name].flatten()[-len(remaining_indices):]
    data.loc[remaining_indices, 'TEC_N_Aver_pred'] = remaining_preds

    features = ['T', 'Month_sin', 'Month_cos', 'TEC_N_Aver_pred']
    prefixes = goal_mapping[filepath]
    for prefix in prefixes:
        features.append(f'{prefix}_Available_Nmax')
        features.append(f'{prefix}_Available_Nmin')
    x = data.loc[:, features].values
    y = data.loc[:, [calc_goal+'_N_Aver']].values

    y_true_combined = data.loc[:, [calc_goal + '_N_Aver', calc_goal + '_Available_Nmax',
                                   calc_goal + '_Available_Nmin']].values

    n = test_start_index
    x_train, x_test = x[:n], x[n:]
    y_train, y_test = y[:n], y[n:]
    y_train_comb_raw, y_test_comb_raw = y_true_combined[:n], y_true_combined[n:]

    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()

    x_train_s = scaler_x.fit_transform(x_train)
    x_test_s = scaler_x.transform(x_test)

    scaler_y.fit(y_train)
    y_train_s =scaler_y.transform(y_train)
    y_test_s = scaler_y.transform(y_test)

    y_train_s_combined = scale_combined(y_train_comb_raw, scaler_y)
    y_test_s_combined = scale_combined(y_test_comb_raw, scaler_y)

    test_idx = data.index[test_start_index:]

    return x_train_s, x_test_s, y_train_s, y_test_s, y_train, y_test, scaler_y, data.index, test_idx, y_train_s_combined, y_test_s_combined, y_test_comb_raw
def prepare_meta_window_data(results, filepath, test_idx, test_start_index, calc_goal, best_window_model_name, best_stat_model_name, best_step_model):
    data = pd.read_csv(filepath, delimiter=';', parse_dates=['Date'], dayfirst=True)

    data['TEC_N_Aver_pred'] = data['TEC_N_Aver']

    data['Month'] = data['Date'].dt.month
    data['Year'] = data['Date'].dt.year
    data['Month_sin'] = np.sin(2 * np.pi * data['Month'] / 12)
    data['Month_cos'] = np.cos(2 * np.pi * data['Month'] / 12)

    first_work_day = data[data['TEC_N_Aver'] > 0].index[0]
    data = data.loc[first_work_day:]

    if filepath == 'data/TEC22_Data.csv':
        data['B1_N_Aver'] = data['B1_N'] / 24
        data['B2_N_Aver'] = data['B2_N'] / 24
        data['B3_N_Aver'] = data['B3_N'] / 24
        data['B4_GT41_N_Aver'] = data['B4_GT41_N'] / 24
        data['B4_GT42_N_Aver'] = data['B4_GT42_N'] / 24
        data['B4_PT40_N_Aver'] = data['B4_N_Aver'] - data['B4_GT41_N_Aver'] - data['B4_GT42_N_Aver']

        data['B1_inWork'] = np.where(data['B1_N_Aver'] >= 125, 1, 0)
        data['B2_inWork'] = np.where(data['B2_N_Aver'] >= 125, 1, 0)
        data['B3_inWork'] = np.where(data['B3_N_Aver'] >= 125, 1, 0)
        data['B4_GT41_inWork'] = np.where(data['B4_GT41_N_Aver'] >= 50, 1, 0)
        data['B4_GT42_inWork'] = np.where(data['B4_GT42_N_Aver'] >= 50, 1, 0)
        data['B4_inWork'] = np.where(data['B4_N_Aver'] >= 76, 1, 0)

        data.loc[data['B1_N_Aver'] < 125, 'B1_N_Aver'] = 0
        data.loc[data['B2_N_Aver'] < 125, 'B2_N_Aver'] = 0
        data.loc[data['B3_N_Aver'] < 125, 'B3_N_Aver'] = 0
        data.loc[data['B4_GT41_N_Aver'] <= 50, 'B4_GT41_N_Aver'] = 0
        data.loc[data['B4_GT42_N_Aver'] <= 50, 'B4_GT42_N_Aver'] = 0
        data.loc[data['B4_N_Aver'] <= 76, 'B4_N_Aver'] = 0
        data.loc[data['TEC_N_Aver'] <= 50, 'TEC_N_Aver'] = 0

        data['B4_GT41_N_Aver'] = np.where(data['B4_N_Aver'] < 76, 0, data['B4_GT41_N_Aver'])
        data['B4_GT42_N_Aver'] = np.where(data['B4_N_Aver'] < 76, 0, data['B4_GT42_N_Aver'])

        data['B1_Available_Nmax'] = data['B1_inWork'] * 250
        data['B2_Available_Nmax'] = data['B2_inWork'] * 250
        data['B3_Available_Nmax'] = data['B3_inWork'] * 250
        data['B4_GT41_Available_Nmax'] = np.where(data['T'] <= -2.3, data['B4_GT41_inWork']*172.7, data['B4_GT41_inWork']*(-0.9484 * data['T'] + 170.78))
        data['B4_GT42_Available_Nmax'] = np.where(data['T'] <= -2.3, data['B4_GT42_inWork']*172.7, data['B4_GT42_inWork']*(-0.9484 * data['T'] + 170.78))
        gt41_conditions = [((data['Year'] >= 2024) & (data['B4_GT41_Available_Nmax'] > 160.0)), (data['Year'] < 2024), ((data['Year'] >= 2024) & (data['B4_GT41_Available_Nmax'] <= 160.0))]
        gt42_conditions = [((data['Year'] >= 2024) & (data['B4_GT42_Available_Nmax'] > 160.0)), (data['Year'] < 2024), ((data['Year'] >= 2024) & (data['B4_GT42_Available_Nmax'] <= 160.0))]
        gt41_choices = [160.0, (data['B4_GT41_Available_Nmax']), (data['B4_GT41_Available_Nmax'])]
        gt42_choices = [160.0, (data['B4_GT42_Available_Nmax']), (data['B4_GT42_Available_Nmax'])]
        data['B4_GT41_Available_Nmax'] = np.select(gt41_conditions , gt41_choices, 160)
        data['B4_GT42_Available_Nmax'] = np.select(gt42_conditions , gt42_choices, 160)
        data['B4_Available_Nmax'] = data['B4_GT41_Available_Nmax'] + data['B4_GT42_Available_Nmax'] + 72*(data['B4_GT41_inWork'] + data['B4_GT42_inWork'])
        data['TEC_Available_Nmax'] = data['B1_Available_Nmax'] + data['B2_Available_Nmax'] + data['B3_Available_Nmax'] + \
                                     data['B4_Available_Nmax']

        data['B1_Available_Nmin'] = data['B1_inWork'] * 125
        data['B2_Available_Nmin'] = data['B2_inWork'] * 125
        data['B3_Available_Nmin'] = data['B3_inWork'] * 125
        data['B4_Available_Nmin'] = np.where(data['T'] <= -2.3,
                                             0.5 * data['B4_GT41_inWork'] * 172.7 + 0.5 * data['B4_GT42_inWork'] * 172.7 + 26 * (data['B4_GT41_inWork'] + data['B4_GT42_inWork']),
                                             0.5 * data['B4_GT41_inWork'] * (-0.9484 * data['T'] + 170.78) + 0.5 * data['B4_GT42_inWork'] * (-0.9484 * data['T'] + 170.78) + 26 * (data['B4_GT41_inWork'] + data['B4_GT42_inWork']))
        data['TEC_Available_Nmin'] = data['B1_Available_Nmin'] + data['B2_Available_Nmin'] + data['B3_Available_Nmin'] + data['B4_Available_Nmin']

        data.dropna(inplace=True)

        data.set_index('Date', inplace=True)

        first_pred_value = results[best_window_model_name].flatten()[0]
        first_test_index = test_idx[0]
        data.loc[first_test_index, 'TEC_N_Aver_pred'] = first_pred_value
        data.loc[test_idx[1:15], 'TEC_N_Aver_pred'] = best_step_model[0][:14]

        remaining_indices = test_idx[15:]
        remaining_preds = results[best_stat_model_name].flatten()[-len(remaining_indices):]
        data.loc[remaining_indices, 'TEC_N_Aver_pred'] = remaining_preds

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
        data['B1_Available_Nmin_Prev'] = data['B1_Available_Nmin'].shift(1)
        data['B2_Available_Nmin_Prev'] = data['B2_Available_Nmin'].shift(1)
        data['B3_Available_Nmin_Prev'] = data['B3_Available_Nmin'].shift(1)
        data['B4_Available_Nmin_Prev'] = data['B4_Available_Nmin'].shift(1)
        data['TEC_Available_Nmin_Prev'] = data['TEC_Available_Nmin'].shift(1)
        data['B1_Available_Nmax_Prev'] = data['B1_Available_Nmax'].shift(1)
        data['B2_Available_Nmax_Prev'] = data['B2_Available_Nmax'].shift(1)
        data['B3_Available_Nmax_Prev'] = data['B3_Available_Nmax'].shift(1)
        data['B4_Available_Nmax_Prev'] = data['B4_Available_Nmax'].shift(1)
        data['TEC_Available_Nmax_Prev'] = data['TEC_Available_Nmax'].shift(1)
        data['TEC_N_Aver_Prev'] = data['TEC_N_Aver'].shift(1)

    if filepath == 'data/TEC14_Data.csv':
        data['B1_GT11_N_Aver'] = data['B1_GT11_N'] / 24
        data['B1_GT12_N_Aver'] = data['B1_GT12_N'] / 24
        data['B1_PT10_N_Aver'] = data['B1_PT10_N'] / 24

        data['B2_GT21_N_Aver'] = data['B2_GT21_N'] / 24
        data['B2_GT22_N_Aver'] = data['B2_GT22_N'] / 24
        data['B2_PT20_N_Aver'] = data['B2_PT20_N'] / 24

        data['B1_GT11_inWork'] = np.where(data['B1_GT11_N_Aver'] >= 30, 1, 0)
        data['B1_GT12_inWork'] = np.where(data['B1_GT12_N_Aver'] >= 30, 1, 0)
        data['B2_GT21_inWork'] = np.where(data['B2_GT21_N_Aver'] >= 30, 1, 0)
        data['B2_GT22_inWork'] = np.where(data['B2_GT22_N_Aver'] >= 30, 1, 0)
        data['B1_inWork'] = np.where(data['B1_N_Aver'] >= 45, 1, 0)
        data['B2_inWork'] = np.where(data['B2_N_Aver'] >= 45, 1, 0)

        data.loc[data['B1_GT11_N_Aver'] < 30, 'B1_GT11_N_Aver'] = 0
        data.loc[data['B1_GT12_N_Aver'] < 30, 'B1_GT12_N_Aver'] = 0
        data.loc[data['B2_GT21_N_Aver'] < 30, 'B2_GT21_N_Aver'] = 0
        data.loc[data['B2_GT22_N_Aver'] < 30, 'B2_GT22_N_Aver'] = 0
        data.loc[data['B1_N_Aver'] < 45, 'B1_N_Aver'] = 0
        data.loc[data['B2_N_Aver'] < 45, 'B2_N_Aver'] = 0
        data.loc[data['TEC_N_Aver'] < 45, 'TEC_N_Aver'] = 0
        data['B1_GT11_N_Aver'] = np.where(data['B1_N_Aver'] < 45, 0,  data['B1_GT11_N_Aver'])
        data['B1_GT12_N_Aver'] = np.where(data['B1_N_Aver'] < 45, 0,  data['B1_GT12_N_Aver'])
        data['B2_GT21_N_Aver'] = np.where(data['B2_N_Aver'] < 45, 0,  data['B2_GT21_N_Aver'])
        data['B2_GT22_N_Aver'] = np.where(data['B2_N_Aver'] < 45, 0,  data['B2_GT22_N_Aver'])

        data['B1_Available_Nmax'] = (data['B1_GT11_inWork'] * (2 / 1000000000 * data['T']**6 + 7 / 1000000000 * data['T']**5 - 4 / 1000000 * data['T']**4 - 0.0001 * data['T']**3 + 0.0006 * data['T']**2 - 0.2475 * data['T'] + 69.719)
                                     + data['B1_GT12_inWork'] * (2 / 1000000000 * data['T']**6 + 7 / 1000000000 * data['T']**5 - 4 / 1000000 * data['T']**4 - 0.0001 * data['T']**3 + 0.0006 * data['T']**2 - 0.2475 * data['T'] + 69.719)
                                     + 26 * (data['B1_GT11_inWork'] + data['B1_GT12_inWork']))
        data['B2_Available_Nmax'] = (data['B2_GT21_inWork'] * (2 / 1000000000 * data['T']**6 + 7 / 1000000000 * data['T']**5 - 4 / 1000000 * data['T']**4 - 0.0001 * data['T']**3 + 0.0006 * data['T']**2 - 0.2475 * data['T'] + 69.719)
                                     + data['B2_GT22_inWork'] * (2 / 1000000000 * data['T']**6 + 7 / 1000000000 * data['T']**5 - 4 / 1000000 * data['T']**4 - 0.0001 * data['T']**3 + 0.0006 * data['T']**2 - 0.2475 * data['T'] + 69.719)
                                     + 26 * (data['B2_GT21_inWork'] + data['B2_GT22_inWork']))
        data['TEC_Available_Nmax'] = data['B1_Available_Nmax'] + data['B2_Available_Nmax']

        data['B1_Available_Nmin'] = (data['B1_GT11_inWork'] * 0.4*(2 / 1000000000 * data['T']**6 + 7 / 1000000000 * data['T']**5 - 4 / 1000000 * data['T']**4 - 0.0001 * data['T']**3 + 0.0006 * data['T']**2 - 0.2475 * data['T'] + 69.719)
                                     + data['B1_GT12_inWork'] * 0.4*(2 / 1000000000 * data['T']**6 + 7 / 1000000000 * data['T']**5 - 4 / 1000000 * data['T']**4 - 0.0001 * data['T']**3 + 0.0006 * data['T']**2 - 0.2475 * data['T'] + 69.719)
                                     + 15 * ( data['B1_GT11_inWork'] + data['B1_GT12_inWork']))
        data['B2_Available_Nmin'] = (data['B2_GT21_inWork'] * 0.4*(2 / 1000000000 * data['T']**6 + 7 / 1000000000 * data['T']**5 - 4 / 1000000 * data['T']**4 - 0.0001 * data['T']**3 + 0.0006 * data['T']**2 - 0.2475 * data['T'] + 69.719)
                                     + data['B2_GT22_inWork'] * 0.4*(2 / 1000000000 * data['T']**6 + 7 / 1000000000 * data['T']**5 - 4 / 1000000 * data['T']**4 - 0.0001 * data['T']**3 + 0.0006 * data['T']**2 - 0.2475 * data['T'] + 69.719)
                                     + 15 * (data['B2_GT21_inWork'] + data['B2_GT22_inWork']))
        data['TEC_Available_Nmin'] = data['B1_Available_Nmin'] + data['B2_Available_Nmin']

        data.dropna(inplace=True)

        data.set_index('Date', inplace=True)

        first_pred_value = results[best_window_model_name].flatten()[0]
        first_test_index = test_idx[0]
        data.loc[first_test_index, 'TEC_N_Aver_pred'] = first_pred_value
        data.loc[test_idx[1:15], 'TEC_N_Aver_pred'] = best_step_model[0][:14]

        remaining_indices = test_idx[15:]
        remaining_preds = results[best_stat_model_name].flatten()[-len(remaining_indices):]
        data.loc[remaining_indices, 'TEC_N_Aver_pred'] = remaining_preds

        data['T_Prev'] = data['T'].shift(1)
        data['Month_sin_Prev'] = data['Month_sin'].shift(1)
        data['Month_cos_Prev'] = data['Month_cos'].shift(1)
        data['Year_Prev'] = data['Year'].shift(1)
        data['B1_N_Aver_Prev'] = data['B1_N_Aver'].shift(1)
        data['B2_N_Aver_Prev'] = data['B2_N_Aver'].shift(1)
        data['B1_Available_Nmax_Prev'] = data['B1_Available_Nmax'].shift(1)
        data['B2_Available_Nmax_Prev'] = data['B2_Available_Nmax'].shift(1)
        data['B1_Available_Nmin_Prev'] = data['B1_Available_Nmin'].shift(1)
        data['B2_Available_Nmin_Prev'] = data['B2_Available_Nmin'].shift(1)
        data['TEC_Available_Nmax_Prev'] = data['TEC_Available_Nmax'].shift(1)
        data['TEC_Available_Nmin_Prev'] = data['TEC_Available_Nmin'].shift(1)
        data['TEC_N_Aver_Prev'] = data['TEC_N_Aver'].shift(1)

    data.dropna(inplace=True)

    features = ['T', 'Month_sin', 'Month_cos', 'TEC_N_Aver_pred',
     'T_Prev', 'Month_sin_Prev', 'Month_cos_Prev','TEC_N_Aver_Prev']
    prefixes = goal_mapping[filepath]
    for prefix in prefixes:
        features.append(f'{prefix}_Available_Nmax')
        features.append(f'{prefix}_Available_Nmin')
        features.append(f'{prefix}_Available_Nmax_Prev')
        features.append(f'{prefix}_Available_Nmin_Prev')
    x = data.loc[:, features].values
    y = data.loc[:, [calc_goal+'_N_Aver']].values

    y_true_combined = data.loc[:, [calc_goal + '_N_Aver', calc_goal + '_Available_Nmax',
                                   calc_goal + '_Available_Nmin']].values

    n = test_start_index
    x_train, x_test = x[:n], x[n:]
    y_train, y_test = y[:n], y[n:]
    y_train_comb_raw, y_test_comb_raw = y_true_combined[:n], y_true_combined[n:]

    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()

    x_train_s = scaler_x.fit_transform(x_train)
    x_test_s = scaler_x.transform(x_test)

    scaler_y.fit(y_train)
    y_train_s =scaler_y.transform(y_train)
    y_test_s = scaler_y.transform(y_test)

    y_train_s_combined = scale_combined(y_train_comb_raw, scaler_y)
    y_test_s_combined = scale_combined(y_test_comb_raw, scaler_y)

    test_idx = data.index[test_start_index:]
    return x_train_s, x_test_s, y_train_s, y_test_s, y_test, scaler_y, data.index, test_idx, y_train_s_combined, y_test_s_combined, y_test_comb_raw

def prepare_step_data(filepath, test_start_index, n_out):
    data = pd.read_csv(filepath, delimiter=';', parse_dates=['Date'], dayfirst=True)

    data['Month'] = data['Date'].dt.month
    data['Year'] = data['Date'].dt.year
    data['Month_sin'] = np.sin(2 * np.pi * data['Month'] / 12)
    data['Month_cos'] = np.cos(2 * np.pi * data['Month'] / 12)

    first_work_day = data[data['TEC_N_Aver'] > 0].index[0]
    data = data.loc[first_work_day:]

    if filepath == 'data/TEC22_Data.csv':
        data['B1_N_Aver'] = data['B1_N'] / 24
        data['B2_N_Aver'] = data['B2_N'] / 24
        data['B3_N_Aver'] = data['B3_N'] / 24
        data['B4_GT41_N_Aver'] = data['B4_GT41_N'] / 24
        data['B4_GT42_N_Aver'] = data['B4_GT42_N'] / 24
        data['B4_PT40_N_Aver'] = data['B4_N_Aver'] - data['B4_GT41_N_Aver'] - data['B4_GT42_N_Aver']

        data['B1_inWork'] = np.where(data['B1_N_Aver'] >= 125, 1, 0)
        data['B2_inWork'] = np.where(data['B2_N_Aver'] >= 125, 1, 0)
        data['B3_inWork'] = np.where(data['B3_N_Aver'] >= 125, 1, 0)
        data['B4_GT41_inWork'] = np.where(data['B4_GT41_N_Aver'] >= 50, 1, 0)
        data['B4_GT42_inWork'] = np.where(data['B4_GT42_N_Aver'] >= 50, 1, 0)
        data['B4_inWork'] = np.where(data['B4_N_Aver'] >= 76, 1, 0)

        data.loc[data['B1_N_Aver'] < 125, 'B1_N_Aver'] = 0
        data.loc[data['B2_N_Aver'] < 125, 'B2_N_Aver'] = 0
        data.loc[data['B3_N_Aver'] < 125, 'B3_N_Aver'] = 0
        data.loc[data['B4_GT41_N_Aver'] <= 50, 'B4_GT41_N_Aver'] = 0
        data.loc[data['B4_GT42_N_Aver'] <= 50, 'B4_GT42_N_Aver'] = 0
        data.loc[data['B4_N_Aver'] <= 76, 'B4_N_Aver'] = 0
        data.loc[data['TEC_N_Aver'] <= 50, 'TEC_N_Aver'] = 0

        data['B4_GT41_N_Aver'] = np.where(data['B4_N_Aver'] < 76, 0, data['B4_GT41_N_Aver'])
        data['B4_GT42_N_Aver'] = np.where(data['B4_N_Aver'] < 76, 0, data['B4_GT42_N_Aver'])

        data['B1_Available_Nmax'] = data['B1_inWork'] * 250
        data['B2_Available_Nmax'] = data['B2_inWork'] * 250
        data['B3_Available_Nmax'] = data['B3_inWork'] * 250
        data['B4_GT41_Available_Nmax'] = np.where(data['T'] <= -2.3, data['B4_GT41_inWork']*172.7, data['B4_GT41_inWork']*(-0.9484 * data['T'] + 170.78))
        data['B4_GT42_Available_Nmax'] = np.where(data['T'] <= -2.3, data['B4_GT42_inWork']*172.7, data['B4_GT42_inWork']*(-0.9484 * data['T'] + 170.78))
        gt41_conditions = [((data['Year'] >= 2024) & (data['B4_GT41_Available_Nmax'] > 160.0)), (data['Year'] < 2024), ((data['Year'] >= 2024) & (data['B4_GT41_Available_Nmax'] <= 160.0))]
        gt42_conditions = [((data['Year'] >= 2024) & (data['B4_GT42_Available_Nmax'] > 160.0)), (data['Year'] < 2024), ((data['Year'] >= 2024) & (data['B4_GT42_Available_Nmax'] <= 160.0))]
        gt41_choices = [160.0, (data['B4_GT41_Available_Nmax']), (data['B4_GT41_Available_Nmax'])]
        gt42_choices = [160.0, (data['B4_GT42_Available_Nmax']), (data['B4_GT42_Available_Nmax'])]
        data['B4_GT41_Available_Nmax'] = np.select(gt41_conditions , gt41_choices, 160)
        data['B4_GT42_Available_Nmax'] = np.select(gt42_conditions , gt42_choices, 160)
        data['B4_Available_Nmax'] = data['B4_GT41_Available_Nmax'] + data['B4_GT42_Available_Nmax'] + 72*(data['B4_GT41_inWork'] + data['B4_GT42_inWork'])
        data['B4_Available_Nmax'] = data['B4_GT41_Available_Nmax'] + data['B4_GT42_Available_Nmax'] + 72 * (
                    data['B4_GT41_inWork'] + data['B4_GT42_inWork'])
        data['TEC_Available_Nmax'] = data['B1_Available_Nmax'] + data['B2_Available_Nmax'] + data['B3_Available_Nmax'] + \
                                     data['B4_Available_Nmax']

        data['B1_Available_Nmin'] = data['B1_inWork'] * 125
        data['B2_Available_Nmin'] = data['B2_inWork'] * 125
        data['B3_Available_Nmin'] = data['B3_inWork'] * 125
        data['B4_Available_Nmin'] = np.where(data['T'] <= -2.3,
                                             0.5 * data['B4_GT41_inWork'] * 172.7 + 0.5 * data['B4_GT42_inWork'] * 172.7 + 26 * (data['B4_GT41_inWork'] + data['B4_GT42_inWork']),
                                             0.5 * data['B4_GT41_inWork'] * (-0.9484 * data['T'] + 170.78) + 0.5 * data['B4_GT42_inWork'] * (-0.9484 * data['T'] + 170.78) + 26 * (data['B4_GT41_inWork'] + data['B4_GT42_inWork']))
        data['TEC_Available_Nmin'] = data['B1_Available_Nmin'] + data['B2_Available_Nmin'] + data['B3_Available_Nmin'] + \
                                     data['B4_Available_Nmin']

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
        data['B1_Available_Nmin_Prev'] = data['B1_Available_Nmin'].shift(1)
        data['B2_Available_Nmin_Prev'] = data['B2_Available_Nmin'].shift(1)
        data['B3_Available_Nmin_Prev'] = data['B3_Available_Nmin'].shift(1)
        data['B4_Available_Nmin_Prev'] = data['B4_Available_Nmin'].shift(1)
        data['TEC_Available_Nmin_Prev'] = data['TEC_Available_Nmin'].shift(1)
        data['B1_Available_Nmax_Prev'] = data['B1_Available_Nmax'].shift(1)
        data['B2_Available_Nmax_Prev'] = data['B2_Available_Nmax'].shift(1)
        data['B3_Available_Nmax_Prev'] = data['B3_Available_Nmax'].shift(1)
        data['B4_Available_Nmax_Prev'] = data['B4_Available_Nmax'].shift(1)
        data['TEC_Available_Nmax_Prev'] = data['TEC_Available_Nmax'].shift(1)
        data['TEC_N_Aver_Prev'] = data['TEC_N_Aver'].shift(1)

        is_half_block = (data['B4_GT41_inWork'] + data['B4_GT42_inWork'] == 1)
        data['sample_weight'] = np.where(is_half_block, 2.0, 1.0)

        data.set_index('Date', inplace=True)

    if filepath == 'data/TEC14_Data.csv':
        data['B1_GT11_N_Aver'] = data['B1_GT11_N'] / 24
        data['B1_GT12_N_Aver'] = data['B1_GT12_N'] / 24
        data['B1_PT10_N_Aver'] = data['B1_PT10_N'] / 24

        data['B2_GT21_N_Aver'] = data['B2_GT21_N'] / 24
        data['B2_GT22_N_Aver'] = data['B2_GT22_N'] / 24
        data['B2_PT20_N_Aver'] = data['B2_PT20_N'] / 24

        data['B1_GT11_inWork'] = np.where(data['B1_GT11_N_Aver'] >= 30, 1, 0)
        data['B1_GT12_inWork'] = np.where(data['B1_GT12_N_Aver'] >= 30, 1, 0)
        data['B2_GT21_inWork'] = np.where(data['B2_GT21_N_Aver'] >= 30, 1, 0)
        data['B2_GT22_inWork'] = np.where(data['B2_GT22_N_Aver'] >= 30, 1, 0)
        data['B1_inWork'] = np.where(data['B1_N_Aver'] >= 45, 1, 0)
        data['B2_inWork'] = np.where(data['B2_N_Aver'] >= 45, 1, 0)

        data.loc[data['B1_GT11_N_Aver'] < 30, 'B1_GT11_N_Aver'] = 0
        data.loc[data['B1_GT12_N_Aver'] < 30, 'B1_GT12_N_Aver'] = 0
        data.loc[data['B2_GT21_N_Aver'] < 30, 'B2_GT21_N_Aver'] = 0
        data.loc[data['B2_GT22_N_Aver'] < 30, 'B2_GT22_N_Aver'] = 0
        data.loc[data['B1_N_Aver'] < 45, 'B1_N_Aver'] = 0
        data.loc[data['B2_N_Aver'] < 45, 'B2_N_Aver'] = 0
        data.loc[data['TEC_N_Aver'] < 45, 'TEC_N_Aver'] = 0
        data['B1_GT11_N_Aver'] = np.where(data['B1_N_Aver'] < 45, 0,  data['B1_GT11_N_Aver'])
        data['B1_GT12_N_Aver'] = np.where(data['B1_N_Aver'] < 45, 0,  data['B1_GT12_N_Aver'])
        data['B2_GT21_N_Aver'] = np.where(data['B2_N_Aver'] < 45, 0,  data['B2_GT21_N_Aver'])
        data['B2_GT22_N_Aver'] = np.where(data['B2_N_Aver'] < 45, 0,  data['B2_GT22_N_Aver'])

        data['B1_Available_Nmax'] = (data['B1_GT11_inWork'] * 70 + data['B1_GT12_inWork'] * 70 + 26 * (
                    data['B1_GT11_inWork'] + data['B1_GT12_inWork']))
        data['B2_Available_Nmax'] = (data['B2_GT21_inWork'] * 70 + data['B2_GT22_inWork'] * 70 + 26 * (
                    data['B2_GT21_inWork'] + data['B2_GT22_inWork']))
        data['TEC_Available_Nmax'] = data['B1_Available_Nmax'] + data['B2_Available_Nmax']

        data['B1_Available_Nmin'] = (data['B1_GT11_inWork'] * 30 + data['B1_GT12_inWork'] * 30 + 15 * (
                    data['B1_GT11_inWork'] + data['B1_GT12_inWork']))
        data['B2_Available_Nmin'] = (data['B2_GT21_inWork'] * 30 + data['B2_GT22_inWork'] * 30 + 15 * (
                    data['B2_GT21_inWork'] + data['B2_GT22_inWork']))
        data['TEC_Available_Nmin'] = data['B1_Available_Nmin'] + data['B2_Available_Nmin']

        data['T_Prev'] = data['T'].shift(1)
        data['Month_sin_Prev'] = data['Month_sin'].shift(1)
        data['Month_cos_Prev'] = data['Month_cos'].shift(1)
        data['Year_Prev'] = data['Year'].shift(1)
        data['B1_N_Aver_Prev'] = data['B1_N_Aver'].shift(1)
        data['B2_N_Aver_Prev'] = data['B2_N_Aver'].shift(1)
        data['B1_Available_Nmin_Prev'] = data['B1_Available_Nmin'].shift(1)
        data['B2_Available_Nmin_Prev'] = data['B2_Available_Nmin'].shift(1)
        data['B1_Available_Nmax_Prev'] = data['B1_Available_Nmax'].shift(1)
        data['B2_Available_Nmax_Prev'] = data['B2_Available_Nmax'].shift(1)
        data['TEC_Available_Nmin_Prev'] = data['TEC_Available_Nmin'].shift(1)
        data['TEC_Available_Nmax_Prev'] = data['TEC_Available_Nmax'].shift(1)
        data['TEC_N_Aver_Prev'] = data['TEC_N_Aver'].shift(1)

        data['sample_weight'] = 1.0

        data.set_index('Date', inplace=True)

    base_features = ['T', 'Month_sin', 'Month_cos', 'T_Prev', 'TEC_N_Aver_Prev']
    prefixes = goal_mapping[filepath]
    for prefix in prefixes:
        base_features.append(f'{prefix}_Available_Nmax')
        base_features.append(f'{prefix}_Available_Nmin')
        base_features.append(f'{prefix}_Available_Nmax_Prev')
        base_features.append(f'{prefix}_Available_Nmin_Prev')

    df = DataFrame(data)
    cols, names, agg = list(), list(), list()
    for i in range(1, n_out + 1):
        if filepath == 'data/TEC22_Data.csv':
            weights = df['sample_weight'].values
            weights_train = weights[:test_start_index]

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
            cols.append(df[['B1_Available_Nmin']].shift(-i))
            cols.append(df[['B2_Available_Nmin']].shift(-i))
            cols.append(df[['B3_Available_Nmin']].shift(-i))
            cols.append(df[['B4_Available_Nmin']].shift(-i))
            cols.append(df[['TEC_Available_Nmin']].shift(-i))
            cols.append(df[['B1_Available_Nmax']].shift(-i))
            cols.append(df[['B2_Available_Nmax']].shift(-i))
            cols.append(df[['B3_Available_Nmax']].shift(-i))
            cols.append(df[['B4_Available_Nmax']].shift(-i))
            cols.append(df[['TEC_Available_Nmax']].shift(-i))
            names += [f'T_lag{i}', f'Month_sin_lag{i}', f'Month_cos_lag{i}', f'Year_lag{i}',
                      f'TEC_N_Aver_lag{i}',
                      f'B1_inWork_lag{i}', f'B2_inWork_lag{i}', f'B3_inWork_lag{i}',
                      f'B4_GT41_inWork_lag{i}', f'B4_GT42_inWork_lag{i}', f'B4_inWork_lag{i}',
                      f'B1_Available_Nmin_lag{i}', f'B2_Available_Nmin_lag{i}', f'B3_Available_Nmin_lag{i}',
                      f'B4_Available_Nmin_lag{i}', f'TEC_Available_Nmin_lag{i}',
                      f'B1_Available_Nmax_lag{i}', f'B2_Available_Nmax_lag{i}', f'B3_Available_Nmax_lag{i}',
                      f'B4_Available_Nmax_lag{i}', f'TEC_Available_Nmax_lag{i}',
                      ]
        if filepath == 'data/TEC14_Data.csv':
            weights = df['sample_weight'].values
            weights_train = weights[:test_start_index]

            cols.append(df[['T']].shift(-i))
            cols.append(df[['Month_sin']].shift(-i))
            cols.append(df[['Month_cos']].shift(-i))
            cols.append(df[['Year']].shift(-i))
            cols.append(df[['TEC_N_Aver']].shift(-i))
            cols.append(df[['B1_inWork']].shift(-i))
            cols.append(df[['B2_inWork']].shift(-i))
            cols.append(df[['B1_GT11_inWork']].shift(-i))
            cols.append(df[['B1_GT12_inWork']].shift(-i))
            cols.append(df[['B2_GT21_inWork']].shift(-i))
            cols.append(df[['B2_GT22_inWork']].shift(-i))
            cols.append(df[['B1_Available_Nmin']].shift(-i))
            cols.append(df[['B2_Available_Nmin']].shift(-i))
            cols.append(df[['TEC_Available_Nmin']].shift(-i))
            cols.append(df[['B1_Available_Nmax']].shift(-i))
            cols.append(df[['B2_Available_Nmax']].shift(-i))
            cols.append(df[['TEC_Available_Nmax']].shift(-i))
            names += [f'T_lag{i}', f'Month_sin_lag{i}', f'Month_cos_lag{i}',f'Year_lag{i}',f'TEC_N_Aver_lag{i}',
                      f'B1_inWork_lag{i}', f'B2_inWork_lag{i}',
                      f'B1_GT11_inWork_lag{i}', f'B1_GT12_inWork_lag{i}', f'B2_GT21_inWork_lag{i}', f'B2_GT22_inWork_lag{i}',
                      f'B1_Available_Nmin_lag{i}', f'B2_Available_Nmin_lag{i}', f'TEC_Available_Nmin_lag{i}',
                      f'B1_Available_Nmax_lag{i}', f'B2_Available_Nmax_lag{i}', f'TEC_Available_Nmax_lag{i}',
                      ]
    agg = concat(cols, axis=1)
    agg.columns = names
    agg.dropna(inplace=True)

    data_with_lag = pd.concat([data, agg], axis=1)
    data_with_lag.dropna(inplace=True)

    return data_with_lag, base_features, weights_train
def prepare_meta_step_data(filepath, test_start_index, n_out, results,best_window_model_name, best_stat_model_name, best_step_model, test_idx, calc_goal):
    data = pd.read_csv(filepath, delimiter=';', parse_dates=['Date'], dayfirst=True)

    data['TEC_N_Aver_pred'] = data['TEC_N_Aver']

    data['Month'] = data['Date'].dt.month
    data['Year'] = data['Date'].dt.year
    data['Month_sin'] = np.sin(2 * np.pi * data['Month'] / 12)
    data['Month_cos'] = np.cos(2 * np.pi * data['Month'] / 12)

    first_work_day = data[data['TEC_N_Aver'] > 0].index[0]
    data = data.loc[first_work_day:]

    if filepath == 'data/TEC22_Data.csv':
        data['B1_N_Aver'] = data['B1_N'] / 24
        data['B2_N_Aver'] = data['B2_N'] / 24
        data['B3_N_Aver'] = data['B3_N'] / 24
        data['B4_GT41_N_Aver'] = data['B4_GT41_N'] / 24
        data['B4_GT42_N_Aver'] = data['B4_GT42_N'] / 24
        data['B4_PT40_N_Aver'] = data['B4_N_Aver'] - data['B4_GT41_N_Aver'] - data['B4_GT42_N_Aver']

        data['B1_inWork'] = np.where(data['B1_N_Aver'] >= 125, 1, 0)
        data['B2_inWork'] = np.where(data['B2_N_Aver'] >= 125, 1, 0)
        data['B3_inWork'] = np.where(data['B3_N_Aver'] >= 125, 1, 0)
        data['B4_GT41_inWork'] = np.where(data['B4_GT41_N_Aver'] >= 50, 1, 0)
        data['B4_GT42_inWork'] = np.where(data['B4_GT42_N_Aver'] >= 50, 1, 0)
        data['B4_inWork'] = np.where(data['B4_N_Aver'] >= 76, 1, 0)

        data.loc[data['B1_N_Aver'] < 125, 'B1_N_Aver'] = 0
        data.loc[data['B2_N_Aver'] < 125, 'B2_N_Aver'] = 0
        data.loc[data['B3_N_Aver'] < 125, 'B3_N_Aver'] = 0
        data.loc[data['B4_GT41_N_Aver'] <= 50, 'B4_GT41_N_Aver'] = 0
        data.loc[data['B4_GT42_N_Aver'] <= 50, 'B4_GT42_N_Aver'] = 0
        data.loc[data['B4_N_Aver'] <= 76, 'B4_N_Aver'] = 0
        data.loc[data['TEC_N_Aver'] <= 50, 'TEC_N_Aver'] = 0

        data['B4_GT41_N_Aver'] = np.where(data['B4_N_Aver'] < 76, 0, data['B4_GT41_N_Aver'])
        data['B4_GT42_N_Aver'] = np.where(data['B4_N_Aver'] < 76, 0, data['B4_GT42_N_Aver'])

        data['B1_Available_Nmax'] = data['B1_inWork'] * 250
        data['B2_Available_Nmax'] = data['B2_inWork'] * 250
        data['B3_Available_Nmax'] = data['B3_inWork'] * 250
        data['B4_GT41_Available_Nmax'] = np.where(data['T'] <= -2.3, data['B4_GT41_inWork']*172.7, data['B4_GT41_inWork']*(-0.9484 * data['T'] + 170.78))
        data['B4_GT42_Available_Nmax'] = np.where(data['T'] <= -2.3, data['B4_GT42_inWork']*172.7, data['B4_GT42_inWork']*(-0.9484 * data['T'] + 170.78))
        gt41_conditions = [((data['Year'] >= 2024) & (data['B4_GT41_Available_Nmax'] > 160.0)), (data['Year'] < 2024), ((data['Year'] >= 2024) & (data['B4_GT41_Available_Nmax'] <= 160.0))]
        gt42_conditions = [((data['Year'] >= 2024) & (data['B4_GT42_Available_Nmax'] > 160.0)), (data['Year'] < 2024), ((data['Year'] >= 2024) & (data['B4_GT42_Available_Nmax'] <= 160.0))]
        gt41_choices = [160.0, (data['B4_GT41_Available_Nmax']), (data['B4_GT41_Available_Nmax'])]
        gt42_choices = [160.0, (data['B4_GT42_Available_Nmax']), (data['B4_GT42_Available_Nmax'])]
        data['B4_GT41_Available_Nmax'] = np.select(gt41_conditions , gt41_choices, 160)
        data['B4_GT42_Available_Nmax'] = np.select(gt42_conditions , gt42_choices, 160)
        data['B4_Available_Nmax'] = data['B4_GT41_Available_Nmax'] + data['B4_GT42_Available_Nmax'] + 72*(data['B4_GT41_inWork'] + data['B4_GT42_inWork'])
        data['TEC_Available_Nmax'] = data['B1_Available_Nmax'] + data['B2_Available_Nmax'] + data['B3_Available_Nmax'] + \
                                     data['B4_Available_Nmax']

        data['B1_Available_Nmin'] = data['B1_inWork'] * 125
        data['B2_Available_Nmin'] = data['B2_inWork'] * 125
        data['B3_Available_Nmin'] = data['B3_inWork'] * 125
        data['B4_Available_Nmin'] = np.where(data['T'] <= -2.3,
                                             0.5 * data['B4_GT41_inWork'] * 172.7 + 0.5 * data['B4_GT42_inWork'] * 172.7 + 26 * (data['B4_GT41_inWork'] + data['B4_GT42_inWork']),
                                             0.5 * data['B4_GT41_inWork'] * (-0.9484 * data['T'] + 170.78) + 0.5 * data['B4_GT42_inWork'] * (-0.9484 * data['T'] + 170.78) + 26 * (data['B4_GT41_inWork'] + data['B4_GT42_inWork']))
        data['TEC_Available_Nmin'] = data['B1_Available_Nmin'] + data['B2_Available_Nmin'] + data['B3_Available_Nmin'] + \
                                     data['B4_Available_Nmin']

        is_half_block = (data['B4_GT41_inWork'] + data['B4_GT42_inWork'] == 1)
        data['sample_weight'] = np.where(is_half_block, 2.0, 1.0)

        data.dropna(inplace=True)

        data.set_index('Date', inplace=True)

        first_pred_value = results[best_window_model_name].flatten()[0]
        first_test_index = test_idx[0]
        data.loc[first_test_index, 'TEC_N_Aver_pred'] = first_pred_value
        data.loc[test_idx[1:15], 'TEC_N_Aver_pred'] = best_step_model[0][:14]

        remaining_indices = test_idx[15:]
        remaining_preds = results[best_stat_model_name].flatten()[-len(remaining_indices):]
        data.loc[remaining_indices, 'TEC_N_Aver_pred'] = remaining_preds

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
        data['B1_Available_Nmin_Prev'] = data['B1_Available_Nmin'].shift(1)
        data['B2_Available_Nmin_Prev'] = data['B2_Available_Nmin'].shift(1)
        data['B3_Available_Nmin_Prev'] = data['B3_Available_Nmin'].shift(1)
        data['B4_Available_Nmin_Prev'] = data['B4_Available_Nmin'].shift(1)
        data['TEC_Available_Nmin_Prev'] = data['TEC_Available_Nmin'].shift(1)
        data['B1_Available_Nmax_Prev'] = data['B1_Available_Nmax'].shift(1)
        data['B2_Available_Nmax_Prev'] = data['B2_Available_Nmax'].shift(1)
        data['B3_Available_Nmax_Prev'] = data['B3_Available_Nmax'].shift(1)
        data['B4_Available_Nmax_Prev'] = data['B4_Available_Nmax'].shift(1)
        data['TEC_Available_Nmax_Prev'] = data['TEC_Available_Nmax'].shift(1)
        data['TEC_N_Aver_Prev'] = data['TEC_N_Aver'].shift(1)

    if filepath == 'data/TEC14_Data.csv':
        data['B1_GT11_N_Aver'] = data['B1_GT11_N'] / 24
        data['B1_GT12_N_Aver'] = data['B1_GT12_N'] / 24
        data['B1_PT10_N_Aver'] = data['B1_PT10_N'] / 24

        data['B2_GT21_N_Aver'] = data['B2_GT21_N'] / 24
        data['B2_GT22_N_Aver'] = data['B2_GT22_N'] / 24
        data['B2_PT20_N_Aver'] = data['B2_PT20_N'] / 24

        data['B1_GT11_inWork'] = np.where(data['B1_GT11_N_Aver'] >= 30, 1, 0)
        data['B1_GT12_inWork'] = np.where(data['B1_GT12_N_Aver'] >= 30, 1, 0)
        data['B2_GT21_inWork'] = np.where(data['B2_GT21_N_Aver'] >= 30, 1, 0)
        data['B2_GT22_inWork'] = np.where(data['B2_GT22_N_Aver'] >= 30, 1, 0)
        data['B1_inWork'] = np.where(data['B1_N_Aver'] >= 45, 1, 0)
        data['B2_inWork'] = np.where(data['B2_N_Aver'] >= 45, 1, 0)

        data.loc[data['B1_GT11_N_Aver'] < 30, 'B1_GT11_N_Aver'] = 0
        data.loc[data['B1_GT12_N_Aver'] < 30, 'B1_GT12_N_Aver'] = 0
        data.loc[data['B2_GT21_N_Aver'] < 30, 'B2_GT21_N_Aver'] = 0
        data.loc[data['B2_GT22_N_Aver'] < 30, 'B2_GT22_N_Aver'] = 0
        data.loc[data['B1_N_Aver'] < 45, 'B1_N_Aver'] = 0
        data.loc[data['B2_N_Aver'] < 45, 'B2_N_Aver'] = 0
        data.loc[data['TEC_N_Aver'] < 45, 'TEC_N_Aver'] = 0
        data['B1_GT11_N_Aver'] = np.where(data['B1_N_Aver'] < 45, 0,  data['B1_GT11_N_Aver'])
        data['B1_GT12_N_Aver'] = np.where(data['B1_N_Aver'] < 45, 0,  data['B1_GT12_N_Aver'])
        data['B2_GT21_N_Aver'] = np.where(data['B2_N_Aver'] < 45, 0,  data['B2_GT21_N_Aver'])
        data['B2_GT22_N_Aver'] = np.where(data['B2_N_Aver'] < 45, 0,  data['B2_GT22_N_Aver'])

        data['B1_Available_Nmax'] = (data['B1_GT11_inWork'] * (2 / 1000000000 * data['T']**6 + 7 / 1000000000 * data['T']**5 - 4 / 1000000 * data['T']**4 - 0.0001 * data['T']**3 + 0.0006 * data['T']**2 - 0.2475 * data['T'] + 69.719)
                                     + data['B1_GT12_inWork'] * (2 / 1000000000 * data['T']**6 + 7 / 1000000000 * data['T']**5 - 4 / 1000000 * data['T']**4 - 0.0001 * data['T']**3 + 0.0006 * data['T']**2 - 0.2475 * data['T'] + 69.719)
                                     + 26 * (data['B1_GT11_inWork'] + data['B1_GT12_inWork']))
        data['B2_Available_Nmax'] = (data['B2_GT21_inWork'] * (2 / 1000000000 * data['T']**6 + 7 / 1000000000 * data['T']**5 - 4 / 1000000 * data['T']**4 - 0.0001 * data['T']**3 + 0.0006 * data['T']**2 - 0.2475 * data['T'] + 69.719)
                                     + data['B2_GT22_inWork'] * (2 / 1000000000 * data['T']**6 + 7 / 1000000000 * data['T']**5 - 4 / 1000000 * data['T']**4 - 0.0001 * data['T']**3 + 0.0006 * data['T']**2 - 0.2475 * data['T'] + 69.719)
                                     + 26 * (data['B2_GT21_inWork'] + data['B2_GT22_inWork']))
        data['TEC_Available_Nmax'] = data['B1_Available_Nmax'] + data['B2_Available_Nmax']

        data['B1_Available_Nmin'] = (data['B1_GT11_inWork'] * 0.4*(2 / 1000000000 * data['T']**6 + 7 / 1000000000 * data['T']**5 - 4 / 1000000 * data['T']**4 - 0.0001 * data['T']**3 + 0.0006 * data['T']**2 - 0.2475 * data['T'] + 69.719)
                                     + data['B1_GT12_inWork'] * 0.4*(2 / 1000000000 * data['T']**6 + 7 / 1000000000 * data['T']**5 - 4 / 1000000 * data['T']**4 - 0.0001 * data['T']**3 + 0.0006 * data['T']**2 - 0.2475 * data['T'] + 69.719)
                                     + 15 * ( data['B1_GT11_inWork'] + data['B1_GT12_inWork']))
        data['B2_Available_Nmin'] = (data['B2_GT21_inWork'] * 0.4*(2 / 1000000000 * data['T']**6 + 7 / 1000000000 * data['T']**5 - 4 / 1000000 * data['T']**4 - 0.0001 * data['T']**3 + 0.0006 * data['T']**2 - 0.2475 * data['T'] + 69.719)
                                     + data['B2_GT22_inWork'] * 0.4*(2 / 1000000000 * data['T']**6 + 7 / 1000000000 * data['T']**5 - 4 / 1000000 * data['T']**4 - 0.0001 * data['T']**3 + 0.0006 * data['T']**2 - 0.2475 * data['T'] + 69.719)
                                     + 15 * (data['B2_GT21_inWork'] + data['B2_GT22_inWork']))
        data['TEC_Available_Nmin'] = data['B1_Available_Nmin'] + data['B2_Available_Nmin']

        data['sample_weight'] = 1.0

        data.dropna(inplace=True)

        data.set_index('Date', inplace=True)

        first_pred_value = results[best_window_model_name].flatten()[0]
        first_test_index = test_idx[0]
        data.loc[first_test_index, 'TEC_N_Aver_pred'] = first_pred_value
        data.loc[test_idx[1:15], 'TEC_N_Aver_pred'] = best_step_model[0][:14]

        remaining_indices = test_idx[15:]
        remaining_preds = results[best_stat_model_name].flatten()[-len(remaining_indices):]
        data.loc[remaining_indices, 'TEC_N_Aver_pred'] = remaining_preds

        data['T_Prev'] = data['T'].shift(1)
        data['Month_sin_Prev'] = data['Month_sin'].shift(1)
        data['Month_cos_Prev'] = data['Month_cos'].shift(1)
        data['Year_Prev'] = data['Year'].shift(1)
        data['B1_N_Aver_Prev'] = data['B1_N_Aver'].shift(1)
        data['B2_N_Aver_Prev'] = data['B2_N_Aver'].shift(1)
        data['B1_Available_Nmin_Prev'] = data['B1_Available_Nmin'].shift(1)
        data['B2_Available_Nmin_Prev'] = data['B2_Available_Nmin'].shift(1)
        data['TEC_Available_Nmin_Prev'] = data['TEC_Available_Nmin'].shift(1)
        data['B1_Available_Nmax_Prev'] = data['B1_Available_Nmax'].shift(1)
        data['B2_Available_Nmax_Prev'] = data['B2_Available_Nmax'].shift(1)
        data['TEC_Available_Nmax_Prev'] = data['TEC_Available_Nmax'].shift(1)
        data['TEC_N_Aver_Prev'] = data['TEC_N_Aver'].shift(1)

    base_features = ['T', 'Month_sin', 'Month_cos', 'TEC_N_Aver_pred',
     'T_Prev', 'Month_sin_Prev', 'Month_cos_Prev','TEC_N_Aver_Prev']
    prefixes = goal_mapping[filepath]
    for prefix in prefixes:
        base_features.append(f'{prefix}_Available_Nmax')
        base_features.append(f'{prefix}_Available_Nmin')
        base_features.append(f'{prefix}_Available_Nmax_Prev')
        base_features.append(f'{prefix}_Available_Nmin_Prev')
        #base_features.append(f'{prefix}_N_Aver_Prev')

    df = DataFrame(data)

    cols, names, agg = list(), list(), list()
    for i in range(1, n_out + 1):
        weights = df['sample_weight'].values
        weights_train = weights[:test_start_index]

        if filepath == 'data/TEC22_Data.csv':
            cols.append(df[['T']].shift(-i))
            cols.append(df[['Month_sin']].shift(-i))
            cols.append(df[['Month_cos']].shift(-i))
            cols.append(df[['Year']].shift(-i))
            cols.append(df[['TEC_N_Aver_pred']].shift(-i))
            cols.append(df[['B1_inWork']].shift(-i))
            cols.append(df[['B2_inWork']].shift(-i))
            cols.append(df[['B3_inWork']].shift(-i))
            cols.append(df[['B4_GT41_inWork']].shift(-i))
            cols.append(df[['B4_GT42_inWork']].shift(-i))
            cols.append(df[['B4_inWork']].shift(-i))
            cols.append(df[['B1_Available_Nmin']].shift(-i))
            cols.append(df[['B2_Available_Nmin']].shift(-i))
            cols.append(df[['B3_Available_Nmin']].shift(-i))
            cols.append(df[['B4_Available_Nmin']].shift(-i))
            cols.append(df[['TEC_Available_Nmin']].shift(-i))
            cols.append(df[['B1_Available_Nmax']].shift(-i))
            cols.append(df[['B2_Available_Nmax']].shift(-i))
            cols.append(df[['B3_Available_Nmax']].shift(-i))
            cols.append(df[['B4_Available_Nmax']].shift(-i))
            cols.append(df[['TEC_Available_Nmax']].shift(-i))
            cols.append(df[['B1_N_Aver']].shift(-i))
            cols.append(df[['B2_N_Aver']].shift(-i))
            cols.append(df[['B3_N_Aver']].shift(-i))
            cols.append(df[['B4_N_Aver']].shift(-i))
            names += [f'T_lag{i}', f'Month_sin_lag{i}', f'Month_cos_lag{i}', f'Year_lag{i}',
                      f'TEC_N_Aver_pred_lag{i}', f'B1_inWork_lag{i}', f'B2_inWork_lag{i}', f'B3_inWork_lag{i}',
                      f'B4_GT41_inWork_lag{i}', f'B4_GT42_inWork_lag{i}', f'B4_inWork_lag{i}',
                      f'B1_Available_Nmin_lag{i}', f'B2_Available_Nmin_lag{i}', f'B3_Available_Nmin_lag{i}',
                      f'B4_Available_Nmin_lag{i}', f'TEC_Available_Nmin_lag{i}',
                      f'B1_Available_Nmax_lag{i}', f'B2_Available_Nmax_lag{i}', f'B3_Available_Nmax_lag{i}',
                      f'B4_Available_Nmax_lag{i}', f'TEC_Available_Nmax_lag{i}',
                      f'B1_N_Aver_lag{i}', f'B2_N_Aver_lag{i}', f'B3_N_Aver_lag{i}', f'B4_N_Aver_lag{i}']

        if filepath == 'data/TEC14_Data.csv':
            cols.append(df[['T']].shift(-i))
            cols.append(df[['Month_sin']].shift(-i))
            cols.append(df[['Month_cos']].shift(-i))
            cols.append(df[['Year']].shift(-i))
            cols.append(df[['TEC_N_Aver_pred']].shift(-i))
            cols.append(df[['B1_inWork']].shift(-i))
            cols.append(df[['B2_inWork']].shift(-i))
            cols.append(df[['B1_GT11_inWork']].shift(-i))
            cols.append(df[['B1_GT12_inWork']].shift(-i))
            cols.append(df[['B2_GT21_inWork']].shift(-i))
            cols.append(df[['B2_GT22_inWork']].shift(-i))
            cols.append(df[['B1_Available_Nmin']].shift(-i))
            cols.append(df[['B2_Available_Nmin']].shift(-i))
            cols.append(df[['TEC_Available_Nmin']].shift(-i))
            cols.append(df[['B1_Available_Nmax']].shift(-i))
            cols.append(df[['B2_Available_Nmax']].shift(-i))
            cols.append(df[['TEC_Available_Nmax']].shift(-i))
            cols.append(df[['B1_N_Aver']].shift(-i))
            cols.append(df[['B2_N_Aver']].shift(-i))
            names += [f'T_lag{i}', f'Month_sin_lag{i}', f'Month_cos_lag{i}', f'Year_lag{i}',
                      f'TEC_N_Aver_pred_lag{i}', f'B1_inWork_lag{i}', f'B2_inWork_lag{i}',
                      f'B1_GT11_inWork_lag{i}', f'B1_GT12_inWork_lag{i}', f'B2_GT21_inWork_lag{i}', f'B2_GT22_inWork_lag{i}',
                      f'B1_Available_Nmin_lag{i}', f'B2_Available_Nmin_lag{i}', f'TEC_Available_Nmin_lag{i}',
                      f'B1_Available_Nmax_lag{i}', f'B2_Available_Nmax_lag{i}', f'TEC_Available_Nmax_lag{i}',
                      f'B1_N_Aver_lag{i}', f'B2_N_Aver_lag{i}']

    agg = concat(cols, axis=1)
    agg.columns = names
    agg.dropna(inplace=True)

    data_with_lag = pd.concat([data, agg], axis=1)
    data_with_lag.dropna(inplace=True)
    return data_with_lag, base_features, weights_train

def prepare_direct_data(filepath, data_with_lag, step, base_features, train_size):
    current_features = base_features.copy()

    current_features.append(f'T_lag{step}')
    prefixes = goal_mapping[filepath]
    for prefix in prefixes:
        current_features.append(f'{prefix}_Available_Nmax_lag{step}')
        current_features.append(f'{prefix}_Available_Nmin_lag{step}')

    x = data_with_lag[current_features].values
    y = data_with_lag.loc[:, [f'TEC_N_Aver_lag{step}']].values
    y_true_combined = data_with_lag.loc[:, [f'TEC_N_Aver_lag{step}', f'TEC_Available_Nmax_lag{step}',
                                   f'TEC_Available_Nmin_lag{step}']].values

    x_train_raw, x_test_raw = x[:train_size], x[train_size:]
    y_train_raw, y_test_raw = y[:train_size], y[train_size:]
    y_train_comb_raw, y_test_comb_raw = y_true_combined[:train_size], y_true_combined[train_size:]

    scaler_x = MinMaxScaler()
    x_train_scaled = scaler_x.fit_transform(x_train_raw)
    x_test_scaled = scaler_x.transform(x_test_raw)

    scaler_y = MinMaxScaler()
    scaler_y.fit(y_train_raw)
    y_train_scaled =scaler_y.transform(y_train_raw)
    y_test_scaled = scaler_y.transform(y_test_raw)

    y_train_s_combined = scale_combined(y_train_comb_raw, scaler_y)
    y_test_s_combined = scale_combined(y_test_comb_raw, scaler_y)

    return (x_train_scaled, x_test_scaled, y_train_scaled, y_test_scaled,
            scaler_y, y_test_raw, y_train_s_combined, y_test_s_combined, y_test_comb_raw)
def prepare_meta_direct_data(filepath, data_with_lag, step, base_features, train_size, calc_goal):

    current_features = base_features.copy()

    current_features.append(f'T_lag{step}')
    current_features.append(f'TEC_N_Aver_pred_lag{step}')
    current_features.append(calc_goal + f'_Available_Nmax_lag{step}')
    current_features.append(calc_goal + f'_Available_Nmin_lag{step}')

    x = data_with_lag[current_features].values
    y = data_with_lag.loc[:, [calc_goal + f'_N_Aver_lag{step}']].values
    y_true_combined = data_with_lag.loc[:, [calc_goal + f'_N_Aver_lag{step}', calc_goal + f'_Available_Nmax_lag{step}',
                                   calc_goal + f'_Available_Nmin_lag{step}']].values

    x_train_raw, x_test_raw = x[:train_size], x[train_size:]
    y_train_raw, y_test_raw = y[:train_size], y[train_size:]
    y_train_comb_raw, y_test_comb_raw = y_true_combined[:train_size], y_true_combined[train_size:]

    scaler_x = MinMaxScaler()
    x_train_scaled = scaler_x.fit_transform(x_train_raw)
    x_test_scaled = scaler_x.transform(x_test_raw)

    scaler_y = MinMaxScaler()
    scaler_y.fit(y_train_raw)
    y_train_scaled =scaler_y.transform(y_train_raw)
    y_test_scaled = scaler_y.transform(y_test_raw)

    y_train_s_combined = scale_combined(y_train_comb_raw, scaler_y)
    y_test_s_combined = scale_combined(y_test_comb_raw, scaler_y)

    return (x_train_scaled, x_test_scaled, y_train_scaled, y_test_scaled,
            scaler_y, y_test_raw, y_train_s_combined, y_test_s_combined, y_test_comb_raw)

def optuna_cbr_search(trial, x_train, y_train, x_test, y_test, scaler_y):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    params = {
        "iterations": 1000,
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_float("lr", 1e-3, 0.1, log=True),
        "loss_function": "MAE",
        "verbose": 0,
        "early_stopping_rounds": 50
    }

    pruning_callback = CatBoostPruningCallback(trial, "MAE")

    model = CatBoostRegressor(**params)
    model.fit(x_train, y_train, eval_set=(x_test, y_test), use_best_model=True, callbacks=[pruning_callback])

    preds = model.predict(x_test)
    y_pred_unscaled = scaler_y.inverse_transform(preds.reshape(-1, 1))
    y_test_unscaled = scaler_y.inverse_transform(y_test.reshape(-1, 1))
    mae = metrics.mean_absolute_error(y_pred_unscaled, y_test_unscaled)

    return mae
def optuna_rfr_search(trial, x_train, y_train, x_test, y_test, scaler_y):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
        "max_features": trial.suggest_float("max_features", 0.1, 1.0),
        "n_jobs": -1
    }

    model = RandomForestRegressor(**params)
    model.fit(x_train, y_train.ravel())

    preds = model.predict(x_test)
    yhat = scaler_y.inverse_transform(preds.reshape(-1, 1))
    y_true = scaler_y.inverse_transform(y_test.reshape(-1, 1))

    mae = metrics.mean_absolute_error(y_true, yhat)

    return mae
def optuna_lstm_search(trial, x_train, y_train, x_test, y_test, scaler_y, input_shape,
                       y_train_s_combined, y_test_s_combined, loss_type='custom'):

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    n_units_lstm = trial.suggest_int('n_units_lstm', 20, 150) if trial else 50
    n_units_dense = trial.suggest_int('n_units_dense', 10, 50) if trial else 25
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True) if trial else 0.001

    model = Sequential([
        LSTM(n_units_lstm, input_shape=input_shape, unroll=True),
        Dense(n_units_dense),
        Dense(1, dtype='float32')
    ])
    optimizer = Adam(learning_rate=lr)

    model_loss = custom_loss if loss_type == 'custom' else 'mae'
    model.compile(optimizer=optimizer, loss=model_loss)

    current_batch_size = 4096

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train_s_combined))
    train_dataset = (train_dataset
                     .shuffle(buffer_size=len(x_train))
                     .batch(current_batch_size)
                     .prefetch(tf.data.AUTOTUNE))

    val_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test_s_combined))
    val_dataset = val_dataset.batch(current_batch_size).prefetch(tf.data.AUTOTUNE)

    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=25,
        verbose=0,
        callbacks=[optuna.integration.TFKerasPruningCallback(trial, 'val_loss')]
    )

    pred_lstm_s = model.predict(val_dataset, verbose=0)
    y_pred_unscaled = scaler_y.inverse_transform(pred_lstm_s)
    y_test_unscaled = scaler_y.inverse_transform(y_test)

    mae = metrics.mean_absolute_error(y_pred_unscaled, y_test_unscaled)
    return mae
def optuna_mlp_search(trial, x_train, y_train, x_test, y_test, scaler_y, y_train_s_combined, y_test_s_combined):


    optuna.logging.set_verbosity(optuna.logging.WARNING)

    n_layers = trial.suggest_int('n_layers', 1, 3)
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)

    model = Sequential()
    for i in range(n_layers):
        units = trial.suggest_int(f'units_l{i}', 16, 128)
        model.add(Dense(units, activation='relu'))

    model.add(Dense(1, dtype='float32'))
    model.compile(optimizer=Adam(learning_rate=lr), loss=custom_loss)

    current_batch_size = 4096

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train_s_combined))
    train_dataset = (train_dataset 
                     .shuffle(buffer_size=len(x_train))
                     .batch(current_batch_size)
                     .prefetch(tf.data.AUTOTUNE))

    val_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test_s_combined))
    val_dataset = val_dataset.batch(current_batch_size).prefetch(tf.data.AUTOTUNE)

    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=25,
        verbose=0,
        callbacks=[optuna.integration.TFKerasPruningCallback(trial, 'val_loss')]
    )

    preds = model.predict(val_dataset, verbose=0)

    y_pred_unscaled = scaler_y.inverse_transform(preds.reshape(-1, 1))
    y_test_unscaled = scaler_y.inverse_transform(y_test.reshape(-1, 1))
    mae = metrics.mean_absolute_error(y_pred_unscaled, y_test_unscaled)

    return mae
@tf.keras.utils.register_keras_serializable()
def custom_loss(y_true_combined, y_pred):
    dtype = y_pred.dtype
    lambda_bounds = tf.cast(25.0, dtype=dtype)
    zero_const = tf.cast(0.0, dtype=dtype)

    y_true = y_true_combined[:, 0:1]
    avail_nmax = y_true_combined[:, 1:2]
    avail_nmin = y_true_combined[:, 2:3]

    base_loss = tf.reduce_mean(tf.abs(y_true - y_pred))

    upper_penalty = tf.reduce_mean(tf.square(tf.maximum(zero_const, y_pred - avail_nmax)))
    lower_penalty = tf.reduce_mean(tf.square(tf.maximum(zero_const, avail_nmin - y_pred)))

    total_loss = base_loss + lambda_bounds * (upper_penalty + lower_penalty)

    return total_loss
def scale_combined(combined_data, scaler):
    col0 = scaler.transform(combined_data[:, 0:1])
    col1 = scaler.transform(combined_data[:, 1:2])
    col2 = scaler.transform(combined_data[:, 2:3])
    return np.column_stack([col0, col1, col2])

def reconcile_with_mip(df_preds, tec_col, boiler_prefixes):
    results = {p: [] for p in boiler_prefixes}

    for idx, row in df_preds.iterrows():
        tec_p = row[tec_col]

        model = pulp.LpProblem("Power_Balance", pulp.LpMinimize)

        targets = {}
        ons = {}
        diffs = {}

        for p in boiler_prefixes:
            p_val = row[f"{p}_N_Aver_pred"]
            p_min = row[f"{p}_Available_Nmin"]
            p_max = row[f"{p}_Available_Nmax"]

            targets[p] = pulp.LpVariable(f"target_{p}", lowBound=0)
            ons[p] = pulp.LpVariable(f"on_{p}", cat=pulp.LpBinary)
            diffs[p] = pulp.LpVariable(f"d_{p}", lowBound=0)

            model += diffs[p] >= targets[p] - p_val
            model += diffs[p] >= p_val - targets[p]

            model += targets[p] >= ons[p] * p_min
            model += targets[p] <= ons[p] * p_max

        model += pulp.lpSum(targets.values()) == tec_p

        model.objective = pulp.lpSum(diffs.values())

        model.solve(pulp.PULP_CBC_CMD(msg=0))

        for p in boiler_prefixes:
            results[p].append(pulp.value(targets[p]))

    output = pd.DataFrame({tec_col: df_preds[tec_col].values}, index=df_preds.index)

    orig_cols = [f"{p}_N_Aver_pred" for p in boiler_prefixes]
    reconciled_cols = [f'{p}_Reconciled' for p in boiler_prefixes]

    for p in boiler_prefixes:
        orig_col = f"{p}_N_Aver_pred"
        recon_col = f'{p}_Reconciled'
        delta_col = f'{p}_Delta'

        output[orig_col] = df_preds[orig_col].values
        output[recon_col] = results[p]
        output[delta_col] = output[recon_col] - output[orig_col]

    output['Sum_Check'] = output[reconciled_cols].sum(axis=1)

    output['Total_Error_Before'] = output[orig_cols].sum(axis=1) - output[tec_col]

    output['Total_Error_After'] = output['Sum_Check'] - output[tec_col]

    return output
def reconcile_with_l2(df_preds, tec_col, boiler_prefixes):
    results = {p: [] for p in boiler_prefixes}

    for idx, row in df_preds.iterrows():
        tec_p = row[tec_col]

        p_vals = []
        bounds = []

        for p in boiler_prefixes:
            p_val = row[f"{p}_N_Aver_pred"]
            p_min = row[f"{p}_Available_Nmin"]
            p_max = row[f"{p}_Available_Nmax"]

            p_vals.append(p_val)

            if p_val > 0:
                bounds.append((p_min, p_max))
            else:
                bounds.append((0.0, 0.0))

        p_vals = np.array(p_vals)

        def objective(targets): return np.sum((targets - p_vals) ** 2)

        def constraint_balance(targets):return np.sum(targets) - tec_p

        constraints = {"type": "eq", "fun": constraint_balance}

        x0 = p_vals.copy()

        res = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=constraints)
        final_targets = res.x if res.success else x0

        for i, p in enumerate(boiler_prefixes):
            results[p].append(final_targets[i])

    output = pd.DataFrame(
        {tec_col: df_preds[tec_col].values}, index=df_preds.index
    )

    orig_cols = [f"{p}_N_Aver_pred" for p in boiler_prefixes]
    reconciled_cols = [f"{p}_Reconciled" for p in boiler_prefixes]

    for p in boiler_prefixes:
        orig_col = f"{p}_N_Aver_pred"
        recon_col = f"{p}_Reconciled"
        delta_col = f"{p}_Delta"

        output[orig_col] = df_preds[orig_col].values
        output[recon_col] = results[p]
        output[delta_col] = output[recon_col] - output[orig_col]

    output["Sum_Check"] = output[reconciled_cols].sum(axis=1)
    output["Total_Error_Before"] = (
        output[orig_cols].sum(axis=1) - output[tec_col]
    )
    output["Total_Error_After"] = output["Sum_Check"] - output[tec_col]

    return output
import optuna
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from keras import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import Adam
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from pandas import DataFrame, concat
from optuna.integration import CatBoostPruningCallback
import tensorflow as tf

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

        data['B1_inWork'] = np.where(data['B1_N_Aver'] > 125, 1, 0)
        data['B2_inWork'] = np.where(data['B2_N_Aver'] > 125, 1, 0)
        data['B3_inWork'] = np.where(data['B3_N_Aver'] > 125, 1, 0)
        data['B4_GT41_inWork'] = np.where(data['B4_GT41_N_Aver'] > 50, 1, 0)
        data['B4_GT42_inWork'] = np.where(data['B4_GT42_N_Aver'] > 50, 1, 0)
        data['B4_inWork'] = np.where(data['B4_N_Aver'] > 40, 1, 0)

        data.loc[data['B1_N_Aver'] < 125, 'B1_N_Aver'] = 0
        data.loc[data['B2_N_Aver'] < 125, 'B2_N_Aver'] = 0
        data.loc[data['B3_N_Aver'] < 125, 'B3_N_Aver'] = 0
        data.loc[data['B4_GT41_N_Aver'] < 50, 'B4_GT41_N_Aver'] = 0
        data.loc[data['B4_GT42_N_Aver'] < 50, 'B4_GT42_N_Aver'] = 0
        data.loc[data['B4_PT40_N_Aver'] < 40, 'B4_PT40_N_Aver'] = 0
        data.loc[data['TEC_N_Aver'] < 40, 'B1_N_Aver'] = 0

        #data = data.drop(data[(data['B1_N_Aver'] < 125) & (data['B1_N_Aver'] > 0)].index)
        #data = data.drop(data[(data['B2_N_Aver'] < 125) & (data['B2_N_Aver'] > 0)].index)
        #data = data.drop(data[(data['B3_N_Aver'] < 125) & (data['B3_N_Aver'] > 0)].index)
        #data = data.drop(data[(data['B4_GT41_N_Aver'] < 50) & (data['B4_GT41_N_Aver'] > 0)].index)
        #data = data.drop(data[(data['B4_GT42_N_Aver'] < 50) & (data['B4_GT42_N_Aver'] > 0)].index)
        #data = data.drop(data[(data['B4_PT40_N_Aver'] < 40) & (data['B4_PT40_N_Aver'] > 0)].index)
        #data = data.drop(data[(data['TEC_N_Aver'] == 0)].index)

        data['B1_Available_N'] = data['B1_inWork'] * 250
        data['B2_Available_N'] = data['B2_inWork'] * 250
        data['B3_Available_N'] = data['B3_inWork'] * 250
        # data['B4_Available_N'] = np.where(data['T'] < -2,
        #    (data['B4_GT41_inWork'] * 160 + data['B4_GT42_inWork'] * 160 + 72 *(data['B4_GT41_inWork'] + data['B4_GT42_inWork'])),
        #    (data['B4_GT41_inWork'] * 150 + data['B4_GT42_inWork'] * 150 + 72 *(data['B4_GT41_inWork'] + data['B4_GT42_inWork'])))
        data['B4_Available_N'] = (data['B4_GT41_inWork'] * 150 + data['B4_GT42_inWork'] * 150 + 72 * (
                    data['B4_GT41_inWork'] + data['B4_GT42_inWork']))

        data.dropna(inplace=True)
        print(data.shape)
        data.set_index('Date', inplace=True)

        X = data.loc[:, ['T',  'Month_sin', 'Month_cos', 'Year',
                        'B1_inWork', 'B2_inWork', 'B3_inWork', 'B4_GT41_inWork', 'B4_GT42_inWork', 'B4_inWork',
                        'B1_Available_N', 'B2_Available_N', 'B3_Available_N', 'B4_Available_N']].values
        y = data.loc[:, ['TEC_N_Aver']].values

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
        data['B1_inWork'] = np.where(data['B1_N_Aver'] > 50, 1, 0)
        data['B2_inWork'] = np.where(data['B2_N_Aver'] > 50, 1, 0)

        data.loc[data['B1_GT11_N_Aver'] < 30, 'B1_GT11_N_Aver'] = 0
        data.loc[data['B1_GT12_N_Aver'] < 30, 'B1_GT12_N_Aver'] = 0
        data.loc[data['B2_GT21_N_Aver'] < 30, 'B2_GT21_N_Aver'] = 0
        data.loc[data['B2_GT22_N_Aver'] < 30, 'B2_GT22_N_Aver'] = 0
        data.loc[data['B1_N_Aver'] < 50, 'B1_N_Aver'] = 0
        data.loc[data['B2_N_Aver'] < 50, 'B2_N_Aver'] = 0
        data.loc[data['TEC_N_Aver'] < 50, 'TEC_N_Aver'] = 0

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
        print(data.shape)

        data.set_index('Date', inplace=True)

        X = data.loc[:, ['T', 'Month_sin', 'Month_cos',
                         'B1_Available_Nmax', 'B2_Available_Nmax',
                         'B1_Available_Nmin', 'B2_Available_Nmin',
                         'TEC_Available_Nmax','TEC_Available_Nmin']].values
        y = data.loc[:, ['TEC_N_Aver']].values
        y_true_combined = data.loc[:, ['TEC_N_Aver', 'TEC_Available_Nmax',
                                       'TEC_Available_Nmin']].values


    n = test_start_index
    X_train, X_test = X[:n], X[n:]
    y_train, y_test = y[:n], y[n:]
    y_train_comb_raw, y_test_comb_raw = y_true_combined[:n], y_true_combined[n:]

    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_train_s = scaler_x.fit_transform(X_train)
    X_test_s = scaler_x.transform(X_test)

    scaler_y.fit(y_train)
    y_train_s =scaler_y.transform(y_train)
    y_test_s = scaler_y.transform(y_test)

    y_train_s_combined = scale_combined(y_train_comb_raw, scaler_y)
    y_test_s_combined = scale_combined(y_test_comb_raw, scaler_y)

    test_idx = data.index[test_start_index:]

    return X_train_s, X_test_s, y_train_s, y_test_s, y_train, y_test, scaler_y, data.index, test_idx, y_train_s_combined, y_test_s_combined
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

        data['B1_inWork'] = np.where(data['B1_N_Aver'] > 125, 1, 0)
        data['B2_inWork'] = np.where(data['B2_N_Aver'] > 125, 1, 0)
        data['B3_inWork'] = np.where(data['B3_N_Aver'] > 125, 1, 0)
        data['B4_GT41_inWork'] = np.where(data['B4_GT41_N_Aver'] > 50, 1, 0)
        data['B4_GT42_inWork'] = np.where(data['B4_GT42_N_Aver'] > 50, 1, 0)
        data['B4_inWork'] = np.where(data['B4_N_Aver'] > 40, 1, 0)

        data.loc[data['B1_N_Aver'] < 125, 'B1_N_Aver'] = 0
        data.loc[data['B2_N_Aver'] < 125, 'B2_N_Aver'] = 0
        data.loc[data['B3_N_Aver'] < 125, 'B3_N_Aver'] = 0
        data.loc[data['B4_GT41_N_Aver'] < 50, 'B4_GT41_N_Aver'] = 0
        data.loc[data['B4_GT42_N_Aver'] < 50, 'B4_GT42_N_Aver'] = 0
        data.loc[data['B4_PT40_N_Aver'] < 40, 'B4_PT40_N_Aver'] = 0
        data.loc[data['TEC_N_Aver'] < 40, 'B1_N_Aver'] = 0

        #data = data.drop(data[(data['B1_N_Aver'] < 125) & (data['B1_N_Aver'] > 0)].index)
        #data = data.drop(data[(data['B2_N_Aver'] < 125) & (data['B2_N_Aver'] > 0)].index)
        #data = data.drop(data[(data['B3_N_Aver'] < 125) & (data['B3_N_Aver'] > 0)].index)
        #data = data.drop(data[(data['B4_GT41_N_Aver'] < 50) & (data['B4_GT41_N_Aver'] > 0)].index)
        #data = data.drop(data[(data['B4_GT42_N_Aver'] < 50) & (data['B4_GT42_N_Aver'] > 0)].index)
        #data = data.drop(data[(data['B4_PT40_N_Aver'] < 40) & (data['B4_PT40_N_Aver'] > 0)].index)
        #data = data.drop(data[(data['TEC_N_Aver'] == 0)].index)

        data['B1_Available_N'] = data['B1_inWork'] * 250
        data['B2_Available_N'] = data['B2_inWork'] * 250
        data['B3_Available_N'] = data['B3_inWork'] * 250
        # data['B4_Available_N'] = np.where(data['T'] < -2,
        #    (data['B4_GT41_inWork'] * 160 + data['B4_GT42_inWork'] * 160 + 72 *(data['B4_GT41_inWork'] + data['B4_GT42_inWork'])),
        #    (data['B4_GT41_inWork'] * 150 + data['B4_GT42_inWork'] * 150 + 72 *(data['B4_GT41_inWork'] + data['B4_GT42_inWork'])))
        data['B4_Available_N'] = (data['B4_GT41_inWork'] * 150 + data['B4_GT42_inWork'] * 150 + 72 * (
                    data['B4_GT41_inWork'] + data['B4_GT42_inWork']))

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

        data.dropna(inplace=True)

        data.set_index('Date', inplace=True)

        Xw = data.loc[:,['T', 'Month_sin', 'Month_cos', 'Year',
                    'B1_inWork', 'B2_inWork', 'B3_inWork' ,'B4_GT41_inWork', 'B4_GT42_inWork', 'B4_inWork',
                    'B1_Available_N', 'B2_Available_N', 'B3_Available_N', 'B4_Available_N',
                    'T_Prev', 'Month_sin_Prev', 'Month_cos_Prev', 'Year_Prev',
                    'B4_GT41_N_Aver_Prev', 'B4_GT42_N_Aver_Prev' ,'B4_PT40_N_Aver_Prev', 'TEC_N_Aver_Prev',
                    'B1_N_Aver_Prev', 'B2_N_Aver_Prev' ,'B3_N_Aver_Prev', 'B4_N_Aver_Prev',
                    'B1_Available_N_Prev', 'B2_Available_N_Prev' ,'B3_Available_N_Prev', 'B4_Available_N_Prev']].values

        yw = data.loc[:, ['TEC_N_Aver']].values

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
        data['B1_inWork'] = np.where(data['B1_N_Aver'] > 50, 1, 0)
        data['B2_inWork'] = np.where(data['B2_N_Aver'] > 50, 1, 0)

        data.loc[data['B1_GT11_N_Aver'] < 30, 'B1_GT11_N_Aver'] = 0
        data.loc[data['B1_GT12_N_Aver'] < 30, 'B1_GT12_N_Aver'] = 0
        data.loc[data['B2_GT21_N_Aver'] < 30, 'B2_GT21_N_Aver'] = 0
        data.loc[data['B2_GT22_N_Aver'] < 30, 'B2_GT22_N_Aver'] = 0
        data.loc[data['B1_N_Aver'] < 50, 'B1_N_Aver'] = 0
        data.loc[data['B2_N_Aver'] < 50, 'B2_N_Aver'] = 0
        data.loc[data['TEC_N_Aver'] < 50, 'TEC_N_Aver'] = 0

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

        data.set_index('Date', inplace=True)

        X = data.loc[:, ['T', 'Month_sin', 'Month_cos',
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

    n = test_start_index

    n = test_start_index
    X_train, X_test = X[:n], X[n:]
    y_train, y_test = y[:n], y[n:]
    y_train_comb_raw, y_test_comb_raw = y_true_combined[:n], y_true_combined[n:]

    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_train_s = scaler_x.fit_transform(X_train)
    X_test_s = scaler_x.transform(X_test)

    scaler_y.fit(y_train)
    y_train_s =scaler_y.transform(y_train)
    y_test_s = scaler_y.transform(y_test)

    y_train_s_combined = scale_combined(y_train_comb_raw, scaler_y)
    y_test_s_combined = scale_combined(y_test_comb_raw, scaler_y)

    test_idx = data.index[test_start_index:]

    return X_train_s, X_test_s, y_train_s, y_test_s, scaler_y, data.index, test_idx, y_train_s_combined, y_test_s_combined

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

        data['B1_inWork'] = np.where(data['B1_N_Aver'] > 125, 1, 0)
        data['B2_inWork'] = np.where(data['B2_N_Aver'] > 125, 1, 0)
        data['B3_inWork'] = np.where(data['B3_N_Aver'] > 125, 1, 0)
        data['B4_GT41_inWork'] = np.where(data['B4_GT41_N_Aver'] > 50, 1, 0)
        data['B4_GT42_inWork'] = np.where(data['B4_GT42_N_Aver'] > 50, 1, 0)
        data['B4_inWork'] = np.where(data['B4_N_Aver'] > 40, 1, 0)

        data.loc[data['B1_N_Aver'] < 125, 'B1_N_Aver'] = 0
        data.loc[data['B2_N_Aver'] < 125, 'B2_N_Aver'] = 0
        data.loc[data['B3_N_Aver'] < 125, 'B3_N_Aver'] = 0
        data.loc[data['B4_GT41_N_Aver'] < 50, 'B4_GT41_N_Aver'] = 0
        data.loc[data['B4_GT42_N_Aver'] < 50, 'B4_GT42_N_Aver'] = 0
        data.loc[data['B4_PT40_N_Aver'] < 40, 'B4_PT40_N_Aver'] = 0
        data.loc[data['TEC_N_Aver'] < 40, 'B1_N_Aver'] = 0

        data['B1_Available_N'] = data['B1_inWork'] * 250
        data['B2_Available_N'] = data['B2_inWork'] * 250
        data['B3_Available_N'] = data['B3_inWork'] * 250

        data['B4_Available_N'] = (data['B4_GT41_inWork'] * 150 + data['B4_GT42_inWork'] * 150 + 72 * (
                    data['B4_GT41_inWork'] + data['B4_GT42_inWork']))

        data.dropna(inplace=True)

        data.set_index('Date', inplace=True)

        first_pred_value = results[best_window_model_name].flatten()[0]
        first_test_index = test_idx[0]
        data.loc[first_test_index, 'TEC_N_Aver_pred'] = first_pred_value
        data.loc[test_idx[1:15], 'TEC_N_Aver_pred'] = best_step_model[0][:14]

        remaining_indices = test_idx[15:]
        remaining_preds = results[best_stat_model_name].flatten()[-len(remaining_indices):]
        data.loc[remaining_indices, 'TEC_N_Aver_pred'] = remaining_preds

        X = data.loc[:,
            ['T', 'Month_sin','Month_cos', 'Year',
            'B1_inWork', 'B2_inWork', 'B3_inWork', 'B4_GT41_inWork', 'B4_GT42_inWork', 'B4_inWork',
            'B1_Available_N', 'B2_Available_N', 'B3_Available_N', 'B4_Available_N', 'TEC_N_Aver_pred']].values
        y = data.loc[:, [calc_goal+'_N_Aver']].values

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
        data['B1_inWork'] = np.where(data['B1_N_Aver'] > 50, 1, 0)
        data['B2_inWork'] = np.where(data['B2_N_Aver'] > 50, 1, 0)

        data.loc[data['B1_GT11_N_Aver'] < 30, 'B1_GT11_N_Aver'] = 0
        data.loc[data['B1_GT12_N_Aver'] < 30, 'B1_GT12_N_Aver'] = 0
        data.loc[data['B2_GT21_N_Aver'] < 30, 'B2_GT21_N_Aver'] = 0
        data.loc[data['B2_GT22_N_Aver'] < 30, 'B2_GT22_N_Aver'] = 0
        data.loc[data['B1_N_Aver'] < 50, 'B1_N_Aver'] = 0
        data.loc[data['B2_N_Aver'] < 50, 'B2_N_Aver'] = 0
        data.loc[data['TEC_N_Aver'] < 50, 'TEC_N_Aver'] = 0

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

        X = data.loc[:,
            ['T', 'Month_sin', 'Month_cos',
             calc_goal + '_Available_Nmax', calc_goal + '_Available_Nmax',
             'TEC_N_Aver_pred']].values
        y = data.loc[:, [calc_goal + '_N_Aver']].values

        y_true_combined = data.loc[:, [calc_goal + '_N_Aver', calc_goal + '_Available_Nmax',
                                       calc_goal + '_Available_Nmin']].values

    n = test_start_index
    X_train, X_test = X[:n], X[n:]
    y_train, y_test = y[:n], y[n:]
    y_train_comb_raw, y_test_comb_raw = y_true_combined[:n], y_true_combined[n:]

    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_train_s = scaler_x.fit_transform(X_train)
    X_test_s = scaler_x.transform(X_test)

    scaler_y.fit(y_train)
    y_train_s =scaler_y.transform(y_train)
    y_test_s = scaler_y.transform(y_test)

    y_train_s_combined = scale_combined(y_train_comb_raw, scaler_y)
    y_test_s_combined = scale_combined(y_test_comb_raw, scaler_y)

    test_idx = data.index[test_start_index:]

    return X_train_s, X_test_s, y_train_s, y_test_s, y_train, y_test, scaler_y, data.index, test_idx, y_train_s_combined, y_test_s_combined
def prepare_meta_window_data(results, filepath, test_idx, test_start_index, calc_goal, best_window_model_name, best_stat_model_name, best_step_model):

    print("best_window_model_name:", best_window_model_name)
    print("best_stat_model_name:", best_stat_model_name)

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

        data['B1_inWork'] = np.where(data['B1_N_Aver'] > 125, 1, 0)
        data['B2_inWork'] = np.where(data['B2_N_Aver'] > 125, 1, 0)
        data['B3_inWork'] = np.where(data['B3_N_Aver'] > 125, 1, 0)
        data['B4_GT41_inWork'] = np.where(data['B4_GT41_N_Aver'] > 50, 1, 0)
        data['B4_GT42_inWork'] = np.where(data['B4_GT42_N_Aver'] > 50, 1, 0)
        data['B4_inWork'] = np.where(data['B4_N_Aver'] > 40, 1, 0)

        data.loc[data['B1_N_Aver'] < 125, 'B1_N_Aver'] = 0
        data.loc[data['B2_N_Aver'] < 125, 'B2_N_Aver'] = 0
        data.loc[data['B3_N_Aver'] < 125, 'B3_N_Aver'] = 0
        data.loc[data['B4_GT41_N_Aver'] < 50, 'B4_GT41_N_Aver'] = 0
        data.loc[data['B4_GT42_N_Aver'] < 50, 'B4_GT42_N_Aver'] = 0
        data.loc[data['B4_PT40_N_Aver'] < 40, 'B4_PT40_N_Aver'] = 0
        data.loc[data['TEC_N_Aver'] < 40, 'B1_N_Aver'] = 0

        #data = data.drop(data[(data['B1_N_Aver'] < 125) & (data['B1_N_Aver'] > 0)].index)
        #data = data.drop(data[(data['B2_N_Aver'] < 125) & (data['B2_N_Aver'] > 0)].index)
        #data = data.drop(data[(data['B3_N_Aver'] < 125) & (data['B3_N_Aver'] > 0)].index)
        #data = data.drop(data[(data['B4_GT41_N_Aver'] < 50) & (data['B4_GT41_N_Aver'] > 0)].index)
        #data = data.drop(data[(data['B4_GT42_N_Aver'] < 50) & (data['B4_GT42_N_Aver'] > 0)].index)
        #data = data.drop(data[(data['B4_PT40_N_Aver'] < 40) & (data['B4_PT40_N_Aver'] > 0)].index)
        #data = data.drop(data[(data['TEC_N_Aver'] == 0)].index)

        data['B1_Available_N'] = data['B1_inWork'] * 250
        data['B2_Available_N'] = data['B2_inWork'] * 250
        data['B3_Available_N'] = data['B3_inWork'] * 250
        # data['B4_Available_N'] = np.where(data['T'] < -2,
        #    (data['B4_GT41_inWork'] * 160 + data['B4_GT42_inWork'] * 160 + 72 *(data['B4_GT41_inWork'] + data['B4_GT42_inWork'])),
        #    (data['B4_GT41_inWork'] * 150 + data['B4_GT42_inWork'] * 150 + 72 *(data['B4_GT41_inWork'] + data['B4_GT42_inWork'])))
        data['B4_Available_N'] = (data['B4_GT41_inWork'] * 150 + data['B4_GT42_inWork'] * 150 + 72 * (
                    data['B4_GT41_inWork'] + data['B4_GT42_inWork']))

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
        data['B1_Available_N_Prev'] = data['B1_Available_N'].shift(1)
        data['B2_Available_N_Prev'] = data['B2_Available_N'].shift(1)
        data['B3_Available_N_Prev'] = data['B3_Available_N'].shift(1)
        data['B4_Available_N_Prev'] = data['B4_Available_N'].shift(1)
        data['TEC_N_Aver_pred_Prev'] = data['TEC_N_Aver_pred'].shift(1)

        data.dropna(inplace=True)

        Xw = data.loc[:,
            ['T', 'Month_sin', 'Month_cos', 'Year',
             'B1_inWork', 'B2_inWork', 'B3_inWork', 'B4_GT41_inWork', 'B4_GT42_inWork', 'B4_inWork',
             'B1_Available_N', 'B2_Available_N', 'B3_Available_N', 'B4_Available_N', 'TEC_N_Aver_pred',
             'T_Prev', 'Month_sin_Prev', 'Month_cos_Prev', 'Year_Prev',
             'B4_GT41_N_Aver_Prev', 'B4_GT42_N_Aver_Prev', 'B4_PT40_N_Aver_Prev', 'B1_N_Aver_Prev', 'B2_N_Aver_Prev', 'B3_N_Aver_Prev', 'B4_N_Aver_Prev',
             'B1_Available_N_Prev', 'B2_Available_N_Prev', 'B3_Available_N_Prev', 'B4_Available_N_Prev', 'TEC_N_Aver_pred_Prev',

             ]].values
        yw = data.loc[:, [calc_goal+'_N_Aver']].values

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
        data['B1_inWork'] = np.where(data['B1_N_Aver'] > 50, 1, 0)
        data['B2_inWork'] = np.where(data['B2_N_Aver'] > 50, 1, 0)

        data.loc[data['B1_GT11_N_Aver'] < 30, 'B1_GT11_N_Aver'] = 0
        data.loc[data['B1_GT12_N_Aver'] < 30, 'B1_GT12_N_Aver'] = 0
        data.loc[data['B2_GT21_N_Aver'] < 30, 'B2_GT21_N_Aver'] = 0
        data.loc[data['B2_GT22_N_Aver'] < 30, 'B2_GT22_N_Aver'] = 0
        data.loc[data['B1_N_Aver'] < 50, 'B1_N_Aver'] = 0
        data.loc[data['B2_N_Aver'] < 50, 'B2_N_Aver'] = 0
        data.loc[data['TEC_N_Aver'] < 50, 'TEC_N_Aver'] = 0

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
        data['TEC_N_Aver_pred_Prev'] = data['TEC_N_Aver_pred'].shift(1)

        data.dropna(inplace=True)

        X = data.loc[:,
             ['T', 'Month_sin', 'Month_cos',
              calc_goal + '_Available_Nmax',
              calc_goal + '_Available_Nmin',
              'TEC_Available_Nmin', 'TEC_Available_Nmax',
              'TEC_N_Aver_pred',
              'T_Prev', 'Month_sin_Prev', 'Month_cos_Prev',
              calc_goal + '_N_Aver_Prev',
              calc_goal + '_Available_Nmax_Prev',
              calc_goal + '_Available_Nmin_Prev',
              'TEC_Available_Nmin_Prev', 'TEC_Available_Nmax_Prev',
              'TEC_N_Aver_pred_Prev',
              ]].values
        y = data.loc[:, [calc_goal + '_N_Aver']].values

        y_true_combined = data.loc[:, [calc_goal + '_N_Aver', calc_goal + '_Available_Nmax',
                                       calc_goal + '_Available_Nmin']].values

    n = test_start_index
    X_train, X_test = X[:n], X[n:]
    y_train, y_test = y[:n], y[n:]
    y_train_comb_raw, y_test_comb_raw = y_true_combined[:n], y_true_combined[n:]

    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_train_s = scaler_x.fit_transform(X_train)
    X_test_s = scaler_x.transform(X_test)

    scaler_y.fit(y_train)
    y_train_s =scaler_y.transform(y_train)
    y_test_s = scaler_y.transform(y_test)

    y_train_s_combined = scale_combined(y_train_comb_raw, scaler_y)
    y_test_s_combined = scale_combined(y_test_comb_raw, scaler_y)

    print('X_train_s:',X_train_s.shape)
    print('X_test_s:',X_test_s.shape)
    print('y_train_s:',y_train_s.shape)
    print('y_test_s:',y_test_s.shape)
    print('y_train_s_combined:',y_train_s_combined.shape)
    print('y_test_s_combined:',y_test_s_combined.shape)

    test_idx = data.index[test_start_index:]
    return X_train_s, X_test_s, y_train_s, y_test_s, scaler_y, data.index, test_idx, y_train_s_combined, y_test_s_combined

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

        data['B1_inWork'] = np.where(data['B1_N_Aver'] > 125, 1, 0)
        data['B2_inWork'] = np.where(data['B2_N_Aver'] > 125, 1, 0)
        data['B3_inWork'] = np.where(data['B3_N_Aver'] > 125, 1, 0)
        data['B4_GT41_inWork'] = np.where(data['B4_GT41_N_Aver'] > 50, 1, 0)
        data['B4_GT42_inWork'] = np.where(data['B4_GT42_N_Aver'] > 50, 1, 0)
        data['B4_inWork'] = np.where(data['B4_N_Aver'] > 40, 1, 0)

        data.loc[data['B1_N_Aver'] < 125, 'B1_N_Aver'] = 0
        data.loc[data['B2_N_Aver'] < 125, 'B2_N_Aver'] = 0
        data.loc[data['B3_N_Aver'] < 125, 'B3_N_Aver'] = 0
        data.loc[data['B4_GT41_N_Aver'] < 50, 'B4_GT41_N_Aver'] = 0
        data.loc[data['B4_GT42_N_Aver'] < 50, 'B4_GT42_N_Aver'] = 0
        data.loc[data['B4_PT40_N_Aver'] < 40, 'B4_PT40_N_Aver'] = 0
        data.loc[data['TEC_N_Aver'] < 40, 'B1_N_Aver'] = 0


        data['B1_Available_N'] = data['B1_inWork'] * 250
        data['B2_Available_N'] = data['B2_inWork'] * 250
        data['B3_Available_N'] = data['B3_inWork'] * 250
        # data['B4_Available_N'] = np.where(data['T'] < -2,
        #    (data['B4_GT41_inWork'] * 160 + data['B4_GT42_inWork'] * 160 + 72 *(data['B4_GT41_inWork'] + data['B4_GT42_inWork'])),
        #    (data['B4_GT41_inWork'] * 150 + data['B4_GT42_inWork'] * 150 + 72 *(data['B4_GT41_inWork'] + data['B4_GT42_inWork'])))
        data['B4_Available_N'] = (data['B4_GT41_inWork'] * 150 + data['B4_GT42_inWork'] * 150 + 72 * (
                data['B4_GT41_inWork'] + data['B4_GT42_inWork']))

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

        data.set_index('Date', inplace=True)

        base_features = ['T', 'Month_sin', 'Month_cos',
                         'B1_Available_N', 'B2_Available_N', 'B3_Available_N', 'B4_Available_N',
                         'T_Prev', 'TEC_N_Aver_Prev']

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
        data['B1_inWork'] = np.where(data['B1_N_Aver'] > 50, 1, 0)
        data['B2_inWork'] = np.where(data['B2_N_Aver'] > 50, 1, 0)

        data.loc[data['B1_GT11_N_Aver'] < 30, 'B1_GT11_N_Aver'] = 0
        data.loc[data['B1_GT12_N_Aver'] < 30, 'B1_GT12_N_Aver'] = 0
        data.loc[data['B2_GT21_N_Aver'] < 30, 'B2_GT21_N_Aver'] = 0
        data.loc[data['B2_GT22_N_Aver'] < 30, 'B2_GT22_N_Aver'] = 0
        data.loc[data['B1_N_Aver'] < 50, 'B1_N_Aver'] = 0
        data.loc[data['B2_N_Aver'] < 50, 'B2_N_Aver'] = 0
        data.loc[data['TEC_N_Aver'] < 50, 'TEC_N_Aver'] = 0

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

        base_features = ['T', 'Month_sin', 'Month_cos',
                         'B1_Available_Nmin', 'B2_Available_Nmin',
                         'B1_Available_Nmax', 'B2_Available_Nmax',
                         'TEC_Available_Nmin', 'TEC_Available_Nmax',
                         'B1_Available_Nmin_Prev', 'B2_Available_Nmin_Prev',
                         'B1_Available_Nmax_Prev', 'B2_Available_Nmax_Prev',
                         'TEC_Available_Nmin_Prev', 'TEC_Available_Nmax_Prev',
                         'T_Prev', 'TEC_N_Aver_Prev' ]

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
            cols.append(df[['B1_Available_N']].shift(-i))
            cols.append(df[['B2_Available_N']].shift(-i))
            cols.append(df[['B3_Available_N']].shift(-i))
            cols.append(df[['B4_Available_N']].shift(-i))
            names += [f'T_lag{i}', f'Month_sin_lag{i}', f'Month_cos_lag{i}', f'Year_lag{i}',
                      f'TEC_N_Aver_lag{i}', f'B1_inWork_lag{i}', f'B2_inWork_lag{i}', f'B3_inWork_lag{i}',
                      f'B4_GT41_inWork_lag{i}', f'B4_GT42_inWork_lag{i}', f'B4_inWork_lag{i}',
                      f'B1_Available_N_lag{i}', f'B2_Available_N_lag{i}', f'B3_Available_N_lag{i}',
                      f'B4_Available_N_lag{i}']
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
            names += [f'T_lag{i}', f'Month_sin_lag{i}', f'Month_cos_lag{i}', f'Year_lag{i}',
                      f'TEC_N_Aver_lag{i}', f'B1_inWork_lag{i}', f'B2_inWork_lag{i}',
                      f'B1_GT11_inWork{i}', f'B1_GT12_inWork{i}', f'B2_GT21_inWork{i}', f'B2_GT22_inWork{i}',
                      f'B1_Available_Nmin_lag{i}', f'B2_Available_Nmin_lag{i}', f'TEC_Available_Nmin_lag{i}',
                      f'B1_Available_Nmax_lag{i}', f'B2_Available_Nmax_lag{i}', f'TEC_Available_Nmax_lag{i}',
                      ]
    agg = concat(cols, axis=1)
    agg.columns = names
    agg.dropna(inplace=True)

    data_with_lag = pd.concat([data, agg], axis=1)
    data_with_lag.dropna(inplace=True)
    return data_with_lag, base_features, weights_train
def prepare_meta_step_data(filepath, test_start_index, n_out, results,best_window_model_name, best_stat_model_name, best_step_model, test_idx):
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

        data['B1_inWork'] = np.where(data['B1_N_Aver'] > 125, 1, 0)
        data['B2_inWork'] = np.where(data['B2_N_Aver'] > 125, 1, 0)
        data['B3_inWork'] = np.where(data['B3_N_Aver'] > 125, 1, 0)
        data['B4_GT41_inWork'] = np.where(data['B4_GT41_N_Aver'] > 50, 1, 0)
        data['B4_GT42_inWork'] = np.where(data['B4_GT42_N_Aver'] > 50, 1, 0)
        data['B4_inWork'] = np.where(data['B4_N_Aver'] > 40, 1, 0)

        data.loc[data['B1_N_Aver'] < 125, 'B1_N_Aver'] = 0
        data.loc[data['B2_N_Aver'] < 125, 'B2_N_Aver'] = 0
        data.loc[data['B3_N_Aver'] < 125, 'B3_N_Aver'] = 0
        data.loc[data['B4_GT41_N_Aver'] < 50, 'B4_GT41_N_Aver'] = 0
        data.loc[data['B4_GT42_N_Aver'] < 50, 'B4_GT42_N_Aver'] = 0
        data.loc[data['B4_PT40_N_Aver'] < 40, 'B4_PT40_N_Aver'] = 0
        data.loc[data['TEC_N_Aver'] < 40, 'B1_N_Aver'] = 0

        #data = data.drop(data[(data['B1_N_Aver'] < 125) & (data['B1_N_Aver'] > 0)].index)
        #data = data.drop(data[(data['B2_N_Aver'] < 125) & (data['B2_N_Aver'] > 0)].index)
        #data = data.drop(data[(data['B3_N_Aver'] < 125) & (data['B3_N_Aver'] > 0)].index)
        #data = data.drop(data[(data['B4_GT41_N_Aver'] < 50) & (data['B4_GT41_N_Aver'] > 0)].index)
        #data = data.drop(data[(data['B4_GT42_N_Aver'] < 50) & (data['B4_GT42_N_Aver'] > 0)].index)
        #data = data.drop(data[(data['B4_PT40_N_Aver'] < 40) & (data['B4_PT40_N_Aver'] > 0)].index)
        #data = data.drop(data[(data['TEC_N_Aver'] == 0)].index)

        data['B1_Available_N'] = data['B1_inWork'] * 250
        data['B2_Available_N'] = data['B2_inWork'] * 250
        data['B3_Available_N'] = data['B3_inWork'] * 250
        # data['B4_Available_N'] = np.where(data['T'] < -2,
        #    (data['B4_GT41_inWork'] * 160 + data['B4_GT42_inWork'] * 160 + 72 *(data['B4_GT41_inWork'] + data['B4_GT42_inWork'])),
        #    (data['B4_GT41_inWork'] * 150 + data['B4_GT42_inWork'] * 150 + 72 *(data['B4_GT41_inWork'] + data['B4_GT42_inWork'])))
        data['B4_Available_N'] = (data['B4_GT41_inWork'] * 150 + data['B4_GT42_inWork'] * 150 + 72 * (
                data['B4_GT41_inWork'] + data['B4_GT42_inWork']))

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
        data['B1_Available_N_Prev'] = data['B1_Available_N'].shift(1)
        data['B2_Available_N_Prev'] = data['B2_Available_N'].shift(1)
        data['B3_Available_N_Prev'] = data['B3_Available_N'].shift(1)
        data['B4_Available_N_Prev'] = data['B4_Available_N'].shift(1)
        data['TEC_N_Aver_Prev'] = data['TEC_N_Aver'].shift(1)

        base_features = ['T', 'Month_sin', 'Month_cos', 'TEC_N_Aver_pred', 'B4_N_Aver_Prev',
                         'B1_inWork', 'B2_inWork', 'B3_inWork', 'B4_GT41_inWork', 'B4_GT42_inWork', 'B4_inWork',
                         'B1_Available_N', 'B2_Available_N', 'B3_Available_N', 'B4_Available_N',
                         'T_Prev', 'TEC_N_Aver_Prev']
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
        data['B1_inWork'] = np.where(data['B1_N_Aver'] > 50, 1, 0)
        data['B2_inWork'] = np.where(data['B2_N_Aver'] > 50, 1, 0)

        data.loc[data['B1_GT11_N_Aver'] < 30, 'B1_GT11_N_Aver'] = 0
        data.loc[data['B1_GT12_N_Aver'] < 30, 'B1_GT12_N_Aver'] = 0
        data.loc[data['B2_GT21_N_Aver'] < 30, 'B2_GT21_N_Aver'] = 0
        data.loc[data['B2_GT22_N_Aver'] < 30, 'B2_GT22_N_Aver'] = 0
        data.loc[data['B1_N_Aver'] < 50, 'B1_N_Aver'] = 0
        data.loc[data['B2_N_Aver'] < 50, 'B2_N_Aver'] = 0
        data.loc[data['TEC_N_Aver'] < 50, 'TEC_N_Aver'] = 0

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

        base_features = ['T', 'Month_sin', 'Month_cos',
                         'B1_Available_Nmin_Prev', 'B2_Available_Nmin_Prev','TEC_Available_Nmin_Prev',
                         'B1_Available_Nmax_Prev', 'B2_Available_Nmax_Prev','TEC_Available_Nmax_Prev',
                         'B1_Available_Nmin', 'B2_Available_Nmin', 'TEC_Available_Nmin',
                         'B1_Available_Nmax', 'B2_Available_Nmax', 'TEC_Available_Nmax',
                         'T_Prev', 'TEC_N_Aver_Prev']

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
            cols.append(df[['B1_Available_N']].shift(-i))
            cols.append(df[['B2_Available_N']].shift(-i))
            cols.append(df[['B3_Available_N']].shift(-i))
            cols.append(df[['B4_Available_N']].shift(-i))
            cols.append(df[['B1_N_Aver']].shift(-i))
            cols.append(df[['B2_N_Aver']].shift(-i))
            cols.append(df[['B3_N_Aver']].shift(-i))
            cols.append(df[['B4_N_Aver']].shift(-i))
            names += [f'T_lag{i}', f'Month_sin_lag{i}', f'Month_cos_lag{i}', f'Year_lag{i}',
                      f'TEC_N_Aver_pred_lag{i}', f'B1_inWork_lag{i}', f'B2_inWork_lag{i}', f'B3_inWork_lag{i}',
                      f'B4_GT41_inWork_lag{i}', f'B4_GT42_inWork_lag{i}', f'B4_inWork_lag{i}',
                      f'B1_Available_N_lag{i}', f'B2_Available_N_lag{i}', f'B3_Available_N_lag{i}',
                      f'B4_Available_N_lag{i}',
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
    if filepath == 'data/TEC22_Data.csv':
        step_features = base_features + [
            f'T_lag{step}',
            f'B1_inWork_lag{step}', f'B2_inWork_lag{step}', f'B3_inWork_lag{step}',
            f'B4_GT41_inWork_lag{step}', f'B4_GT42_inWork_lag{step}', f'B4_inWork_lag{step}',
            f'B1_Available_Nmin_lag{step}', f'B2_Available_Nmin_lag{step}',
            f'B3_Available_N_lag{step}', f'B4_Available_N_lag{step}'
        ]
    if filepath == 'data/TEC14_Data.csv':
        step_features = base_features + [
            f'T_lag{step}',
            f'B1_Available_Nmin_lag{step}', f'B2_Available_Nmin_lag{step}',f'TEC_Available_Nmin_lag{step}',
            f'B1_Available_Nmax_lag{step}', f'B2_Available_Nmax_lag{step}', f'TEC_Available_Nmax_lag{step}',
        ]

    x = data_with_lag[step_features].values
    y = data_with_lag.loc[:, [f'TEC_N_Aver_lag{step}']].values

    X_train_raw, X_test_raw = x[:train_size], x[train_size:]
    y_train_raw, y_test_raw = y[:train_size], y[train_size:]

    scaler_x = MinMaxScaler()
    X_train_scaled = scaler_x.fit_transform(X_train_raw)
    X_test_scaled = scaler_x.transform(X_test_raw)

    scaler_y = MinMaxScaler()
    y_train_scaled = scaler_y.fit_transform(y_train_raw)
    y_test_scaled = scaler_y.transform(y_test_raw)

    return (X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled,
            scaler_y, y_test_raw)
def prepare_meta_direct_data(filepath, data_with_lag, step, base_features, train_size, calc_goal):
    if filepath == 'data/TEC22_Data.csv':
        step_features = base_features + [
            f'T_lag{step}',
            f'B1_inWork_lag{step}', f'B2_inWork_lag{step}', f'B3_inWork_lag{step}',
            f'B4_GT41_inWork_lag{step}', f'B4_GT42_inWork_lag{step}', f'B4_inWork_lag{step}',
            f'B1_Available_N_lag{step}', f'B2_Available_N_lag{step}', f'B3_Available_N_lag{step}', f'B4_Available_N_lag{step}',
            f'TEC_N_Aver_pred_lag{step}'
        ]
    if filepath == 'data/TEC14_Data.csv':
        step_features = base_features + [
            f'T_lag{step}',
            f'B1_Available_Nmin_lag{step}', f'B2_Available_Nmin_lag{step}',f'TEC_Available_Nmin_lag{step}',
            f'B1_Available_Nmax_lag{step}', f'B2_Available_Nmax_lag{step}', f'TEC_Available_Nmax_lag{step}',
            f'TEC_N_Aver_pred_lag{step}'
        ]

    x = data_with_lag[step_features].values
    y = data_with_lag.loc[:, [calc_goal + f'_N_Aver_lag{step}']].values

    X_train_raw, X_test_raw = x[:train_size], x[train_size:]
    y_train_raw, y_test_raw = y[:train_size], y[train_size:]

    scaler_x = MinMaxScaler()
    X_train_scaled = scaler_x.fit_transform(X_train_raw)
    X_test_scaled = scaler_x.transform(X_test_raw)

    scaler_y = MinMaxScaler()
    y_train_scaled = scaler_y.fit_transform(y_train_raw)
    y_test_scaled = scaler_y.transform(y_test_raw)

    return (X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled,
            scaler_y, y_test_raw)

def optuna_cbr_search(trial, X_train, y_train, X_test, y_test, scaler_y):
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
    model.fit(X_train, y_train, eval_set=(X_test, y_test), use_best_model=True, callbacks=[pruning_callback])

    preds = model.predict(X_test)
    y_pred_unscaled = scaler_y.inverse_transform(preds.reshape(-1, 1))
    y_test_unscaled = scaler_y.inverse_transform(y_test.reshape(-1, 1))
    mae = metrics.mean_absolute_error(y_pred_unscaled, y_test_unscaled)

    return mae
def optuna_rfr_search(trial, X_train, y_train, X_test, y_test, scaler_y):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
        "max_features": trial.suggest_float("max_features", 0.1, 1.0),
        "n_jobs": -1
    }

    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train.ravel())

    preds = model.predict(X_test)
    yhat = scaler_y.inverse_transform(preds.reshape(-1, 1))
    y_true = scaler_y.inverse_transform(y_test.reshape(-1, 1))

    mae = metrics.mean_absolute_error(y_true, yhat)

    return mae
def optuna_lstm_search(trial, X_train, y_train, X_test, y_test, scaler_y, input_shape, y_train_s_combined, y_test_s_combined):
    #optuna.logging.set_verbosity(optuna.logging.WARNING)
    n_units_lstm = trial.suggest_int('n_units_lstm', 20, 150) if trial else 50
    n_units_dense = trial.suggest_int('n_units_dense', 10, 50) if trial else 25
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True) if trial else 0.001

    model = Sequential([
        LSTM(n_units_lstm, input_shape=input_shape),
        Dense(n_units_dense),
        Dense(1)
    ])
    optimizer = Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss=custom_loss)
    model.fit(X_train, y_train_s_combined, validation_data=(X_test, y_test_s_combined), epochs=10, batch_size=32, verbose=0)
    pred_lstm_s = model.predict(X_test)
    y_pred_unscaled = scaler_y.inverse_transform(pred_lstm_s)
    y_test_unscaled = scaler_y.inverse_transform(y_test)
    mae = metrics.mean_absolute_error(y_pred_unscaled, y_test_unscaled)
    return mae

@tf.keras.utils.register_keras_serializable()
def custom_loss(y_true_combined, y_pred):
    lambda_bounds = 25.0

    y_true = y_true_combined[:, 0:1]
    avail_nmax = y_true_combined[:, 1:2]
    avail_nmin = y_true_combined[:, 2:3]

    base_loss = tf.reduce_mean(tf.abs(y_true - y_pred))

    upper_penalty = tf.reduce_mean(tf.square(tf.maximum(0.0, y_pred - avail_nmax)))

    lower_penalty = tf.reduce_mean(tf.square(tf.maximum(0.0, avail_nmin - y_pred)))

    total_loss = base_loss + lambda_bounds * tf.reduce_mean(upper_penalty + lower_penalty)
    return total_loss

def scale_combined(combined_data, scaler):
    col0 = scaler.transform(combined_data[:, 0:1])
    col1 = scaler.transform(combined_data[:, 1:2])
    col2 = scaler.transform(combined_data[:, 2:3])
    return np.column_stack([col0, col1, col2])
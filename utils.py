import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def prepare_data(filepath, test_start_index):
    data = pd.read_csv(filepath, delimiter=';', parse_dates=['Date'], dayfirst=True)

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
    data['B4_inWork'] = np.where(data['B4_N_Aver'] > 125, 1, 0)

    data['B1_Available_N'] = data['B1_inWork'] * 250
    data['B2_Available_N'] = data['B2_inWork'] * 250
    data['B3_Available_N'] = data['B3_inWork'] * 250
    data['B4_Available_N'] = (data['B4_GT41_inWork'] * 150 + data['B4_GT42_inWork'] * 150 + 75 *
                              (data['B4_GT41_inWork'] + data['B4_GT42_inWork']))

    data.dropna(inplace=True)

    data = data[400:]

    data.set_index('Date', inplace=True)

    X = data.loc[:, ['T', 'B1_inWork', 'B2_inWork', 'B3_inWork', 'B4_GT41_inWork', 'B4_GT42_inWork', 'B4_inWork', 'Month_sin',
            'Month_cos', 'Year',
            'B1_Available_N', 'B2_Available_N', 'B3_Available_N', 'B4_Available_N']].values
    y = data.loc[:, ['TEC_N_Aver']].values

    n = test_start_index
    X_train, X_test = X[:n], X[n:]
    y_train, y_test = y[:n], y[n:]

    # 2. Масштабирование
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_train_s = scaler_x.fit_transform(X_train)
    X_test_s = scaler_x.transform(X_test)
    y_train_s = scaler_y.fit_transform(y_train)
    y_test_s = scaler_y.transform(y_test)

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

    Xw = data.loc[:,['T', 'Month_sin', 'Month_cos', 'Year',
                'B1_inWork', 'B2_inWork', 'B3_inWork' ,'B4_GT41_inWork', 'B4_GT42_inWork', 'B4_inWork',
                'B1_Available_N', 'B2_Available_N', 'B3_Available_N', 'B4_Available_N',
                'T_Prev', 'Month_sin_Prev', 'Month_cos_Prev', 'Year_Prev',
                'B4_GT41_N_Aver_Prev', 'B4_GT42_N_Aver_Prev' ,'B4_PT40_N_Aver_Prev', 'TEC_N_Aver_Prev',
                'B1_N_Aver_Prev', 'B2_N_Aver_Prev' ,'B3_N_Aver_Prev', 'B4_N_Aver_Prev',
                'B1_Available_N_Prev', 'B2_Available_N_Prev' ,'B3_Available_N_Prev', 'B4_Available_N_Prev']].values

    yw = data.loc[:, ['TEC_N_Aver']].values

    n = test_start_index

    Xw_train, Xw_test = Xw[:n], Xw[n:]
    yw_train, yw_test = yw[:n], yw[n:]

    # 2. Масштабирование
    scaler_xw = MinMaxScaler()
    scaler_yw = MinMaxScaler()

    Xw_train_s = scaler_xw.fit_transform(Xw_train)
    Xw_test_s = scaler_xw.transform(Xw_test)
    yw_train_s = scaler_yw.fit_transform(yw_train)
    yw_test_s = scaler_yw.transform(yw_test)

    return X_train_s, X_test_s, y_train_s, y_test_s, Xw_train_s, Xw_test_s, yw_train_s, yw_test_s, y_train, y_test, scaler_y, scaler_yw, data.index
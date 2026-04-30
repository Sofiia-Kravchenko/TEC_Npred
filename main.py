import os
import warnings

import pandas as pd
import tensorflow as tf
import argparse

from calc_body import calc_power_generation

tf.random.set_seed(42)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, default='data/TEC14_Data.csv', help='Путь к файлу данных') #data/TEC22_Data.csv data/TEC14_Data.csv
    parser.add_argument('--start', type=int, default=2000, help='Индекс начала тестовых данных')     #2700 #2000
    parser.add_argument('--forecast_window', type=int, default=14, help='окно прогноза')
    parser.add_argument('--target_power_unit', type=str, nargs='+', default=['B1', 'B2'], help='Цель предсказания')
    args = parser.parse_args()

    data_path = args.file
    test_start_index = args.start
    n_out = args.forecast_window
    model_name = os.path.basename(data_path).split('.')[0]
    checkpoint_dir = f'checkpoint/{model_name}/'
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(checkpoint_dir+'multistep', exist_ok=True)
    tec_results={}
    tg_results = {}
    results = {}
    test_idx = []
    best_window_model_name = ''
    best_stat_model_name = ''
    best_step_model = []

    calc_goal = 'TEC_N_Aver'
    checkpoint_unit_type = 'Npred'

    warnings.filterwarnings("ignore")

    tec_results, test_idx, best_window_model_name, best_stat_model_name, best_step_model = calc_power_generation(data_path, test_start_index, checkpoint_dir, checkpoint_unit_type, n_out, calc_goal, tec_results, test_idx, best_window_model_name, best_stat_model_name, best_step_model)
    results[calc_goal]=tec_results[best_stat_model_name]
    for calc_goal in args.target_power_unit:
        print(f"--- Prediction for: {calc_goal} ---")
        checkpoint_unit_type = calc_goal

        tg_results, test_idx, best_window_model_name, best_stat_model_name, best_step_model = calc_power_generation(data_path, test_start_index, checkpoint_dir, checkpoint_unit_type, n_out, calc_goal, tec_results, test_idx, best_window_model_name, best_stat_model_name, best_step_model)
    results[calc_goal]=tec_results[best_stat_model_name]

    df_report = pd.DataFrame(results)
    df_report.to_csv('rez_report.csv', index=False)

if __name__ == "__main__":
    main()

import os
import warnings

import tensorflow as tf
import argparse

from calc_body import calc_power_generation

tf.random.set_seed(42)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, default='data/TEC22_Data.csv', help='Путь к файлу данных') #data/TEC22_Data.csv data/TEC14_Data.csv
    parser.add_argument('--start', type=int, default=2200, help='Индекс начала тестовых данных')     #2562
    parser.add_argument('--forecast_window', type=int, default=14, help='окно прогноза')
    parser.add_argument('--target_power_unit', type=str, default='B4', help='Цель предсказания')
    args = parser.parse_args()

    data_path = args.file
    test_start_index = args.start
    n_out = args.forecast_window
    model_name = os.path.basename(data_path).split('.')[0]
    checkpoint_dir = f'checkpoint/{model_name}/'
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(checkpoint_dir+'multistep', exist_ok=True)
    results={}
    test_idx = []
    best_window_model_name = ''
    best_stat_model_name = ''
    best_step_model = []

    calc_goal = 'TEC_N_Aver'
    checkpoint_unit_type = 'Npred'

    warnings.filterwarnings("ignore")

    results, test_idx, best_window_model_name, best_stat_model_name, best_step_model = calc_power_generation(data_path, test_start_index, checkpoint_dir, checkpoint_unit_type, n_out, calc_goal, results, test_idx, best_window_model_name, best_stat_model_name, best_step_model)

    calc_goal = args.target_power_unit
    checkpoint_unit_type = calc_goal

    print("best_window_model_name:", best_window_model_name)
    print("best_stat_model_name:", best_stat_model_name)

    results, test_idx, best_window_model_name, best_stat_model_name, best_step_model = calc_power_generation(data_path, test_start_index, checkpoint_dir, checkpoint_unit_type, n_out, calc_goal, results, test_idx, best_window_model_name, best_stat_model_name, best_step_model)

if __name__ == "__main__":
    main()
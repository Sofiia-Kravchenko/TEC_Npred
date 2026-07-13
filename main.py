import os
import sys
import ctypes

sys.stderr = open(os.devnull, 'w')

try:
    libc = ctypes.CDLL(None)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    libc.dup2(devnull_fd, 2)
except Exception: pass

os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.stderr = open(os.devnull, 'w')

import warnings
import numpy as np
import pandas as pd
import tensorflow as tf
import argparse

from calc_body import calc_power_generation
from print_results import plot_final_graph
from utils import reconcile_with_mip, reconcile_with_l2
tf.keras.mixed_precision.set_global_policy('mixed_float16')

tf.random.set_seed(42)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, default='data/TEC22_Data.csv', help='Путь к файлу данных') #data/TEC22_Data.csv data/TEC14_Data.csv
    parser.add_argument('--start', type=int, default=2700, help='Индекс начала тестовых данных')     #2700 #2000
    parser.add_argument('--forecast_window', type=int, default=14, help='окно прогноза')
    parser.add_argument('--target_power_unit', type=str, nargs='+', default=['B1', 'B2', 'B3', 'B4'], help='Цель предсказания')
    parser.add_argument('--hierarchical_features', type=int, default=1, help='Иерархические признаки (0-да, 1-нет)')
    parser.add_argument('--cliping_and_customLoss', type=int, default=0, help='Кастомная обработка данных (0-да, 1-нет)')
    args = parser.parse_args()

    data_path = args.file
    test_start_index = args.start
    n_out = args.forecast_window
    hierarchical_features = args.hierarchical_features
    cliping_and_customLoss = args.cliping_and_customLoss
    model_name = os.path.basename(data_path).split('.')[0]

    suffix = f"_hf{hierarchical_features}_ccl{cliping_and_customLoss}"

    checkpoint_dir = f'checkpoint/{model_name}{suffix}/'
    reports_dir = f'data/reports/{model_name}{suffix}/'

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(checkpoint_dir+'multistep', exist_ok=True)
    tec_results={}
    tec_constraints = {}
    tg_results = {}
    tg_constraints = {}
    results = {}
    test_idx = []
    best_window_model_name = ''
    best_stat_model_name = ''
    best_step_model = []

    calc_goal = 'TEC'

    warnings.filterwarnings("ignore")

    tec_results, tec_constraints, test_idx, best_window_model_name, best_stat_model_name, best_step_model = calc_power_generation(data_path, test_start_index, checkpoint_dir, n_out, calc_goal, tec_results, test_idx, best_window_model_name, best_stat_model_name, best_step_model, reports_dir, hierarchical_features, cliping_and_customLoss)
    results[calc_goal+ '_N_Aver_pred']=tec_results[best_stat_model_name]
    results[calc_goal + '_Available_Nmin'] = tec_constraints[calc_goal + '_Available_Nmin']
    results[calc_goal + '_Available_Nmax'] = tec_constraints[calc_goal + '_Available_Nmax']
    for calc_goal in args.target_power_unit:
        print(f"--- Prediction for: {calc_goal} ---")

        tg_results, tg_constraints, test_idx, best_window_model_name, best_stat_model_name, best_step_model = calc_power_generation(data_path, test_start_index, checkpoint_dir, n_out, calc_goal, tec_results, test_idx, best_window_model_name, best_stat_model_name, best_step_model, reports_dir, hierarchical_features, cliping_and_customLoss)
        results[calc_goal + '_N_Aver_pred'] = tg_results[best_stat_model_name]
        results[calc_goal + '_Available_Nmin'] = tg_constraints[calc_goal + '_Available_Nmin']
        results[calc_goal + '_Available_Nmax'] = tg_constraints[calc_goal + '_Available_Nmax']
        print(calc_goal, 'best_stat_model_name:', best_stat_model_name)
        results = {k: np.array(v).flatten() for k, v in results.items()}
        df_report = pd.DataFrame(results)
    goal_mapping = {
        "data/TEC22_Data.csv": ["B1", "B2", "B3", "B4", "TEC_N_Aver_pred"],
        "data/TEC14_Data.csv": ["B1", "B2", "TEC_N_Aver_pred"]
    }

    elements = goal_mapping[data_path]

    tec_p = [e for e in elements if "TEC" in e][0]
    boiler_ps = [e for e in elements if e.startswith("B")]

    final_report = reconcile_with_l2(
        df_report,
        tec_col=tec_p,
        boiler_prefixes=boiler_ps
    )
    plot_final_graph(final_report)

    final_report.to_csv(reports_dir + 'final_tec_report_reconciled.csv', index=False, sep=';')
    print("Файл сохранен: final_tec_report_reconciled.csv")


if __name__ == "__main__":
    main()

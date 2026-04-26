import numpy as np
import pandas as pd
from sklearn import metrics

def print_stat_results(results, y_test, calc_goal):
    print("\n" + "=" * 50)
    print("FINAL MODEL COMPARISON REPORT")
    print("=" * 50)

    report = []
    min_len = min([len(p) for p in results.values()])
    for name, pred in results.items():
        y_pred_final = pred[-min_len:]
        y_true_final = y_test[-min_len:]

        mae = metrics.mean_absolute_error(y_true_final, y_pred_final)
        rmse = np.sqrt(metrics.mean_squared_error(y_true_final, y_pred_final))
        wape  = np.sum(np.abs(y_true_final - y_pred_final)) / np.sum(np.abs(y_true_final)) * 100
        r2 = metrics.r2_score(y_true_final, y_pred_final)

        report.append({
            'Model Name': name,
            'MAE': mae,
            'RMSE': rmse,
            'WAPE (%)': wape,
            'R2 Score': r2
        })

    df_report = pd.DataFrame(report)
    df_report = df_report.sort_values(by='MAE').reset_index(drop=True)

    print(df_report.to_string(index=False, float_format=lambda x: "{:.4f}".format(x)))
    if calc_goal == 'TEC_N_Aver':
        df_report.to_csv('Power_Station_model_evaluation_report.csv', index=False)
    else:
        df_report.to_csv(calc_goal+'_model_evaluation_report.csv', index=False)
    print("\n[INFO] Report saved to 'model_evaluation_report.csv'")

    min_len = min([len(p) for p in results.values()])
    export_df = pd.DataFrame({'Actual_Q': y_test[-min_len:].flatten()})

    for name, pred in results.items():
        export_df[name] = pred[-min_len:].flatten()

    if calc_goal == 'TEC_N_Aver':
        export_df.to_excel('Power_Station_final_predictions_comparison.xlsx', index=False)
    else:
        export_df.to_excel(calc_goal+'_final_predictions_comparison.xlsx', index=False)
    print("[INFO] Predictions exported to 'final_predictions_comparison.xlsx'")

    window_models = df_report[df_report['Model Name'].str.endswith('_window')]

    best_window_model_name = ''

    if not window_models.empty:
        best_model = window_models.iloc[0]

        best_window_model_name = best_model['Model Name']


    standard_models = df_report[~df_report['Model Name'].str.endswith('_window')]
    best_stat_model_name = ''

    if not standard_models.empty:
        best_model = standard_models.iloc[0]

        best_stat_model_name = best_model['Model Name']

    print("best_window_model_name:", best_window_model_name)
    print("best_stat_model_name:", best_stat_model_name)
    return best_window_model_name, best_stat_model_name

def print_step_results(lstm_multi_results, cbr_multi_results):
    best_step_model = ''
    print("\n" + "=" * 60)
    print("HORIZON ANALYSIS (LSTM vs CBR)")
    print("-" * 60)
    print(f"{'Day':<5} | {'LSTM MAE':<12} | {'LSTM MSE':<12} | {'LSTM WAPE':<12} | {'LSTM R2':<12} | {'CBR MAE':<12} | {'CBR MSE':<12} | {'CBR WAPE':<12} | {'CBR R2':<12}")

    for i in range(lstm_multi_results.shape[0]):
        print(f"{i + 1:<5} | {lstm_multi_results[i][0]:<12.4f} | {lstm_multi_results[i][1]:<12.4f} | {lstm_multi_results[i][2]:<12.4f} | {lstm_multi_results[i][3]:<12.4f} | {cbr_multi_results[i][0]:<12.4f} | {cbr_multi_results[i][1]:<12.4f} | {cbr_multi_results[i][2]:<12.4f} | {cbr_multi_results[i][3]:<12.4f}")

    if lstm_multi_results.shape[0] < cbr_multi_results[i][0]: best_step_model = 'lstm'
    else: best_step_model = 'CBR'
    return best_step_model
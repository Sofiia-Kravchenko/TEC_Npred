import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
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
    if calc_goal == 'TEC':
        df_report.to_csv('Power_Station_model_evaluation_report.csv', index=False)
    else:
        df_report.to_csv(calc_goal+'_model_evaluation_report.csv', index=False)
    print("\n[INFO] Report saved to 'model_evaluation_report.csv'")

    min_len = min([len(p) for p in results.values()])
    export_df = pd.DataFrame({'Actual_Q': y_test[-min_len:].flatten()})

    for name, pred in results.items():
        export_df[name] = pred[-min_len:].flatten()

    if calc_goal == 'TEC':
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
def plot_compare_models_violations(y_true_real, y_pred_custom, y_pred_mae, n_max, n_min, title_suffix=""):
    """
    Визуализирует и сравнивает предсказания Custom Loss и MAE моделей с подсветкой нарушений.

    y_true_real: реальные тестовые значения (инвертированные)
    y_pred_custom: "чистые" предсказания модели Custom Loss до обрезки
    y_pred_mae: предсказания модели MAE
    n_max, n_min: верхняя и нижняя границы
    """
    plt.figure(figsize=(25, 12))

    steps = np.arange(len(y_true_real))
    y_true = y_true_real.flatten()
    y_cust = y_pred_custom.flatten()
    y_mae = y_pred_mae.flatten()
    n_max_f = n_max.flatten()
    n_min_f = n_min.flatten()

    # 1. Фоновый коридор допустимых значений
    plt.fill_between(steps, n_min_f, n_max_f, color='gray', alpha=0.06, label='Valid Zone')

    # 2. Отрисовка коридора границ (пунктиром)
    plt.plot(steps, n_max_f, '--', color='#e74c3c', alpha=0.5, linewidth=2.4, label='Upper Bound (n_max)')
    plt.plot(steps, n_min_f, '--', color='#9b59b6', alpha=0.5, linewidth=2.4, label='Lower Bound (n_min)')

    # 3. Базовая линия истинных значений
    plt.plot(steps, y_true, label='True Values', color='#2ed573', alpha=0.7, linewidth=3.0)

    # --- МОДЕЛЬ CUSTOM LOSS ---
    plt.plot(steps, y_cust, label='Predictions (Custom Loss)', color='#1e90ff', linewidth=4.0)

    cust_over = y_cust > n_max_f
    cust_under = y_cust < n_min_f
    if np.any(cust_over):
        plt.scatter(steps[cust_over], y_cust[cust_over], color='#0056b3', marker='v', s=140, zorder=5,
                    label=f'Custom: Over Max ({np.sum(cust_over)} pts)')
    if np.any(cust_under):
        plt.scatter(steps[cust_under], y_cust[cust_under], color='#0056b3', marker='^', s=140, zorder=5,
                    label=f'Custom: Under Min ({np.sum(cust_under)} pts)')

    # --- МОДЕЛЬ MAE ---
    plt.plot(steps, y_mae, label='Predictions (MAE Loss)', color='#f39c12', linewidth=3.6, linestyle='-.')

    mae_over = y_mae > n_max_f
    mae_over_under = y_mae < n_min_f
    if np.any(mae_over):
        plt.scatter(steps[mae_over], y_mae[mae_over], color='#d35400', marker='v', s=140, zorder=4,
                    label=f'MAE: Over Max ({np.sum(mae_over)} pts)')
    if np.any(mae_over_under):
        plt.scatter(steps[mae_over_under], y_mae[mae_over_under], color='#d35400', marker='^', s=140, zorder=4,
                    label=f'MAE: Under Min ({np.sum(mae_over_under)} pts)')

    # Оформление (размеры шрифтов увеличены в 2 раза)
    plt.title(f"Comparison of Boundary Violations: Custom vs MAE {title_suffix}", fontsize=28, fontweight='bold',
              pad=30)
    plt.xlabel("Time Steps / Samples", fontsize=24, labelpad=15)
    plt.ylabel("Value", fontsize=24, labelpad=15)

    # Увеличение шрифта осей чисел (ticks)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    # Настройка сетки и легенды
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9, shadow=True, fontsize=20)

    # Делаем засечки на оси X целыми числами (шаг 2 для читаемости)
    plt.xticks()

    plt.tight_layout()
    plt.show()
def plot_single_model_violations(y_true_real, y_pred, n_max, n_min, model_name, title_suffix=""):
    """
    Визуализирует предсказания ОДНОЙ модели с красивым оформлением,
    крупными шрифтами и подсветкой нарушений границ для индексов 330-350.
    """
    # Мягкая современная цветовая палитра
    color_true = '#2ecc71'  # приятный зеленый
    color_pred = '#3498db'  # технологичный синий
    color_max = '#e74c3c'  # пастельный красный
    color_min = '#9b59b6'  # пастельный фиолетовый
    color_violation = '#e67e22'  # контрастный оранжевый для нарушений

    plt.figure(figsize=(24, 12), facecolor='#fafafa')
    ax = plt.subplot(111, facecolor='#ffffff')

    # Срезы для диапазона индексов от 330 до 350 (включительно)
    start_idx, end_idx = 330, 351

    steps = np.arange(len(y_true_real))
    y_true = y_true_real.flatten()
    y_p = y_pred.flatten()
    n_max_f = n_max.flatten()
    n_min_f = n_min.flatten()

    # 1. Красивый мягкий коридор допустимых значений
    ax.fill_between(steps, n_min_f, n_max_f, color='#7f8c8d', alpha=0.08, label='Valid Zone')

    # 2. Пунктирные границы коридора
    ax.plot(steps, n_max_f, ':', color=color_max, alpha=0.7, linewidth=3.0, label='Upper Bound (n_max)')
    ax.plot(steps, n_min_f, ':', color=color_min, alpha=0.7, linewidth=3.0, label='Lower Bound (n_min)')

    # 3. Линия истинных значений (сзади прогноза)
    ax.plot(steps, y_true, label='True Values', color=color_true, alpha=0.6, linewidth=4.0)

    # 4. Линия предсказаний модели
    ax.plot(steps, y_p, label=f'Predictions ({model_name})', color=color_pred, linewidth=5.0, zorder=3)

    # 5. Поиск нарушений и их красивая отрисовка с белой обводкой (edgecolor)
    p_over = y_p > n_max_f
    p_under = y_p < n_min_f

    ax.set_title(f"Boundary Violations Analysis: {model_name} {title_suffix}", fontsize=32, fontweight='bold', pad=35,
                 color='#2c3e50')
    ax.set_xlabel("Time Steps", fontsize=24, labelpad=20, color='#34495e')
    ax.set_ylabel("Power, MW", fontsize=24, labelpad=20, color='#34495e')

    # Настройка чисел на осях
    plt.xticks(fontsize=20, color='#7f8c8d')
    plt.yticks(fontsize=20, color='#7f8c8d')

    # Эстетичная сетка
    ax.grid(True, linestyle='--', alpha=0.3, color='#bdc3c7')

    # Современный дизайн рамки: убираем верхнюю и правую линии, оставляя воздух
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#bdc3c7')
    ax.spines['bottom'].set_color('#bdc3c7')

    # Красивая и аккуратная легенда
    ax.legend(loc='upper right', frameon=True, facecolor='#ffffff', framealpha=0.95,
              edgecolor='#e2e8f0', shadow=False, fontsize=18, labelspacing=0.6)

    plt.tight_layout()
    plt.show()
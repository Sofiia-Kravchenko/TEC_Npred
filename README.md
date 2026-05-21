# TEC_Npred-master: Power Plant Generation Forecasting System

This software complex is designed for multi-step forecasting of electrical power generation at Thermal Power Plants (e.g., TEC-14, TEC-22) and individual power blocks ($B1, B2$, etc.), taking into account dynamic technical and environmental constraints, with subsequent target reconciliation.

## 🛠 Architecture & Forecasting Approach

The forecasting pipeline is implemented inside the `calc_power_generation` module. It evaluates and scores three distinct modeling approaches:
1. **Statistical Baseline** — Traditional statistical relationships to capture the overall power generation trend.
2. **Window-based Models** — Time-series models utilizing rolling windows to capture short-term operational dynamics.
3. **Direct Multi-step Models** — Specialized architectures located in the `forecast_models/` directory, trained to forecast directly over the user-defined `forecast_window`:
   * **CBRDirectForecast** / **CBRMetaDirectForecast** — Gradient boosting models powered by CatBoost.
   * **LSTMDirectForecast** / **LSTMMetaDirectForecast** — Recurrent neural network models powered by Long Short-Term Memory (LSTM) layers.

### 📊 Data Engineering & Constraint Integration ($N_{min} / N_{max}$)
A core feature of the pipeline is the robust integration of operational boundaries for each individual generating unit as well as the station as a whole. 
* **Pre-calculation**: During the initial **Data Engineering stage**, the system pre-calculates the minimum and maximum allowable operating ranges based on both equipment technical limitations and real-time weather conditions.
* **Three-Fold Application**: These dynamic constraints are deeply embedded throughout the entire machine learning and optimization lifecycle:
  1. **Model Features**: Used directly as input features to provide the models with context on physical boundaries.
  2. **Custom Loss Functions**: Integrated into custom AI loss formulations to heavily penalize predictions that violate safe operating thresholds.
  3. **MIP Constraints**: Utilized as strict mathematical boundaries during the final Mixed-Integer Programming (MIP) balance reconciliation stage.

### 🧠 Automated Hyperparameter Tuning (Optuna)
To maximize prediction accuracy, the framework incorporates an automated hyperparameter optimization pipeline:
* **First Run / Cold Start**: If no prior configuration exists, the system automatically triggers an optimization study using **Optuna** to find the ideal hyperparameters for each model architecture.
* **Subsequent Runs**: Once the optimal parameters are found, they are serialized and saved into the `checkpoint/` directory. On subsequent executions, the pipeline automatically detects and loads these saved configurations, bypassing the time-consuming tuning stage to ensure fast execution.

### Key Features:
* **Exploratory Data Analysis**: Automatically generates a correlation matrix for primary parameters to identify key feature relationships.
* **Validation & Metrics**: Evaluates model performance on the test set and exports granular accuracy reports.
* **Visualization**: Plots interactive comparative charts for the best-performing window and statistical models, highlighting any boundary violations.
* **MIP Reconciliation**: Performs final mathematical alignment between the aggregated plant forecast and individual block forecasts using Mixed-Integer Linear Programming (`reconcile_with_mip`).

---

## 📁 Project Structure

```text
TEC_Npred-master/
├── .venv/                     # Python virtual environment
├── catboost_info/             # CatBoost training logs and temporary files
├── checkpoint/                # Serialized model checkpoints and saved Optuna hyperparameter configs
│   ├── TEC14_Data/            # Trained weights & configs for TEC-14 (including multistep)
│   └── TEC22_Data/            # Trained weights & configs for TEC-22 (including multistep)
├── data/                      # Raw datasets and generated reports
│   ├── reports/               # Model evaluation metrics and output reports
│   ├── TEC14_Data.csv         # Time-series telemetry for TEC-14
│   └── TEC22_Data.csv         # Time-series telemetry for TEC-22
├── forecast_models/           # Custom predictive model architectures
│   ├── CBRDirectForecast.py       # Direct CatBoost forecasting logic
│   ├── CBRMetaDirectForecast.py   # Meta-modeling utilizing CatBoost
│   ├── LSTMDirectForecast.py      # Direct LSTM forecasting logic
│   ├── LSTMMetaDirectForecast.py  # Meta-modeling utilizing LSTM
│   └── models.py                  # Shared helper classes for models
├── notebooks/                 # Jupyter Notebooks for research & prototyping
├── calc_body.py               # Core calculation engine (calc_power_generation)
├── data.xlsx                  # Supplementary aggregated data assets
├── main.py                    # Main pipeline orchestration script
├── print_results.py           # Module responsible for plotting and formatting
└── utils.py                   # Shared utility functions and helpers
```

---

## ⚙️ Installation & Setup

Follow these steps to set up the local development environment and run the pipeline.

### Prerequisites
* **Python**: `3.9` or higher (recommended `3.10` / `3.11`)
* **OS**: Windows, Linux, or macOS

### Step 1: Clone the Repository
```bash
git clone https://github.com
cd TEC_Npred-master
```

### Step 2: Initialize Virtual Environment
It is highly recommended to isolate your dependencies using a virtual environment:
```bash
# On Windows
python -m venv .venv
.venv\Scripts\activate

# On macOS/Linux
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Required Packages
Install the core libraries for data processing, machine learning, and mathematical optimization:
```bash
pip install -r requirements.txt
```

---

## 🚀 Execution & Command-Line Arguments

The main computation pipeline is triggered via `main.py`. The script accepts the following configuration arguments:

```bash
python main.py [arguments]
```

### Available Flags:
* `--file` (str): Path to the target CSV data file.
  * *Default:* `'data/TEC14_Data.csv'` (Alternative: `'data/TEC22_Data.csv'`).
* `--start` (int): Index pinpointing where the test dataset partition begins.
  * *Default:* `2000`.
* `--forecast_window` (int): Forecasting horizon depth measured in time-steps.
  * *Default:* `14`.
* `--target_power_unit` (list of str): Power blocks targeted for independent prediction tracking.
  * *Default:* `['B1', 'B2']` (auto-mapped based on the chosen station data).

### Example Command:
```bash
python main.py --file data/TEC22_Data.csv --start 2700 --forecast_window 24 --target_power_unit B1 B2 B3 B4
```

---

## 📊 Pipeline Artifacts & Output

Upon successful execution, the script produces the following outputs:
1. **Intermediate Summary (`df_report.xlsx`)**: Raw compiled predictions mapped alongside corresponding $N_{min}$ and $N_{max}$ limits.
2. **Evaluation Directory**: Comprehensive accuracy metrics, model comparison charts, and correlation plots are saved into `data/reports/{station_name}/`.
3. **Reconciled Master Report (`final_tec_report_reconciled.csv`)**: Mathematically balanced and constraint-verified final report, exported using a semicolon `;` delimiter.

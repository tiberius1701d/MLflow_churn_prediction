import logging
import os
import warnings

import matplotlib
import mlflow
from sklearn.ensemble import RandomForestClassifier

from loader import get_train_test_split_data

# --- Plotting ---
# Draw plots into files only, never into a window.
# MLflow autologging saves plots (e.g. the confusion matrix) as artifacts,
# so no graphical display is needed.
matplotlib.use("Agg")

# --- Configuration ---
DATA_PATH = "data/telco_churn.csv"
N_ESTIMATORS = 200
MAX_DEPTH = 15
EXPERIMENT_NAME = "Churn_Prediction_Basic"

# --- Setup Logging ---
# Silence verbose MLflow and Alembic logs
logging.getLogger("mlflow").setLevel(logging.ERROR)
logging.getLogger("alembic").setLevel(logging.ERROR)
# Ignore specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module="mlflow")

# MLflow setup

def train():
    # 1. Load Data
    print("Loading data...")
    X_train, X_test, y_train, y_test = get_train_test_split_data(DATA_PATH)

    # 2. Setup MLflow
    # Define experiment name
    mlflow.set_experiment(EXPERIMENT_NAME)
 
    
    # Enable Autologging (disable system metrics for cleaner output)
    # This captures params, metrics, model artifacts, and system metrics automatically
    mlflow.sklearn.autolog(log_model_signatures=True,
                           log_input_examples=True,
                           silent=True,
    )

    with mlflow.start_run(run_name="Model_Training"):
        print("Starting training run ...")

        # 3. Model Training
        rf = RandomForestClassifier(n_estimators=N_ESTIMATORS,
                                    max_depth=MAX_DEPTH,
                                    random_state=42,
        )
        rf.fit(X_train, y_train)

        print(f"Run complete! Artifacts saved to 'mlruns'")

if __name__ == "__main__":
    train()

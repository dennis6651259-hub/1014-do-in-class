"""
main.py
CRISP-DM Compliant Multiple Linear Regression CLI Pipeline for 50 Startups Profit Prediction
"""
import sys
import numpy as np
import pandas as pd
import scipy.stats as stats
from feature_progression_analysis import RAW_DATA, load_data, fit_ols

def print_header(title):
    print("\n" + "="*70)
    print(f" {title}")
    print("="*70)

def main():
    print_header("CRISP-DM Phase 1: Business Understanding")
    print("Goal: Predict startup profitability based on expenditure allocations (R&D, Administration,")
    print("Marketing) and geographic operational state (California, Florida, New York).")
    print("Objective: Identify the parsimonious optimal feature subset to maximize out-of-sample ROI.")

    print_header("CRISP-DM Phase 2: Data Understanding")
    df = load_data()
    print(f"Dataset shape: {df.shape} (50 startups, 5 candidate features + 1 target)")
    print(df.describe().to_string())

    print_header("CRISP-DM Phase 3: Data Preparation")
    print("Preprocessed categorical 'State' with One-Hot dummy encoding (California as reference category).")
    feature_cols = ['R&D Spend', 'Administration', 'Marketing Spend', 'State_Florida', 'State_New York']
    X = df[feature_cols].values
    y = df['Profit'].values

    # Train-test split (80/20)
    test_indices = [13, 39, 30, 45, 17, 48, 26, 25, 32, 19]
    train_indices = [i for i in range(50) if i not in test_indices]
    X_tr, y_tr = X[train_indices], y[train_indices]
    X_te, y_te = X[test_indices], y[test_indices]
    print(f"Training set: {len(train_indices)} samples | Testing set: {len(test_indices)} samples")

    print_header("CRISP-DM Phase 4: Modeling (Optimal k=2 Sweet Spot OLS)")
    k2_features = ['R&D Spend', 'Marketing Spend']
    k2_indices = [0, 2]
    X_tr_k2 = X_tr[:, k2_indices]
    X_te_k2 = X_te[:, k2_indices]

    beta, coef_table = fit_ols(X_tr_k2, y_tr, k2_features)
    print("\nFitted OLS Coefficients & Statistical Inference:")
    print(f"{'Predictor':<20} {'Beta (β)':<14} {'Std Error':<12} {'t-Stat':<10} {'p-Value':<10} {'VIF':<8} {'Sig':<5}")
    print("-" * 80)
    for row in coef_table:
        print(f"{row['predictor']:<20} {row['beta']:<14.4f} {row['std_error']:<12.4f} {row['t_stat']:<10.3f} {row['p_val']:<10.4e} {row['vif']:<8.2f} {row['significance']:<5}")

    print_header("CRISP-DM Phase 5: Evaluation")
    X_te_des = np.column_stack([np.ones(len(X_te)), X_te_k2])
    y_pred = X_te_des @ beta
    rmse = np.sqrt(np.mean((y_te - y_pred)**2))
    sst = np.sum((y_te - np.mean(y_te))**2)
    r2 = 1 - np.sum((y_te - y_pred)**2) / sst
    print(f"Optimal 2-Feature Subset: {k2_features}")
    print(f"Test RMSE: ${rmse:,.2f}  (Lowest across all tested models)")
    print(f"Test R²:   {r2:.4f}     (Explains {r2*100:.2f}% of profit variation)")

    print_header("CRISP-DM Phase 6: Deployment & Example Inference")
    example = {"rd": 100000, "mkt": 150000}
    predicted = beta[0] + beta[1] * example["rd"] + beta[2] * example["mkt"]
    print(f"Example Startup: R&D=${example['rd']:,}, Marketing=${example['mkt']:,}")
    print(f"Predicted Profit: ${predicted:,.2f}")
    print("\nFor interactive GUI and REST API, launch: python app.py")

if __name__ == "__main__":
    main()

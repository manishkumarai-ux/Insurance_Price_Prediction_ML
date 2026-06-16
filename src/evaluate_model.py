import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
mean_absolute_error,
mean_squared_error,
r2_score,
root_mean_squared_error
)

from data_preprocessing import *

# --------------------------------------------------

# Load Data

# --------------------------------------------------

df = load_data("../data/insurance.csv")

df_encoded = encode_features(df)

X, y = prepare_features_target(df_encoded)

X_train, X_test, y_train, y_test = split_data(
X,
y
)

y_train_log, y_test_log = apply_log_transformation(
y_train,
y_test
)

X_train_scaled, X_test_scaled, scaler = scale_features(
X_train,
X_test
)

# --------------------------------------------------

# Load Saved Model

# --------------------------------------------------

model = pickle.load(
open(
"../models/best_model.pkl",
"rb"
)
)

# --------------------------------------------------

# Predictions

# --------------------------------------------------

y_pred_log = model.predict(
X_test_scaled
)

# Convert predictions back to original scale

y_pred = np.exp(y_pred_log)

# --------------------------------------------------

# Evaluation Metrics

# --------------------------------------------------

mae = mean_absolute_error(
y_test,
y_pred
)

mse = mean_squared_error(
y_test,
y_pred
)

rmse = root_mean_squared_error(
y_test,
y_pred
)

r2 = r2_score(
y_test,
y_pred
)

print("\nModel Evaluation Results\n")

print(f"R2 Score : {r2:.4f}")
print(f"MAE      : {mae:.2f}")
print(f"MSE      : {mse:.2f}")
print(f"RMSE     : {rmse:.2f}")

# --------------------------------------------------

# Actual vs Predicted Plot

# --------------------------------------------------

plt.figure(figsize=(8,6))

sns.scatterplot(
x=y_test,
y=y_pred
)

plt.xlabel("Actual Expenses")
plt.ylabel("Predicted Expenses")

plt.title(
"Actual vs Predicted Expenses"
)

plt.tight_layout()

plt.savefig(
"../images/actual_vs_predicted.png"
)

plt.show()

# --------------------------------------------------

# Residual Analysis

# --------------------------------------------------

residuals = y_test - y_pred

plt.figure(figsize=(8,6))

sns.histplot(
residuals,
kde=True
)

plt.title(
"Residual Distribution"
)

plt.xlabel(
"Prediction Error"
)

plt.tight_layout()

plt.savefig(
"../images/residual_distribution.png"
)

plt.show()

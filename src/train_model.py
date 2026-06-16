import pandas as pd
import pickle

from sklearn.linear_model import (
LinearRegression,
Ridge,
Lasso,
ElasticNet
)

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import (
RandomForestRegressor,
GradientBoostingRegressor
)

from sklearn.metrics import (
mean_absolute_error,
mean_squared_error,
r2_score,
root_mean_squared_error
)

from data_preprocessing import *

# --------------------------------------------------

# Load and Prepare Data

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

# Models Dictionary

# --------------------------------------------------

models = {

```
"Linear Regression":
    LinearRegression(),

"Ridge Regression":
    Ridge(),

"Lasso Regression":
    Lasso(),

"ElasticNet Regression":
    ElasticNet(),

"Decision Tree":
    DecisionTreeRegressor(
        random_state=42
    ),

"Random Forest":
    RandomForestRegressor(
        random_state=42
    ),

"Gradient Boosting":
    GradientBoostingRegressor(
        random_state=42
    )
```

}

# --------------------------------------------------

# Train Models

# --------------------------------------------------

results = {}

best_model = None
best_r2 = -999

for name, model in models.items():

```
model.fit(
    X_train_scaled,
    y_train_log
)

y_pred_log = model.predict(
    X_test_scaled
)

mae = mean_absolute_error(
    y_test_log,
    y_pred_log
)

mse = mean_squared_error(
    y_test_log,
    y_pred_log
)

rmse = root_mean_squared_error(
    y_test_log,
    y_pred_log
)

r2 = r2_score(
    y_test_log,
    y_pred_log
)

results[name] = {

    "R2": round(r2, 4),
    "MAE": round(mae, 4),
    "MSE": round(mse, 4),
    "RMSE": round(rmse, 4)
}

if r2 > best_r2:

    best_r2 = r2
    best_model = model
```

# --------------------------------------------------

# Results DataFrame

# --------------------------------------------------

results_df = pd.DataFrame(
results
).T

results_df = results_df.sort_values(
by="R2",
ascending=False
)

print("\nModel Performance Comparison\n")

print(results_df)

# --------------------------------------------------

# Save Best Model

# --------------------------------------------------

pickle.dump(
best_model,
open(
"../models/best_model.pkl",
"wb"
)
)

print("\nBest model saved successfully.")

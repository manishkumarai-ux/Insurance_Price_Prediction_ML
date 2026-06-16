import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
"""
Load insurance dataset
"""
df = pd.read_csv(file_path)

```
print(f"Dataset Shape: {df.shape}")

return df
```

def check_data(df):
"""
Basic data inspection
"""
print("\nDataset Information")
print(df.info())

```
print("\nMissing Values")
print(df.isnull().sum())

print("\nDuplicate Records")
print(df.duplicated().sum())
```

def encode_features(df):
"""
Convert categorical features into numeric format
"""

```
df_encoded = pd.get_dummies(
    df,
    drop_first=True
)

return df_encoded
```

def prepare_features_target(df_encoded):
"""
Split features and target variable
"""

```
X = df_encoded.drop(
    "expenses",
    axis=1
)

y = df_encoded["expenses"]

return X, y
```

def split_data(X, y):
"""
Train-Test Split
"""

```
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

return X_train, X_test, y_train, y_test
```

def apply_log_transformation(y_train, y_test):
"""
Log transformation for target variable
"""

```
y_train_log = np.log(y_train)

y_test_log = np.log(y_test)

return y_train_log, y_test_log
```

def scale_features(X_train, X_test):
"""
Feature Scaling using StandardScaler
"""

```
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)

return (
    X_train_scaled,
    X_test_scaled,
    scaler
)
```

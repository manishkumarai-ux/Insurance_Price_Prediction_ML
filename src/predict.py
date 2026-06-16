import pickle
import numpy as np
import pandas as pd

# --------------------------------------------------

# Load Saved Objects

# --------------------------------------------------

model = pickle.load(
open(
"../models/best_model.pkl",
"rb"
)
)

scaler = pickle.load(
open(
"../models/scaler.pkl",
"rb"
)
)

# --------------------------------------------------

# User Input

# --------------------------------------------------

customer_data = {

```
"age": 35,
"bmi": 28.5,
"children": 2,

"sex_male": 1,

"smoker_yes": 0,

"region_northwest": 1,
"region_southeast": 0,
"region_southwest": 0
```

}

# --------------------------------------------------

# Convert to DataFrame

# --------------------------------------------------

input_df = pd.DataFrame(
[customer_data]
)

# --------------------------------------------------

# Scale Features

# --------------------------------------------------

input_scaled = scaler.transform(
input_df
)

# --------------------------------------------------

# Prediction

# --------------------------------------------------

pred_log = model.predict(
input_scaled
)

predicted_expense = np.exp(
pred_log
)

print(
f"\nPredicted Insurance Expense: ${predicted_expense[0]:,.2f}"
)

import pandas as pd
import joblib
import os

# ==========================
# Load Model
# ==========================

model = joblib.load("models/xgboost_model.pkl")

# ==========================
# Load Test Dataset
# ==========================

test = pd.read_csv("data/test_nfaJ3J5.csv")

print(test.columns)

# ==========================
# Convert Object Columns
# ==========================

for col in test.columns:
    if test[col].dtype == 'object':
        test[col] = test[col].astype('category').cat.codes

# ==========================
# Create Dummy Features
# ==========================

test['lag_1'] = 0
test['lag_2'] = 0
test['rolling_mean_3'] = 0

# ==========================
# Prediction Features
# ==========================

X_test = test[[
    'week',
    'store_id',
    'sku_id',
    'total_price',
    'base_price',
    'is_featured_sku',
    'is_display_sku',
    'lag_1',
    'lag_2',
    'rolling_mean_3'
]]

# ==========================
# Predict
# ==========================

predictions = model.predict(X_test)

# ==========================
# Create Submission File
# ==========================

submission = pd.DataFrame({
    'id': test['record_ID'],
    'units_sold': predictions
})

# ==========================
# Save Predictions
# ==========================

os.makedirs("outputs", exist_ok=True)

submission.to_csv(
    "outputs/predictions.csv",
    index=False
)

print("Predictions saved successfully!")
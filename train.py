import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from xgboost import XGBRegressor
import joblib
import os

# Load dataset
df = pd.read_csv("data/final_train.csv")

# Print data types
print(df.dtypes)

# ==========================
# Convert object columns
# ==========================

df['week'] = pd.to_numeric(df['week'], errors='coerce')

# Remove missing values
df.dropna(inplace=True)

# ==========================
# Features
# ==========================

X = df[[
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
# Target
# ==========================

y = df['units_sold']

# ==========================
# Train Test Split
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==========================
# Model
# ==========================

model = XGBRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=8,
    random_state=42
)

# ==========================
# Train
# ==========================

model.fit(X_train, y_train)

# ==========================
# Predict
# ==========================

predictions = model.predict(X_test)

# ==========================
# Evaluate
# ==========================

mae = mean_absolute_error(y_test, predictions)

print(f"MAE: {mae}")

rsme = root_mean_squared_error(y_test, predictions)
print(f"RSME: {rsme}")

# ==========================
# Save Model
# ==========================

os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/xgboost_model.pkl")

print("Model saved successfully!")
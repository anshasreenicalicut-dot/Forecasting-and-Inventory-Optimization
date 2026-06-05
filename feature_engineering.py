import pandas as pd

# Load dataset
df = pd.read_csv("data/cleaned_train.csv")

print("Original Shape:", df.shape)

# ==========================
# Convert object columns
# ==========================

for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].astype('category').cat.codes

# ==========================
# Create Features
# ==========================

df['lag_1'] = df['units_sold'].shift(1)

df['lag_2'] = df['units_sold'].shift(2)

df['rolling_mean_3'] = (
    df['units_sold']
    .rolling(window=3)
    .mean()
)

# ==========================
# Fill missing values
# ==========================

df.fillna(0, inplace=True)

print("Final Shape:", df.shape)

# Save dataset
df.to_csv("data/final_train.csv", index=False)

print("Feature engineering completed!")
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import autocorrelation_plot

# ==========================================
# LOAD DATA
# ==========================================

df = pd.read_csv("data/cleaned_train.csv")

print("=" * 50)
print("DATASET SHAPE")
print("=" * 50)
print(df.shape)

print("\nDATASET INFO")
print("=" * 50)
print(df.info())

print("\nFIRST 5 ROWS")
print("=" * 50)
print(df.head())

# ==========================================
# MISSING VALUES
# ==========================================

print("\nMISSING VALUES")
print("=" * 50)
print(df.isnull().sum())

# ==========================================
# DESCRIPTIVE STATISTICS
# ==========================================

print("\nSUMMARY STATISTICS")
print("=" * 50)
print(df.describe())

# ==========================================
# SALES TREND ANALYSIS
# ==========================================

weekly_sales = df.groupby('week')['units_sold'].sum()

plt.figure(figsize=(12, 5))
plt.plot(weekly_sales)
plt.title("Weekly Sales Trend")
plt.xlabel("Week")
plt.ylabel("Units Sold")
plt.grid(True)
plt.show()

# ==========================================
# MOVING AVERAGE
# ==========================================

moving_avg = weekly_sales.rolling(window=4).mean()

plt.figure(figsize=(12, 5))
plt.plot(weekly_sales, label='Actual Sales')
plt.plot(moving_avg, label='4-Week Moving Average')
plt.title("Sales Trend with Moving Average")
plt.xlabel("Week")
plt.ylabel("Units Sold")
plt.legend()
plt.grid(True)
plt.show()

# ==========================================
# SALES DISTRIBUTION
# ==========================================

plt.figure(figsize=(8, 5))
sns.histplot(df['units_sold'], bins=30, kde=True)
plt.title("Distribution of Units Sold")
plt.show()

# ==========================================
# CORRELATION HEATMAP
# ==========================================

numeric_df = df.select_dtypes(include=['int64', 'float64'])

plt.figure(figsize=(12, 8))
sns.heatmap(
    numeric_df.corr(),
    annot=False,
    cmap="coolwarm"
)
plt.title("Correlation Heatmap")
plt.show()

# ==========================================
# AUTOCORRELATION
# ==========================================

plt.figure(figsize=(10, 5))
autocorrelation_plot(weekly_sales)
plt.title("Autocorrelation Plot")
plt.show()

print("\nEDA COMPLETED SUCCESSFULLY!")
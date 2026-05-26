import pandas as pd

# Load dataset
train = pd.read_csv("data/train_0irEZ2H.csv")

# Display columns
print(train.columns)

# Check missing values
print(train.isnull().sum())

# Sort by week
train = train.sort_values(by='week')

# Save cleaned data
train.to_csv("data/cleaned_train.csv", index=False)

print("Preprocessing completed!")
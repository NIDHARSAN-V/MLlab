import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt  
# Sample DataFrame
data = {
    "Area": [1200, 1500, 800, 1100, 1600, 1000, 1400, 900],
    "Location": ["Suburb", "City", "Village", "Suburb", "City", "Village", "City", "Suburb"],
    "Bedrooms": [3, 4, 2, 3, 4, 2, 3, 2],
    "Price": [200000, 300000, 100000, 180000, 320000, 120000, 280000, 150000]
}

df = pd.DataFrame(data)
df = df.dropna()

# Show columns and choose target
print("Columns:", list(df.columns))
target = input("Enter target column (e.g., Price): ")

# Separate features and target
X_raw = df.drop(columns=[target])
y = df[target]

# One-hot encoding
X = pd.get_dummies(X_raw)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Take user input without loop
print("\nEnter values for prediction:")
area = int(input("Enter Area (e.g., 1300): "))
location = input("Enter Location (City/Suburb/Village): ")
bedrooms = int(input("Enter number of Bedrooms: "))

# ✅ Wrap each scalar value in a list
input_df = pd.DataFrame({
    "Area": [area],
    "Location": [location],
    "Bedrooms": [bedrooms]
})

# Encode user input the same way as training data
input_encoded = pd.get_dummies(input_df)
input_encoded = input_encoded.reindex(columns=X.columns, fill_value=0)


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
# Sample data with numeric and categorical features
data = {
    "Area": [1200, 1500, 800, 1100, 1600, 1000, 1400, 900],
    "Location": ["Suburb", "City", "Village", "Suburb", "City", "Village", "City", "Suburb"],
    "Bedrooms": [3, 4, 2, 3, 4, 2, 3, 2],
    "Price": [250000, 400000, 150000, 300000, 420000, 180000, 390000, 210000]
}

df = pd.DataFrame(data)

print("Columns:", list(df.columns))
target = "Price"

X = df.drop(columns=[target])
y = df[target]

# Encode categorical variables
X = pd.get_dummies(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
model = SVR()
model.fit(X_train, y_train)

print("\nEnter values for prediction:")

# Take user inputs with appropriate types
area = float(input("Area (numeric): "))
location = input("Location (City/Suburb/Village): ")
bedrooms = int(input("Bedrooms (numeric): "))

# Create input dataframe from user inputs
input_df = pd.DataFrame({
    "Area": [area],
    "Bedrooms": [bedrooms],
    "Location": [location]
})

# Encode user input
input_encoded = pd.get_dummies(input_df)

# Align input features with training data features
input_encoded = input_encoded.reindex(columns=X.columns, fill_value=0)

# Predict price
prediction = model.predict(input_encoded)
print("\nPredicted Price:", prediction[0])

# Evaluate model on test set
y_pred = model.predict(X_test)
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))







import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Sample data - replace or extend this as needed
data = {
    "Age": ["Young", "Middle", "Old", "Middle", "Young", "Old", "Young", "Middle"],
    "Salary": ["Low", "High", "High", "Medium", "Low", "Medium", "Low", "High"],
    "Purchased": ["No", "Yes", "Yes", "Yes", "No", "Yes", "No", "Yes"]
}
df = pd.DataFrame(data)

print("Columns:", list(df.columns))
target = "Purchased"


X = df.drop(columns=[target])
y_raw = df[target]

# Encode target if categorical
if y_raw.dtype == 'object' or y_raw.dtype.name == 'category':
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
else:
    y = y_raw.values

# Encode categorical features (none in sample, but safe to do)
X = pd.get_dummies(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train SVM classifier
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# Take input for prediction
print("\nEnter values for prediction:")
age = input()
salary=input()
input_df = pd.DataFrame({
    'Age':[age],
    'Salary':[salary]
})

# Encode input and align columns
input_encoded = pd.get_dummies(input_df)
input_encoded = input_encoded.reindex(columns=X.columns, fill_value=0)

# Predict
prediction = model.predict(input_encoded)

# Decode prediction if label encoded
if y_raw.dtype == 'object' or y_raw.dtype.name == 'category':
    pred_class = le.inverse_transform(prediction)
    print("\nPrediction:", pred_class[0])
else:
    print("\nPrediction:", prediction[0])

# Model accuracy
print("Accuracy on test set:", accuracy_score(y_test, model.predict(X_test)))



# model = SVC(kernel='linear')
#              or
# model = SVC(kernel='rbf')
#              or
# model = SVC(kernel='poly')
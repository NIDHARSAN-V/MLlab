import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Sample data with Age and Salary as categorical
data = {
    "Age": ["Low", "Medium", "High", "Medium", "Low", "High", "Low", "Medium"],
    "Salary": ["Low", "Medium", "High", "High", "Low", "High", "Low", "Medium"],
    "Purchased": ["No", "Yes", "Yes", "Yes", "No", "Yes", "No", "Yes"]
}

df = pd.DataFrame(data)

target = "Purchased"

X_raw = df.drop(columns=[target])
y = df[target]

# One-hot encode all categorical columns
X = pd.get_dummies(X_raw)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
model = GaussianNB()
model.fit(X_train, y_train)

# User input
print("\nEnter values for prediction:")
age = input("Enter Age (Low/Medium/High): ")
salary = input("Enter Salary (Low/Medium/High): ")

input_df = pd.DataFrame({
    "Age": [age],
    "Salary": [salary]
})

input_encoded = pd.get_dummies(input_df)
input_encoded = input_encoded.reindex(columns=X.columns, fill_value=0)

prediction = model.predict(input_encoded)
print("\nPredicted category:", prediction[0])

print("Accuracy on test set:", accuracy_score(y_test, model.predict(X_test)))

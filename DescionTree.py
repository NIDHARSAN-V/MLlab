import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt  # Missing import

# Step 1: Define the dataset
data = {
    'Age': [22, 25, 47, 52, 46, 56],
    'Income': ['Low', 'High', 'Medium', 'Medium', 'Low', 'High'],
    'Student': ['No', 'No', 'Yes', 'Yes', 'No', 'Yes'],
    'Buys_Computer': ['No', 'No', 'Yes', 'Yes', 'No', 'Yes']
}

df = pd.DataFrame(data)

# Step 2: Prepare features and target
X = pd.get_dummies(df.drop(columns='Buys_Computer'))
y = df['Buys_Computer']

# Step 3: Train model with fixed random state for reproducibility
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Step 4: Get user input
print("Enter values for: Age, Income (Low/Medium/High), Student (Yes/No)")
age = input("Age: ")
income = input("Income (Low/Medium/High): ")
student = input("Student (Yes/No): ")

# Step 5: Create a DataFrame for user input
user_data = pd.DataFrame([{
    'Age': float(age),
    'Income': income,
    'Student': student
}])

# Step 6: Encode and align with training data
user_data_encoded = pd.get_dummies(user_data)
user_data_encoded = user_data_encoded.reindex(columns=X.columns, fill_value=0)

# Step 7: Make prediction
prediction = model.predict(user_data_encoded)
print("Prediction:", prediction[0])

# Step 8: Show accuracy
accuracy = accuracy_score(y_test, model.predict(X_test))
print("Model accuracy:", round(accuracy * 100, 2), "%")

# Plot the decision tree
plt.figure(figsize=(20, 10))
plot_tree(model, feature_names=X.columns, class_names=[str(c) for c in model.classes_], filled=True)
plt.title("Decision Tree Visualization")
plt.show()

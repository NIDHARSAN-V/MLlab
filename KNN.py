import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def main():
    # Sample dataset as a DataFrame directly in code
    data = {
        'Age': [22, 25, 47, 52, 46, 56, 23, 40, 36, 28],
        'Income': ['Low', 'High', 'Medium', 'Medium', 'Low', 'High', 'Low', 'Medium', 'High', 'Low'],
        'Student': ['No', 'No', 'Yes', 'Yes', 'No', 'Yes', 'Yes', 'No', 'No', 'Yes'],
        'Buys_Computer': ['No', 'No', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'No']
    }
    df = pd.DataFrame(data)

    target_col = 'Buys_Computer'
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # One-hot encode categorical features
    X = pd.get_dummies(X)

    # Ask for K
    k = int(input("Enter value of K for K-NN: "))

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train model
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)

    # Collect user input
    print("Enter values for: Age, Income (Low/Medium/High), Student (Yes/No)")
    age = input("Age: ")
    income = input("Income (Low/Medium/High): ")
    student = input("Student (Yes/No): ")

    # Create DataFrame from user input and apply one-hot encoding
    user_data = pd.DataFrame([{
        'Age': float(age),
        'Income': income,
        'Student': student
    }])
    user_df = pd.get_dummies(user_data)

    # Align columns with training data features
    user_df = user_df.reindex(columns=X.columns, fill_value=0)

    # Predict
    prediction = model.predict(user_df)
    print("K-NN Prediction:", prediction[0])

    # Evaluate accuracy
    print("Accuracy on test set:", accuracy_score(y_test, model.predict(X_test)))

if __name__ == "__main__":
    main()



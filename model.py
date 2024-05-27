import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from joblib import dump  # For saving the model

def train_and_save_model(data_path, model_filename):
    try:
        # Load data
        data = pd.read_csv(data_path)
        X = data.drop(columns=['Dyslexia'])  # Features
        y = data['Dyslexia']  # Target variable

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the model
        model = DecisionTreeClassifier()
        model.fit(X_train, y_train)

        # Evaluate the model
        accuracy = accuracy_score(y_test, model.predict(X_test))
        print(f"Model Accuracy: {accuracy}")

        # Save the model
        dump(model, model_filename)
        print(f"Model saved successfully to: {model_filename}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage 
data_path = "Dys_Cleaned1.csv"  # This is now passed to the function
model_filename = "model.pkl"
train_and_save_model(data_path, model_filename)

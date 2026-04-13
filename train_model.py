import pandas as pd
import numpy as np
import json
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

def train_and_save_model():
    print("Loading dataset.csv...")
    df = pd.read_csv('data/dataset.csv')

    # 1. Clean data and find all unique symptoms
    # The columns are usually named 'Disease', 'Symptom_1', 'Symptom_2', etc.
    symptom_cols = [col for col in df.columns if 'Symptom' in col]
    
    unique_symptoms = set()
    for col in symptom_cols:
        # Get unique values, drop NaNs, and remove extra spaces
        symptoms = df[col].dropna().astype(str).str.strip().unique()
        unique_symptoms.update(symptoms)
        
    # Remove any empty strings if they exist
    unique_symptoms = [s for s in list(unique_symptoms) if s]
    unique_symptoms.sort() # Sort alphabetically

    print(f"Found {len(unique_symptoms)} unique symptoms.")

    # 2. Transform the dataset into a binary (0 and 1) format for ML
    print("Converting dataset to binary format... (this might take a few seconds)")
    X = pd.DataFrame(0, index=df.index, columns=unique_symptoms)
    y = df['Disease'].str.strip()

    # Fill in the 1s where the symptom is present
    for index, row in df.iterrows():
        for col in symptom_cols:
            symptom = str(row[col]).strip()
            if symptom != 'nan' and symptom in unique_symptoms:
                X.at[index, symptom] = 1

    # 3. Save the symptom list for Grok later
    os.makedirs('models', exist_ok=True)
    with open('models/symptom_list.json', 'w') as f:
        json.dump(unique_symptoms, f)
    
    # 4. Split data into Training and Testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 5. Train the Random Forest Model
    print("Training Random Forest Model...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    # 6. Evaluate Model
    print("Evaluating Model...")
    predictions = rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    print(f"\n--- Model Evaluation ---")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    
    # 7. Save the trained model
    joblib.dump(rf_model, 'models/disease_rf_model.joblib')
    print("\nModel saved successfully to models/disease_rf_model.joblib!")

if __name__ == "__main__":
    train_and_save_model()
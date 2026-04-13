import joblib
import json
import numpy as np
import os

class MLPredictor:
    def __init__(self):
        # Load the saved model and symptom list we created in Step 2
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        
        model_path = os.path.join(project_root, 'models', 'disease_rf_model.joblib')
        symptoms_path = os.path.join(project_root, 'models', 'symptom_list.json')
        
        print("Loading ML Model and Symptom List...")
        self.rf_model = joblib.load(model_path)
        
        with open(symptoms_path, 'r') as f:
            self.symptom_list = json.load(f)

    def predict_top_diseases(self, user_symptoms, top_n=3):
        """
        Takes a list of symptoms (e.g., ['itching', 'skin_rash']) 
        and returns the top N predicted diseases with their probabilities.
        """
        # 1. Create an array of 0s (same length as our total valid symptoms)
        symptom_array = np.zeros(len(self.symptom_list))
        
        # 2. For every symptom the user has, flip the 0 to a 1
        for symptom in user_symptoms:
            if symptom in self.symptom_list:
                index = self.symptom_list.index(symptom)
                symptom_array[index] = 1
                
        # 3. If no valid symptoms were found, return empty
        if sum(symptom_array) == 0:
            return []

        # 4. Get probabilities from the Random Forest model
        # predict_proba returns a list of probabilities for every single disease
        probabilities = self.rf_model.predict_proba([symptom_array])[0]
        
        # 5. Combine disease names with their probabilities
        disease_probs = list(zip(self.rf_model.classes_, probabilities))
        
        # 6. Sort them from highest probability to lowest
        disease_probs.sort(key=lambda x: x[1], reverse=True)
        
        # 7. Format the output nicely to send to Grok
        # Example: [("Fungal infection", "95.0%"), ("Allergy", "3.0%")]
        top_predictions = []
        for disease, prob in disease_probs[:top_n]:
            if prob > 0: # Only include it if probability is greater than 0
                formatted_prob = f"{prob * 100:.1f}%"
                top_predictions.append((disease, formatted_prob))
                
        return top_predictions

# Test it out if you run this file directly
if __name__ == "__main__":
    predictor = MLPredictor()
    test_symptoms = ["itching", "skin_rash"]
    results = predictor.predict_top_diseases(test_symptoms)
    print(f"Testing symptoms {test_symptoms}:")
    print(f"Predictions: {results}")
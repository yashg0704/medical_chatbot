from src.llm_clients import extract_symptoms, generate_final_response, generate_followup_question
import pandas as pd

def get_disease_details(disease_name, desc_df, prec_df):
    """Finds the real medical description and precautions from the datasets"""
    details = ""
    clean_disease_name = disease_name.strip()
    
    desc_row = desc_df[desc_df['Disease'].str.strip() == clean_disease_name]
    if not desc_row.empty:
        details += f"Description: {desc_row.iloc[0]['Description']}\n"
        
    prec_row = prec_df[prec_df['Disease'].str.strip() == clean_disease_name]
    if not prec_row.empty:
        precautions = [str(p).strip() for p in prec_row.iloc[0, 1:].values if pd.notna(p)]
        details += f"Precautions: {', '.join(precautions)}\n"
        
    return details

def run_diagnostic_pipeline(user_input, accumulated_symptoms, predictor, valid_symptoms, desc_df, prec_df):
    """
    Runs the pipeline with a feedback loop AND a breakout condition.
    """
    # 1. Extract NEW symptoms from what the user just typed
    new_symptoms = extract_symptoms(user_input, valid_symptoms)
    
    # --- THE FIX: BREAKOUT CONDITION ---
    # If the user typed something, but the LLM found ZERO new symptoms, 
    # AND we already have symptoms in memory, it means the user is out of symptoms!
    user_is_done = (len(new_symptoms) == 0 and len(accumulated_symptoms) > 0)
    
    # 2. Add new symptoms to our running memory list
    for symptom in new_symptoms:
        if symptom not in accumulated_symptoms:
            accumulated_symptoms.append(symptom)
            
    # If we still have zero symptoms overall
    if not accumulated_symptoms:
        return accumulated_symptoms, [], "I couldn't identify any specific medical symptoms. Could you describe how you are feeling in different words?", False
        
    # 3. Predict using ALL symptoms gathered so far
    predictions = predictor.predict_top_diseases(accumulated_symptoms)
    
    # 4. Check the Confidence Score
    top_prob = 0.0
    if predictions:
        top_prob = float(predictions[0][1].replace('%', ''))
        
    # 5. DECISION ROUTER
    # Ask follow up ONLY IF: Confidence is low, AND the user isn't done, AND we haven't asked too many times
    if top_prob < 60.0 and not user_is_done and len(accumulated_symptoms) < 5:
        follow_up_text = generate_followup_question(accumulated_symptoms, predictions)
        return accumulated_symptoms, predictions, follow_up_text, False 
        
    # 6. FINAL DIAGNOSIS (Triggers if confidence > 60%, OR if the user said "nothing else")
    top_disease = predictions[0][0] if predictions else ""
    disease_context = get_disease_details(top_disease, desc_df, prec_df) if top_disease else ""
    
    final_response = generate_final_response(
        user_text=user_input,
        extracted_symptoms=accumulated_symptoms,
        ml_predictions=predictions,
        disease_details=disease_context
    )
    
    return accumulated_symptoms, predictions, final_response, True 
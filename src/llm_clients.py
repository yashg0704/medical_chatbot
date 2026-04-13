import os
import json
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Initialize the Grok (xAI) client
# Note: Grok uses the xAI API, which is compatible with the OpenAI Python SDK
# If using OpenAI instead, just remove base_url and use OPENAI_API_KEY
client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

# Use grok-beta or grok-2 depending on your API access (or gpt-3.5-turbo if using OpenAI)
MODEL_NAME = "llama-3.3-70b-versatile"

def extract_symptoms(user_text, valid_symptoms_list):
    """
    Uses Grok to extract symptoms from user text and map them strictly 
    to our ML model's valid symptoms.
    """
    system_prompt = f"""
    You are an expert medical natural language processing agent.
    Your task is to extract symptoms from the user's text and map them strictly to the provided valid symptom list.
    
    VALID SYMPTOMS: {valid_symptoms_list}
    
    RULES:
    1. ONLY use symptoms from the VALID SYMPTOMS list.
    2. If a user describes a symptom using different words, map it to the closest valid symptom.
    3. Output ONLY a valid JSON array of strings. No conversational text. No markdown blocks.
    Example output: ["headache", "high_fever", "vomiting"]
    """

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text}
            ],
            temperature=0.1 # Low temperature makes it more strict/deterministic
        )
        
        # Parse the JSON response into a Python list
        extracted_text = response.choices[0].message.content.strip()
        
        # Clean up in case the LLM adds markdown ```json [...] ```
        if extracted_text.startswith("```"):
            extracted_text = extracted_text.split("\n", 1)[1].rsplit("\n", 1)[0]
            
        symptoms_list = json.loads(extracted_text)
        return symptoms_list

    except Exception as e:
        print(f"Error extracting symptoms: {e}")
        return []

def generate_final_response(user_text, extracted_symptoms, ml_predictions, disease_details=""):
    """
    Acts as the final Medical LLM Validator.
    Takes the ML model predictions and provides a safe, detailed medical response.
    """
    system_prompt = """
    You are a professional Medical AI Assistant. 
    You are validating the predictions of an internal Machine Learning model.
    Always be empathetic, professional, and include a strong medical disclaimer that you are an AI, not a doctor.
    """

    user_prompt = f"""
    User Query: "{user_text}"
    Extracted Symptoms: {extracted_symptoms}
    ML Model Top Predictions (with confidence scores): {ml_predictions}
    Additional Medical Context (Description/Precautions): {disease_details}
    
    TASK:
    1. Evaluate if the ML model's top prediction makes clinical sense based on the symptoms.
    2. If it makes sense, explain the disease and provide the precautions.
    3. If the confidence is very low or the prediction seems wrong, state your uncertainty and suggest what it might actually be.
    4. Provide the final response to the patient.
    """

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.5
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Sorry, I encountered an error generating the medical response: {e}"

def generate_followup_question(accumulated_symptoms, ml_predictions):
    """
    Generates a follow-up question when the ML model confidence is too low.
    """
    system_prompt = """
    You are an empathetic triage nurse. 
    The patient has reported some symptoms, but the internal diagnostic model is not confident enough to make a safe prediction yet.
    Your job is to ask a natural, conversational follow-up question to gather more symptoms.
    
    RULES:
    1. DO NOT diagnose the patient or mention specific diseases yet.
    2. Acknowledge their current symptoms.
    3. Ask if they are experiencing other related symptoms (you can use the ML predictions to guess what to ask, e.g., if the ML suspects GERD, ask about chest burning or nausea).
    4. Keep it brief and friendly.
    """

    user_prompt = f"""
    Patient's current identified symptoms: {accumulated_symptoms}
    Internal ML Suspicions (Do NOT tell the patient these names yet): {ml_predictions}
    
    Please ask a follow-up question to find out what else they might be feeling.
    """

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.6
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Could you tell me a bit more about how you are feeling? Any other symptoms?"
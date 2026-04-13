import streamlit as st
import pandas as pd
import json
from src.ml_predictor import MLPredictor
from src.pipeline import run_diagnostic_pipeline

# --- 1. Load Data ---
@st.cache_data
def load_medical_context():
    desc_df = pd.read_csv('data/symptom_Description.csv', skipinitialspace=True)
    prec_df = pd.read_csv('data/symptom_precaution.csv', skipinitialspace=True)
    desc_df.columns = desc_df.columns.str.strip()
    prec_df.columns = prec_df.columns.str.strip()
    return desc_df, prec_df

# --- 2. Initialize App ---
st.set_page_config(page_title="Medical Diagnostic AI", page_icon="🏥")
st.title("🏥 Medical Diagnostic Assistant")
st.write("Describe your symptoms naturally. I will ask follow-up questions until I have enough information!")

if 'predictor' not in st.session_state:
    st.session_state.predictor = MLPredictor()
    st.session_state.desc_df, st.session_state.prec_df = load_medical_context()
    with open('models/symptom_list.json', 'r') as f:
        st.session_state.valid_symptoms = json.load(f)

# === NEW: MEMORY STATE ===
if "accumulated_symptoms" not in st.session_state:
    st.session_state.accumulated_symptoms = []

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! Please describe your symptoms."}]

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- 3. UI Chat Logic ---
if user_input := st.chat_input("E.g., I have a bad headache..."):
    
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            
            # Run the pipeline with our memory state
            updated_symptoms, predictions, response_text, is_final = run_diagnostic_pipeline(
                user_input=user_input,
                accumulated_symptoms=st.session_state.accumulated_symptoms,
                predictor=st.session_state.predictor,
                valid_symptoms=st.session_state.valid_symptoms,
                desc_df=st.session_state.desc_df,
                prec_df=st.session_state.prec_df
            )
            
            # Save the updated symptoms back to memory
            st.session_state.accumulated_symptoms = updated_symptoms
            
            # Show diagnostic data for debugging/presentation
            if updated_symptoms:
                st.info(f"🧠 **Current Known Symptoms:** {', '.join(updated_symptoms)}")
                st.warning(f"📊 **Current ML Confidence:** {predictions}")
                
                if not is_final:
                    st.info("🔄 *Confidence is below 60%. Asking follow-up question...*")
                else:
                    st.success("✅ *Confidence reached! Generating final diagnosis...*")
                    # Clear memory so they can start a new diagnosis next time
                    st.session_state.accumulated_symptoms = []
            
            # Display text
            st.markdown(response_text)
            st.session_state.messages.append({"role": "assistant", "content": response_text})
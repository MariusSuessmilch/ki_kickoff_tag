import streamlit as st
import pandas as pd
import os
import base64
from io import BytesIO
import requests
from datetime import datetime
from utils import generate_image, evaluate_image, load_data, save_data

# Set page configurations
st.set_page_config(
    page_title="Stadt der Zukunft - KI-Wettbewerb",
    page_icon="🏙️",
    layout="wide"
)

# Initialize session state variables if they don't exist
if 'generated_image' not in st.session_state:
    st.session_state.generated_image = None
if 'image_url' not in st.session_state:
    st.session_state.image_url = None
if 'evaluation_results' not in st.session_state:
    st.session_state.evaluation_results = None
if 'submitted' not in st.session_state:
    st.session_state.submitted = False

# Create data directory if it doesn't exist
os.makedirs('data', exist_ok=True)

# Load existing data
data = load_data()

# Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Gehe zu:", ["Wettbewerb", "Siegerehrung", "Gallerie"])

if page == "Wettbewerb":
    st.title("🏙️ Die Stadt der Zukunft - KI-Kunstwettbewerb")
    st.write("""
    Willkommen zum KI-Kunstwettbewerb! Erstelle dein eigenes Bild zum Thema "Die Stadt der Zukunft" 
    und lass es von unserem KI-Jury bewerten!
    """)
    
    # Input form
    with st.form(key="submission_form"):
        name = st.text_input("Dein Name:", max_chars=50)
        prompt_base = "Erstelle ein detailliertes, kreatives und futuristisches Bild einer Stadt der Zukunft."
        custom_prompt = st.text_area(
            "Beschreibe deine Stadt der Zukunft:", 
            value="Eine saubere und grüne Stadt mit fliegenden Autos und Wolkenkratzern.",
            help="Beschreibe, wie du dir die Stadt der Zukunft vorstellst. Sei kreativ!"
        )
        
        submit_button = st.form_submit_button(label="Bild erstellen")
        
        if submit_button and name:
            with st.spinner("Dein Bild wird erstellt..."):
                full_prompt = f"{prompt_base} {custom_prompt}"
                try:
                    # Generate image using DALL-E through OpenAI API
                    response = generate_image(full_prompt)
                    st.session_state.image_url = response['url']
                    
                    # Download and save the image in memory
                    image_response = requests.get(st.session_state.image_url)
                    img_data = BytesIO(image_response.content)
                    st.session_state.generated_image = img_data
                    
                    # Display the generated image
                    st.image(st.session_state.image_url, caption="Deine Stadt der Zukunft", use_column_width=True)
                except Exception as e:
                    st.error(f"Fehler bei der Bilderstellung: {str(e)}")
        elif submit_button and not name:
            st.warning("Bitte gib deinen Namen ein!")

    # Display the generated image outside the form if it exists
    if st.session_state.generated_image and not st.session_state.submitted:
        st.image(st.session_state.image_url, caption="Deine Stadt der Zukunft", use_column_width=True)
        
        # Evaluate button
        if st.button("Jetzt bewerten lassen"):
            with st.spinner("Dein Bild wird bewertet..."):
                try:
                    # Convert image to base64 for GPT-4 Vision API
                    img_data = st.session_state.generated_image
                    img_data.seek(0)
                    img_base64 = base64.b64encode(img_data.read()).decode('utf-8')
                    
                    # Evaluate image with GPT-4 Vision
                    evaluation = evaluate_image(img_base64, custom_prompt)
                    st.session_state.evaluation_results = evaluation
                    
                    # Display evaluation results
                    st.subheader("Bewertung deines Bildes:")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Kreativität", f"{evaluation['creativity']}/10")
                    with col2:
                        st.metric("Themenpassung", f"{evaluation['theme_relevance']}/10")
                    with col3:
                        st.metric("Zukunftsvision", f"{evaluation['vision_quality']}/10")
                    
                    st.subheader(f"Gesamtwertung: {evaluation['total_score']}/30")
                    st.write(f"**Feedback**: {evaluation['feedback']}")
                    
                    # Save submission button
                    if st.button("Teilnahme einreichen"):
                        # Save the submission to our data
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        
                        # Store image as base64 in the data
                        img_data.seek(0)
                        image_base64 = base64.b64encode(img_data.read()).decode('utf-8')
                        
                        new_submission = {
                            'timestamp': timestamp,
                            'name': name,
                            'prompt': custom_prompt,
                            'image': image_base64,
                            'creativity': evaluation['creativity'],
                            'theme_relevance': evaluation['theme_relevance'],
                            'vision_quality': evaluation['vision_quality'],
                            'total_score': evaluation['total_score'],
                            'feedback': evaluation['feedback']
                        }
                        
                        data = data.append(new_submission, ignore_index=True)
                        save_data(data)
                        
                        st.success("Deine Teilnahme wurde erfolgreich eingereicht!")
                        st.session_state.submitted = True
                        
                except Exception as e:
                    st.error(f"Fehler bei der Bewertung: {str(e)}")
    
    # After submission, show a thank you message
    if st.session_state.submitted:
        st.success("Vielen Dank für deine Teilnahme! Du kannst dir die Rangliste unter 'Siegerehrung' ansehen.")
        if st.button("Neues Bild erstellen"):
            st.session_state.generated_image = None
            st.session_state.image_url = None
            st.session_state.evaluation_results = None
            st.session_state.submitted = False
            st.rerun()

elif page == "Siegerehrung":
    st.title("🏆 Siegerehrung - Top 10")
    
    if data.empty:
        st.info("Es wurden noch keine Bilder eingereicht. Sei der Erste!")
    else:
        # Sort data by total score (descending)
        top_entries = data.sort_values(by='total_score', ascending=False).head(10)
        
        # Display top 10 entries
        for index, entry in enumerate(top_entries.itertuples(), 1):
            with st.container():
                st.subheader(f"{index}. Platz: {entry.name} - {entry.total_score} Punkte")
                
                col1, col2 = st.columns([1, 1])
                with col1:
                    # Convert base64 to image
                    image_data = base64.b64decode(entry.image)
                    image = BytesIO(image_data)
                    st.image(image, caption=f"Stadt der Zukunft von {entry.name}", use_column_width=True)
                
                with col2:
                    st.write(f"**Prompt**: {entry.prompt}")
                    st.write(f"**Kreativität**: {entry.creativity}/10")
                    st.write(f"**Themenpassung**: {entry.theme_relevance}/10")
                    st.write(f"**Zukunftsvision**: {entry.vision_quality}/10")
                    st.write(f"**Feedback**: {entry.feedback}")
                
                st.markdown("---")

elif page == "Gallerie":
    st.title("🖼️ Bildergallerie")
    
    if data.empty:
        st.info("Es wurden noch keine Bilder eingereicht. Sei der Erste!")
    else:
        # Sort data by timestamp (newest first)
        sorted_data = data.sort_values(by='timestamp', ascending=False)
        
        # Create a grid of images (3 columns)
        cols = st.columns(3)
        
        for idx, entry in enumerate(sorted_data.itertuples()):
            col_idx = idx % 3
            with cols[col_idx]:
                # Convert base64 to image
                image_data = base64.b64decode(entry.image)
                image = BytesIO(image_data)
                st.image(image, caption=f"{entry.name} - {entry.total_score} Punkte", use_column_width=True)
                
                with st.expander("Details"):
                    st.write(f"**Prompt**: {entry.prompt}")
                    st.write(f"**Kreativität**: {entry.creativity}/10")
                    st.write(f"**Themenpassung**: {entry.theme_relevance}/10")
                    st.write(f"**Zukunftsvision**: {entry.vision_quality}/10")
                    st.write(f"**Feedback**: {entry.feedback}")

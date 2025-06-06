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
    page_icon=":cityscape:",
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
if 'form_name' not in st.session_state:
    st.session_state.form_name = ""
if 'form_prompt' not in st.session_state:
    st.session_state.form_prompt = "Eine saubere und grüne Stadt mit fliegenden Autos und Wolkenkratzern."

# Create data directory if it doesn't exist
os.makedirs('data', exist_ok=True)

# Load existing data
if 'data' not in st.session_state:
    st.session_state.data = load_data()
data = st.session_state.data

# Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Gehe zu:", ["Wettbewerb", "Siegerehrung", "Gallerie"])

# Wenn die Seite gewechselt hat, setze das Reset-Flag
if 'current_page' not in st.session_state:
    st.session_state.current_page = page
if st.session_state.current_page != page:
    st.session_state.reset_form = True
    st.session_state.current_page = page

# Formular-Reset-Logik: Muss nach Navigation stehen!
if st.session_state.get('reset_form', False):
    # Nur zurücksetzen, wenn wir auf der Wettbewerbs-Seite sind
    if page == "Wettbewerb":
        st.session_state.generated_image = None
        st.session_state.image_url = None
        st.session_state.evaluation_results = None
        st.session_state.submitted = False
        st.session_state.form_name = ""
        st.session_state.form_prompt = "Eine saubere und grüne Stadt mit fliegenden Autos und Wolkenkratzern."
    st.session_state.reset_form = False

if page == "Wettbewerb":
    st.title("🏙️ Die Stadt der Zukunft - KI-Kunstwettbewerb")
    st.write("""
    Willkommen zum KI-Kunstwettbewerb! Erstelle dein eigenes Bild zum Thema "Die Stadt der Zukunft" 
    und lass es von unserem KI-Jury bewerten!
    """)
    
    # Input form
    with st.form(key="submission_form"):
        # Formularfelder mit eigenem State verknüpfen
        if 'form_name' not in st.session_state:
            st.session_state.form_name = ""
        if 'form_prompt' not in st.session_state:
            st.session_state.form_prompt = "Eine saubere und grüne Stadt mit fliegenden Autos und Wolkenkratzern."
        name = st.text_input("Dein Name:", max_chars=70, key="form_name")
        custom_prompt = st.text_area(
            "Beschreibe deine Stadt der Zukunft:", 
            key="form_prompt",
            help="Beschreibe, wie du dir die Stadt der Zukunft vorstellst. Sei kreativ!"
        )
        
        submit_button = st.form_submit_button(label="Bild erstellen")
        
        if submit_button and name:
            with st.spinner("Dein Bild wird erstellt..."):
                full_prompt = custom_prompt
                try:
                    # Generate image using DALL-E through OpenAI API
                    response = generate_image(full_prompt)
                    st.session_state.image_url = response['url']
                    
                    # Download and save the image in memory
                    image_response = requests.get(st.session_state.image_url)
                    img_data = BytesIO(image_response.content)
                    st.session_state.generated_image = img_data
                    
                    # Erfolg-Nachricht, aber kein Bild anzeigen (wird außerhalb des Formulars angezeigt)
                    st.success("Bild erfolgreich erstellt!")

                except Exception as e:
                    st.error(f"Fehler bei der Bilderstellung: {str(e)}")
        elif submit_button and not name:
            st.warning("Bitte gib deinen Namen ein!")

    # Display the generated image outside the form if it exists
    if st.session_state.generated_image and not st.session_state.submitted:
        # Debug: Prüfe, ob Bild im Session-State ist und gib Größe aus

        st.image(st.session_state.image_url, caption="Deine Stadt der Zukunft", use_container_width=True)
        
        # Submit button - directly submits the image without showing evaluation first
        if st.button("Teilnahme einreichen"):
            with st.spinner("Dein Bild wird bewertet und eingereicht..."):
                try:
                    # Convert image to base64 for GPT-4 Vision API
                    img_data = st.session_state.generated_image
                    img_data.seek(0)
                    img_base64 = base64.b64encode(img_data.read()).decode('utf-8')
                    
                    # Evaluate image with GPT-4 Vision (but don't display results)
                    evaluation = evaluate_image(img_base64, st.session_state.form_prompt)
                    st.session_state.evaluation_results = evaluation
                    
                    # Save the submission to our data
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    
                    # Store image as base64 in the data
                    img_data.seek(0)
                    image_base64 = base64.b64encode(img_data.read()).decode('utf-8')
                    # Debug: base64 Länge und Ausschnitt anzeigen

                    
                    new_submission = {
                        'timestamp': timestamp,
                        'name': st.session_state.form_name,
                        'prompt': st.session_state.form_prompt,
                        'image': image_base64,
                        'creativity': evaluation['creativity'],
                        'theme_relevance': evaluation['theme_relevance'],
                        'vision_quality': evaluation['vision_quality'],
                        'total_score': evaluation['total_score'],
                        'feedback': evaluation['feedback']
                    }
                    
                    # Create a DataFrame from the new submission and concatenate with existing data
                    new_row = pd.DataFrame([new_submission])
                    st.session_state.data = pd.concat([st.session_state.data, new_row], ignore_index=True)
                    save_data(st.session_state.data)
                    
                    st.success("Deine Teilnahme wurde erfolgreich eingereicht!")
                    st.session_state.submitted = True
                    
                except Exception as e:
                    st.error(f"Fehler bei der Bewertung: {str(e)}")
    
    # After submission, show reset button
    if st.session_state.submitted:
        if st.button("Neues Bild erstellen"):
            # Setze nur das Reset-Flag und rerun, der eigentliche Reset erfolgt am Skript-Anfang
            st.session_state.reset_form = True
            st.experimental_set_query_params()  # Setze URL-State zurück
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
                    st.image(image, caption=f"Stadt der Zukunft von {entry.name}", use_container_width=True)
                
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
        # Platz für zukünftige Features
            
        # Sort data by timestamp (newest first)
        sorted_data = data.sort_values(by='timestamp', ascending=False)
        
        # Create a grid of images (3 columns)
        cols = st.columns(3)
        
        for idx, entry in enumerate(sorted_data.itertuples()):
            col_idx = idx % 3
            with cols[col_idx]:
                # Versuche, das Bild zu decodieren und anzuzeigen
                try:
                    image_data = base64.b64decode(entry.image)
                    image = BytesIO(image_data)
                    st.image(image, caption=f"{entry.name} - {entry.total_score} Punkte", use_container_width=True)
                except Exception as e:
                    st.error("Fehler beim Anzeigen des Bildes")
                
                with st.expander("Details"):
                    st.write(f"**Prompt**: {entry.prompt}")
                    st.write(f"**Kreativität**: {entry.creativity}/10")
                    st.write(f"**Themenpassung**: {entry.theme_relevance}/10")
                    st.write(f"**Zukunftsvision**: {entry.vision_quality}/10")
                    st.write(f"**Feedback**: {entry.feedback}")

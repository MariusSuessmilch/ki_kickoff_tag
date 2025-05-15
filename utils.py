import pandas as pd
import os
import json
from openai import OpenAI

# Initialize OpenAI client
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

# Datei-Pfad für die CSV-Datei
DATA_FILE = 'data/submissions.csv'

def load_data():
    """Lade die Einreichungen aus der CSV-Datei."""
    try:
        if os.path.exists(DATA_FILE):
            return pd.read_csv(DATA_FILE)
        return pd.DataFrame(columns=[
            'timestamp', 'name', 'prompt', 'image',
            'creativity', 'theme_relevance', 'vision_quality',
            'total_score', 'feedback'
        ])
    except Exception as e:
        print(f"Fehler beim Laden der Daten: {e}")
        return pd.DataFrame()

def save_data(data):
    """Speichere die Einreichungen in der CSV-Datei."""
    try:
        os.makedirs(os.path.dirname(DATA_FILE), exist_ok=True)
        data.to_csv(DATA_FILE, index=False)
    except Exception as e:
        print(f"Fehler beim Speichern der Daten: {e}")

def generate_image(prompt):
    """Generate an image using DALL-E 3 based on the provided prompt."""
    try:
        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            n=1,
            size="1024x1024",
        )
        return {"url": response.data[0].url}
    except Exception as e:
        raise Exception(f"Fehler bei der Bilderstellung: {str(e)}")

def evaluate_image(base64_image, prompt):
    """Evaluate the image using GPT-4 Vision based on creativity, theme relevance, and vision quality."""
    try:
        system_prompt = """Du bist ein freundlicher Kunst-Juror für den Wettbewerb "Stadt der Zukunft". 
        Bewerte das eingereichte Bild kritisch nach diesen drei Kriterien:
        1. Kreativität (1-10): Wie originell und kreativ setzt das Bild das Thema "Stadt der Zukunft" um?
        2. Themenpassung (1-10): Wie gut passt das Bild zum Thema "Stadt der Zukunft"?
        3. Zukunftsvision (1-10): Wie überzeugend und durchdacht ist die Vision für die Stadt der Zukunft?
        
        Gib deine Bewertung als JSON zurück und füge ein kurzes, Feedback aus, dass die Werte für die 3 Kriterien erklärt.
        Sei streng in den Bewertunsgzahlen, aber gib freundliches Feedback, dass für Erwachsene und Kinder geeignet ist.
        
        Antworte in diesem JSON-Format:
        {
            "creativity": Zahl (1-10),
            "theme_relevance": Zahl (1-10),
            "vision_quality": Zahl (1-10),
            "total_score": Zahl (Summe der obigen),
            "feedback": "kurzes, erklärendes Feedback"
        }
        """
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {"type": "text", "text": f"Evaluiere das folgende Bild auf Basis der 3 Kriterien:."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]}
            ],
            response_format={"type": "json_object"},
            max_tokens=1000
        )
        
        result = json.loads(response.choices[0].message.content)
        
        # Ensure all scores are within range 1-10
        result["creativity"] = max(1, min(10, result["creativity"]))
        result["theme_relevance"] = max(1, min(10, result["theme_relevance"]))
        result["vision_quality"] = max(1, min(10, result["vision_quality"]))
        
        # Calculate total score (sum of the three categories)
        result["total_score"] = result["creativity"] + result["theme_relevance"] + result["vision_quality"]
        
        return result
    except Exception as e:
        raise Exception(f"Fehler bei der Bildbewertung: {str(e)}")

def load_data():
    """Load submissions data from CSV file."""
    csv_path = 'data/submissions.csv'
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    else:
        # Create empty dataframe with required columns
        columns = ['timestamp', 'name', 'prompt', 'image', 'creativity', 
                  'theme_relevance', 'vision_quality', 'total_score', 'feedback']
        return pd.DataFrame(columns=columns)

def save_data(data):
    """Save submissions data to CSV file."""
    csv_path = 'data/submissions.csv'
    data.to_csv(csv_path, index=False)

import pandas as pd
import os
import json
from openai import OpenAI

# The newest OpenAI model is "gpt-4o" which was released May 13, 2024.
# Do not change this unless explicitly requested by the user

# Initialize OpenAI client
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

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
        system_prompt = """You are an expert art judge for a children's competition about "City of the Future". 
        You must evaluate the submitted image based on three criteria:
        1. Creativity (1-10): How original and imaginative is the image?
        2. Theme Relevance (1-10): How well does the image match the theme "City of the Future"?
        3. Vision Quality (1-10): How compelling and well-thought-out is the vision for the city of the future?
        
        Provide a JSON response with these scores and a short, encouraging feedback for the child. 
        Be fair but generous, as these are children's submissions.
        
        Respond with JSON in this format:
        {
            "creativity": number (1-10),
            "theme_relevance": number (1-10),
            "vision_quality": number (1-10),
            "total_score": number (sum of above),
            "feedback": "brief, encouraging feedback"
        }
        """
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {"type": "text", "text": f"Evaluate this image based on the prompt: '{prompt}'. Remember to be encouraging as this is a child's submission."},
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

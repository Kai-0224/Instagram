import json
import os
import datetime
from googletrans import Translator
from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO
import base64

# Initialize the Google GenAI client
client = genai.Client(api_key=os.environ.get("Genai_API"))
today = datetime.date.today()

# Define file paths
input_file = f"generated_instagram_caption_{today}.txt"
output_file = f"translated_caption_{today}.txt"

# Read the .txt file
try:
    with open(input_file, 'r', encoding='utf-8') as f:
        en_content = f.read()
except FileNotFoundError:
    print(f"Error: File {input_file} not found. Please run Caption_RAG.py first.")
    exit(1)
    
# Translation
translator = Translator()
translated = translator.translate(en_content, src='en', dest='zh-TW')

with open(output_file, "w", encoding="utf-8") as file:
    file.write(translated.text)

### Analyze copywriting and generate design requirements
def analyze_post_content(post_content):
    sections = [
        "Theme and Purpose",
        "Composition and Scene Design", 
        "Color and Style",
        "Details and Texture",
        "Atmosphere and Lighting",
        "Call to Action",
        "Emotion and Storytelling"
    ]
    
    analysis_result = {}
    for section in sections:
        input_text = f"Analyze the following post content and provide a brief and concise insight on {section}. Post content: {post_content}."
        
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=input_text
            )
            analysis_result[section] = response.text
            print(f"Analyzed: {section}")
            
        except Exception as e:
            print(f"Error analyzing {section}: {e}")
            analysis_result[section] = f"Error: {e}"
    return analysis_result

def save_analysis_to_json(analysis, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, ensure_ascii=False, indent=4)

if en_content:
    analysis = analyze_post_content(en_content)
    save_analysis_to_json(analysis, f"post_analysis_result_{today}.json")

### Generate images
def generate_images_with_analysis(analysis_result, en_content):
    key_insights = []
    
    for section, data in analysis_result.items():
        if not section.startswith('_') and not str(data).startswith('Error:'):
            summary = str(data)[:100] + "..." if len(str(data)) > 100 else str(data)
            key_insights.append(f"{section}: {summary}")
    
    if not key_insights:
        prompt = f"""
        Create a professional Instagram product promotion image for Tanji Company.
        
        Content: {en_content[:300]}
        
        Requirements:
        - High-quality, Instagram-friendly visuals
        - Space for text editing, and do not put text in it
        - Include limited-time offer label design
        - Match Tanji Company brand identity
        - 1080x1080 square format
        """
    else:
        prompt = f"""
        Create a professional Instagram product promotion image based on the following analysis:

        Content summary: {en_content[:200]}...
        
        Design requirements:
        {chr(10).join(key_insights[:3])}
        
        Technical requirements:
        - High-quality, Instagram-friendly visuals
        - Space for text editing, and do not put text in it
        - Include limited-time offer label design
        - Match Tanji Company brand identity
        - 1080x1080 square format
        """
    
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash-preview-image-generation",
            contents=prompt,
            config=types.GenerateContentConfig(
            response_modalities=['TEXT', 'IMAGE']
                )
        )
        
        return response.candidates[0].content.parts
    except Exception as e:
        print(f"Error generating images: {e}")
        return []

# Generate images
print("Attempting to generate images...")
generated_images = generate_images_with_analysis(analysis, en_content)

if generated_images:
    for i, generated_image in enumerate(generated_images):
        image_filename = f'gemini-native-product-image_{today}_{i}.png'
        if generated_image.text is not None:
            print(generated_image.text)
        elif generated_image.inline_data is not None:
            image = Image.open(BytesIO((generated_image.inline_data.data)))
            image.save(image_filename)
        print(f"Successfully generated and saved image: {image_filename}")
else:
    print("Image generation failed. No image files were saved.")

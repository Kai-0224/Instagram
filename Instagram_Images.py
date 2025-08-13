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
client = genai.Client(api_key=os.getenv("Genai_API"))
today = datetime.date.today()

# Define file paths
input_file = f"generated_instagram_caption_{today}.txt"
output_file = f"translated_caption_{today}.txt"

# Read the .txt file
try:
    with open(input_file, 'r', encoding='utf-8') as f:
        en_content = f.read()
except FileNotFoundError:
    #print(f"File {tran_file} not found. Please run Caption_RAG.py first.")
    exit(1)

# Translation
translator = Translator()
translated = translator.translate(en_content, src='en', dest='zh-TW')

with open(output_file, "w", encoding="utf-8") as file:
    file.write(translated.text)

### Analyze copywriting and generate design requirements
'''
def analyze_post_content_improved(post_content):
    """改進的內容分析函數"""
    sections = {
        "Theme and Purpose": "分析貼文的主題和目的",
        "Composition and Scene Design": "分析構圖和場景設計要求", 
        "Color and Style": "分析色彩和風格偏好",
        "Details and Texture": "分析細節和紋理要求",
        "Atmosphere and Lighting": "分析氛圍和光線效果",
        "Call to Action": "分析呼籲行動的視覺元素",
        "Emotion and Storytelling": "分析情感和故事敘述"
    }
    
    analysis_result = {}
    successful_analyses = 0

    for section_key, section_desc in sections.items():
        input_text = f"""
        分析以下 Instagram 貼文內容，針對「{section_desc}」提供簡潔的洞察和建議。
        
        貼文內容: {post_content}
        
        請提供：
        1. 關鍵要素分析
        2. 視覺設計建議
        3. 具體實作要點
        """
        
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=input_text
            )
            
            analysis_result[section_key] = {
                "description": section_desc,
                "analysis": response.text,
                "status": "success"
            }
            successful_analyses += 1
            #print(f"✅ 完成分析: {section_key}")
            
        except Exception as e:
            #print(f"❌ 分析錯誤 {section_key}: {e}")
            analysis_result[section_key] = {
                "description": section_desc,
                "analysis": f"分析失敗: {e}",
                "status": "error"
            }
    
    # 添加分析摘要
    analysis_result["_summary"] = {
        "total_sections": len(sections),
        "successful_analyses": successful_analyses,
        "success_rate": f"{(successful_analyses/len(sections)*100):.1f}%",
        "timestamp": datetime.datetime.now().isoformat()
    }
    
    return analysis_result
'''
    
def analyze_post_content(post_content):
    # Define sections to ask questions
    sections = [
        "Theme and Purpose",
        "Composition and Scene Design", 
        "Color and Style",
        "Details and Texture",
        "Atmosphere and Lighting",
        "Call to Action",
        "Emotion and Storytelling"
    ]
    
    analysis_result = {}  # Store the response results of each section

    # Use a for loop to ask questions for each section one by one
    for section in sections:
        input_text = f"Analyze the following post content and provide a brief and concise insight on {section}. Post content: {post_content}."
        
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=input_text
            )
            
            # Store the results of each section
            analysis_result[section] = response.text
            #print(f"Analyzed: {section}")
            
        except Exception as e:
            #print(f"Error analyzing {section}: {e}")
            analysis_result[section] = f"Error: {e}"

    return analysis_result

# Function: Save the analysis results as a JSON file
def save_analysis_to_json(analysis, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, ensure_ascii=False, indent=4)

# Analyze the post content and generate responses for each section one by one
if en_content:
    analysis = analyze_post_content(en_content)
    
    # Save analysis results as JSON files
    save_analysis_to_json(analysis, f"post_analysis_result_{today}.json")

### Generate images
def generate_image(prompt):
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash-preview-image-generation",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_modalities=['TEXT', 'IMAGE']
            )
        )
        
        for part in response.candidates[0].content.parts:
            if hasattr(part, 'inline_data') and part.inline_data.mime_type.startswith('image/'):
                image_data = base64.b64decode(part.inline_data.data)
                return image_data
        return None
    except Exception as e:
        #print(f"Error generating image: {e}")
        return None

def generate_image_with_analysis(analysis_result, en_content):
    # Extract the key point of analysis
    key_insights = []
    
    for section, data in analysis_result.items():

        if not section.startswith('_') and not str(data).startswith('Error:'):
            summary = str(data)[:100] + "..." if len(str(data)) > 100 else str(data)
            key_insights.append(f"{section}: {summary}")
    
    if not key_insights:
        image_prompt = f"""
        Create a professional Instagram product promotion image for Tanji Company.
        
        Content: {en_content[:300]}
        
        Requirements:
        - High-quality, Instagram-friendly visuals
        - Space for text editing
        - Include limited-time offer label design
        - Match Tanji Company brand identity
        - 1080x1080 square format
        """
    else:
        # Use the analysis result to build prompt
        image_prompt = f"""
        Create a professional Instagram product promotion image based on the following analysis:

        Content summary: {en_content[:200]}...
        
        Design requirements:
        {chr(10).join(key_insights[:3])}
        
        Technical requirements:
        - High-quality, Instagram-friendly visuals
        - Space for text editing
        - Include limited-time offer label design
        - Match Tanji Company brand identity
        - 1080x1080 square format
        """
    
    return generate_image(image_prompt)

# Generate image
image_bytes = generate_image_with_analysis(analysis, en_content)

if image_bytes:
    image = Image.open(BytesIO(image_bytes))
    image_filename = f'gemini-native-product-image_{today}.png'
    image.save(image_filename)
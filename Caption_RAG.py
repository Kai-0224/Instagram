import datetime
import calendar
import os
import requests
import json
import faiss
import numpy as np
from huggingface_hub import HfApi
from google import genai

### Set the post topic
# List of prompts for the month
prompt_file = "Prompts.txt"
with open(prompt_file, 'r', encoding='utf-8') as f:
    prompts = [prompt.strip() for prompt in f.read().split('\n') if prompt.strip().startswith('Write')]

'''
prompts = [
    "Write an Instagram post for Tanji Company that includes a brand image and greeting, and emphasizes the brand's core concepts, such as high-quality goods and the best prices on the market. The goal is to create a personal connection with the audience and convey the brand's mission.",
    "Write Instagram copy for [Product Name] for Tanji Company that emphasizes the uniqueness of the product and expresses the care that went into selecting it.",
    "Write a personal reflective Instagram copy for Tanji Company, share your feelings as a business owner, use a relaxed and sincere tone, and connect your feelings with your brand values (high-quality products, best prices) to build an emotional connection.",
    "Write an Instagram caption for Tanji Company that describes [product/service name] being used in a real-life scenario. The caption should evoke emotions and make readers imagine themselves using the product in a similar setting.",
    "Write an Instagram caption for Tanji Company that highlights a KOL (Key Opinion Leader) sharing their experience with [product/service]. Encourage the audience to engage in the conversation and try the product themselves.",
    "Write a sponsored Instagram post for Tanji Company in collaboration with [brand/partner]. Share your experience working with them, highlighting the benefits of the partnership and the quality of the product/service.",
    "Write an Instagram caption for Tanji Company showing your daily work routine and how it reflects the brand's core values. The caption should show the real side of running a business and connect it to the brand's mission.",
    "Write an Instagram caption for Tanji Company thanking real customers for their support, sharing their testimonials or real-life photos of using your products. The tone should be sincere and emotional, showcasing customer satisfaction.",
    "Write a promotional Instagram caption for [product/service] for Tanji Company. The tone should be enthusiastic and persuasive, encouraging readers to take action. Highlight the unique selling points and provide a clear call-to-action (e.g., limited-time discount).",
    "Write an Instagram caption for Tanji Company related to a current event or holiday, creatively incorporating a promotion for [product/service]. The caption should be relevant to the occasion and evoke a festive mood.",
    "Write a technical Instagram caption for Tanji Company that focuses on the features and specifications of [product name]. Pair it with a work environment photo that showcases the innovation behind the product. The tone should be professional yet easy to understand.",
    "Write an Instagram caption for Tanji Company that reflects on recent business growth and challenges. Share your thoughts on what you've learned and how you plan to improve. Be honest and transparent, showing your commitment to continuous improvement."
]
'''

# Function to assign prompts to specific days in a month
def assign_prompts_to_dates(year, month, prompts):
    total_days = calendar.monthrange(year, month)[1]

    # Assign prompts to every 2nd or 3rd day of the month
    schedule = {}
    day = 2
    prompt_index = 0
    
    while day <= total_days and prompt_index < len(prompts):
        date_key = datetime.date(year, month, day)
        schedule[date_key] = prompts[prompt_index]
        day += 2 if prompt_index % 2 == 0 else 3
        prompt_index += 1

    # Add default entries for days without specific prompts
    for i in range(1, total_days + 1):
        date_key = datetime.date(year, month, i)
        if date_key not in schedule:
            schedule[date_key] = "Let's take a break."
    
    return schedule

# Function to check today's prompt if available
def check_today_prompt(schedule):
    today = datetime.date.today()  # Get today's date
    
    # Check if today's date is in the schedule
    return schedule.get(today, "Let's take a break.")

# Example: Assigning prompts
year = datetime.datetime.now().year
month = datetime.datetime.now().month

schedule = assign_prompts_to_dates(year, month, prompts)
cc = check_today_prompt(schedule)

### RAG process
# Hugging Face API Token
API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
# URL of the Hugging Face model API
EMBEDDING_MODEL_URL = "https://api-inference.huggingface.co/models/mixedbread-ai/mxbai-embed-large-v1"

headers = {
    "Authorization": f"Bearer {API_TOKEN}"
}

# Document database (can be replaced with your own documents)
documents = [
    "Tanji Company has no exclusive products.",
    "The products sold by Tanji Company have one thing in common: they are all brand new or nearly new.",
    "The brand concept is to allow customers to buy brand-new products that I don't need at the least amount of money.",
    "'Let's take a break' means teatime.",
    "Tanji Company will not produce its own products."
]

# Embedding generation function, calling Hugging Face API to generate embedding
def embed_texts_with_retry(texts, max_retries=3):
    embeddings = []
    
    for text in texts:
        for attempt in range(max_retries):
            try:
                payload = {
                    "inputs": text,
                    "options": {"wait_for_model": True}
                }
                
                response = requests.post(EMBEDDING_MODEL_URL, headers=headers, json=payload)
                
                if response.status_code == 200:
                    embeddings.append(response.json())
                    break
                elif response.status_code == 429:  # Rate limit
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                else:
                    raise Exception(f"API Error: {response.status_code}")
                    
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                time.sleep(1)
    
    return np.array(embeddings)

'''        
def embed_texts(texts):
    embeddings = []
    for text in texts:
        payload = {
            "inputs": text,
            "options": {
                "wait_for_model": True
            }
        }
        response = requests.post(EMBEDDING_MODEL_URL, headers=headers, json=payload)
        if response.status_code == 200:
            embeddings.append(response.json())
        else:
            raise Exception(f"Error: {response.status_code}, {response.text}")
    return np.array(embeddings)
'''

# Build a vector database (using FAISS)
def create_faiss_index(embeddings):
    d = embeddings.shape[1]  # Dimension of the embedding vectors
    index = faiss.IndexFlatL2(d)  # Using the L2 distance
    index.add(embeddings)  # Add vectors to database
    return index

# Search for the most relevant documents
def search(query, index, documents, top_k=2):
    query_embedding = embed_texts([query])
    distances, indices = index.search(query_embedding, top_k)
    results = [documents[i] for i in indices[0]]
    return results

# Build embedding vectors and a vector database
document_embeddings = embed_texts(documents)
index = create_faiss_index(document_embeddings)

# Query and retrieve relevant documents
search_results = search(cc, index, documents)

# Pass the retrieved documents as context to the generative model
context = " ".join(search_results)

### Generate copywriting
# Initialize the Google GenAI client
client = genai.Client(api_key=os.getenv("Genai_API"))
today = datetime.date.today()

# Define the file name to be saved
output_file = f"generated_instagram_caption_{today}.txt"

# Generate and save content
prompt = f"Context: {context}\n\nQuestion: {cc}\n\nAnswer:"

try:
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    
    # Extract the generated text
    generated_content = response.text
    
    # Save to file
    with open(output_file, "w", encoding="utf-8") as file:
        file.write(generated_content)
        
    #print(f"Generated content saved to {tran_file}")
    
except Exception as e:
    #print(f"Error generating content: {e}")
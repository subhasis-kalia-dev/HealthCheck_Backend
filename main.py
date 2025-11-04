import os
import io
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from google.cloud import vision
from google.cloud.vision_v1 import types
from google.protobuf.json_format import MessageToDict
from openai import OpenAI

# Load environment variables (like OPENAI_API_KEY) from .env file
load_dotenv()

# --- Initialize Clients ---
try:
    # Google Cloud Vision Client uses GOOGLE_APPLICATION_CREDENTIALS for auth
    vision_client = vision.ImageAnnotatorClient()
except Exception as e:
    print(f"Error initializing Google Vision client. Check GOOGLE_APPLICATION_CREDENTIALS: {e}")
    # In a real app, you might raise an error here, but we continue for FastAPI definition.

try:
    # OpenAI Client uses OPENAI_API_KEY environment variable
    openai_client = OpenAI()
except Exception as e:
    print(f"Error initializing OpenAI client. Check OPENAI_API_KEY: {e}")


# --- FastAPI Setup ---
app = FastAPI(
    title="Nutrition Label Analyzer API",
    description="Backend service for OCR (Cloud Vision) and LLM Summarization (OpenAI) of food labels."
)

# Allow CORS for local development with your React frontend
rigins = [
    "http://localhost:5173",  # Your React development server
    "http://127.0.0.1:8000",  # Local testing
    
    # --- Production URLs ---
    # The LIVE Netlify Frontend URL:
    "https://ingredienthealthcheck.netlify.app", 
    # The LIVE Render Backend URL (allows API to talk to itself, good practice):
    "https://healthcheck-backend-g2hd.onrender.com", 
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all request headers
)

# --- LLM Prompts ---
SYSTEM_PROMPT = """
You are an expert nutrition analyst and food scientist.
Your task is to analyze the provided raw text from an OCR scan of a food product label.

Analyze and summarize the data according to the following guidelines:
1.  **Key Ingredients:** List the first 5-7 major ingredients.
2.  **Nutritional Highlights:** State the values for Energy (Calories), Total Fat, Total Sugar, and Total Sodium *per serving*.
3.  **Overall Assessment:** Provide a single, short paragraph (max 100 words) summarizing whether the product is high or low in sodium, fat, or sugar based on general health guidelines.

Format your response clearly, using a professional and factual tone.
Do not include any bullet points or lists in the final response. Structure it as a cohesive report.
"""

def format_llm_input(ocr_text: str, labels: list) -> str:
    """Formats the raw OCR and detected labels into a clean string for the LLM."""
    input_str = "--- RAW OCR TEXT ---\n"
    input_str += ocr_text + "\n\n"
    
    if labels:
        label_list = [f"{label['description']} (Confidence: {label['score']:.2f})" for label in labels]
        input_str += "--- DETECTED OBJECTS/SCENE LABELS ---\n"
        input_str += ", ".join(label_list) + "\n"
        
    input_str += "\n--- END OF DATA ---"
    return input_str

# --- Vision and Analysis Endpoint ---

@app.post("/analyze-image")
async def analyze_image_with_llm(image_file: UploadFile = File(...)):
    """Receives an image, performs OCR/Label detection, and uses LLM for structured analysis."""
    
    try:
        # 1. Read the image content from the upload
        image_content = await image_file.read()

        # 2. Package data for Vision API
        image = types.Image(content=image_content)

        # Define the features we want to request
        features = [
            types.Feature(type=types.Feature.Type.TEXT_DETECTION),
            types.Feature(type=types.Feature.Type.LABEL_DETECTION, max_results=5)
        ]

        # 3. Call Google Cloud Vision API
        response = vision_client.annotate_image(
            request={"image": image, "features": features}
        )
        
        # Convert Protobuf response to dictionary for easier parsing
        vision_results = MessageToDict(response._pb)

        # 4. Extract Detected Data (Text and Labels)
        
        # Get the full text from OCR (usually the first element in textAnnotations)
        detected_text = vision_results.get('textAnnotations', [{}])[0].get('description', 'No text detected.')
        
        # Get the labels
        labels = vision_results.get('labelAnnotations', [])

        # 5. Format input for the LLM
        llm_input_content = format_llm_input(detected_text, labels)

    except Exception as e:
        print(f"Vision API/Processing Error: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Image processing failed with Google Cloud Vision: {e}"
        )

    # --- LLM Analysis Phase ---
    try:
        llm_response = openai_client.chat.completions.create(
            model="gpt-4o",  # Using a powerful model for better structured extraction
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": llm_input_content}
            ]
        )
        summary_text = llm_response.choices[0].message.content

        # 6. Return the final structured summary
        return {"llm_analysis_summary": summary_text}

    except Exception as e:
        print(f"OpenAI API Call Error: {e}")
        # Handle API key issues, rate limits, or other LLM errors
        raise HTTPException(
            status_code=500, 
            detail="Failed to generate summary from AI. Please check OPENAI_API_KEY and service status."
        )

# --- RUNNING INSTRUCTIONS ---
# To run this file locally, save it as 'vision_analysis_backend.py' and execute in your terminal:
# 1. pip install -r requirements.txt (with all necessary packages)
# 2. Set environment variables:
#    export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/key-file.json"
#    (Ensure you have a .env file or export OPENAI_API_KEY='your-key')
# 3. uvicorn vision_analysis_backend:app --reload --host 0.0.0.0 --port 8000

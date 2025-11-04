import os
import json
import base64
from io import BytesIO

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from google.cloud import vision
from google.protobuf.json_format import MessageToDict
from google.oauth2 import service_account
from openai import OpenAI

load_dotenv()

openai_client = None
try:
    openai_client = OpenAI()
except Exception as e:
    print(f"ERROR: Failed to initialize OpenAI Client: {e}")

vision_client = None

def initialize_vision_client():
    global vision_client
    
    json_base64 = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON")
    if json_base64:
        try:
            key_json_bytes = base64.b64decode(json_base64)
            key_json_str = key_json_bytes.decode('utf-8')
            key_info = json.loads(key_json_str)

            credentials = service_account.Credentials.from_service_account_info(key_info)
            
            vision_client = vision.ImageAnnotatorClient(credentials=credentials)
            print("INFO: Successfully initialized Vision client from GOOGLE_APPLICATION_CREDENTIALS_JSON.")
            return

        except Exception as e:
            print(f"ERROR: Failed to initialize client from GOOGLE_APPLICATION_CREDENTIALS_JSON: {e}")

    try:
        vision_client = vision.ImageAnnotatorClient()
        print("INFO: Successfully initialized Vision client using Application Default Credentials (ADC) or path.")
        
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to initialize Vision client using any method. Details: {e}")
        raise RuntimeError("Failed to initialize Google Vision client. Check GOOGLE_APPLICATION_CREDENTIALS.")

initialize_vision_client()

app = FastAPI()

origins = [
    "http://localhost:5173",
    "https://ingredienthealthcheck.netlify.app",
    "https://healthcheck-backend-g2hd.onrender.com"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def vision_process_uploaded_image(image_file: UploadFile):
    if vision_client is None:
        raise HTTPException(status_code=500, detail="Vision client not initialized.")
        
    try:
        image_content = await image_file.read() 
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read uploaded image file: {e}")

    try:
        image = vision.Image(content=image_content)

        features = [
            vision.Feature(type_=vision.Feature.Type.TEXT_DETECTION),
            vision.Feature(type_=vision.Feature.Type.LABEL_DETECTION),
        ]

        response = vision_client.annotate_image(
            {"image": image, "features": features}
        )

        vision_results = MessageToDict(response._pb)
        
        detected_text = vision_results.get('textAnnotations', [{}])[0].get('description', 'No text detected.')
        
        labels = [
            f"{label.get('description')} (Score: {round(float(label.get('score', 0)), 2)})"
            for label in vision_results.get('labelAnnotations', [])[:5]
        ]
        
        return {
            "text": detected_text,
            "labels": labels
        }

    except Exception as e:
        print(f"Cloud Vision API Error: {e}")
        raise HTTPException(status_code=500, detail=f"Cloud Vision API call failed: {e}. Check image format and API access.")

async def analyze_with_openai(detected_text: str, labels: list) -> str:
    if not os.getenv('OPENAI_API_KEY'):
        return "LLM Analysis failed: OPENAI_API_KEY is not configured on the server."
        
    prompt = f"""
    You are an expert food and health analyst. Your task is to analyze a food label.
    
    --- RAW TEXT (OCR) ---
    {detected_text}
    
    --- DETECTED OBJECTS ---
    {', '.join(labels)}
    
    --- ANALYSIS INSTRUCTIONS ---
    1. Identify the main ingredients from the RAW TEXT.
    2. Provide a brief health summary (1-2 sentences) focusing on potential allergens, high sugar/salt content, or harmful additives.
    3. Conclude with a clear recommendation (e.g., "Good choice," "Consume in moderation," or "Avoid").
    
    Provide the analysis in a clear, readable format. Do not include introductory text or the prompt itself.
    """

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert food and health analyst providing concise, actionable summaries."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4,
            max_tokens=400
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"OpenAI API Error: {e}")
        return f"LLM Analysis failed due to an API error: {e}"

@app.post("/analyze_image/")
async def analyze_image_endpoint(image_file: UploadFile = File(...)):
    print(f"INFO: Received file: {image_file.filename}")
    
    vision_results = await vision_process_uploaded_image(image_file)
    detected_text = vision_results["text"]
    labels = vision_results["labels"]
    
    print(f"INFO: Vision detected {len(detected_text)} characters and {len(labels)} labels.")
    
    openai_summary = await analyze_with_openai(detected_text, labels)
    
    return {
        "summary": openai_summary,
        "raw_text_detected": detected_text,
        "detected_labels": labels
    }

@app.get("/")
def read_root():
    return {"message": "Image Analysis API is running. Use the /analyze_image/ endpoint for POST requests."}

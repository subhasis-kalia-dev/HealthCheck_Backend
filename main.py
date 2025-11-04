import os
import json
import base64
from io import BytesIO

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from google.cloud import vision
from google.protobuf.json_format import MessageToDict

# --- CORE VISION AUTHENTICATION FIX ---
# This function handles authentication for both local (ADC file) and Render (JSON string).
def initialize_vision_client():
    # 1. Check for the Base64 JSON string (used for Render deployment)
    json_base64 = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")
    if json_base64:
        try:
            # Decode the Base64 string to get the raw service account key JSON
            key_json = base64.b64decode(json_base64).decode('utf-8')
            key_info = json.loads(key_json)
            
            # Use the credentials directly from the key dictionary
            credentials = vision.credentials.from_service_account_info(key_info)
            client = vision.ImageAnnotatorClient(credentials=credentials)
            print("Google Cloud Vision client initialized using Base64 JSON (Render).")
            return client
        except Exception as e:
            print(f"ERROR: Failed to initialize client from GOOGLE_APPLICATION_CREDENTIALS_JSON: {e}")
            return None
    
    # 2. If Base64 is not found, try default ADC search (used for local deployment)
    try:
        client = vision.ImageAnnotatorClient()
        print("Google Cloud Vision client initialized using default ADC (Local).")
        return client
    except Exception as e:
        print(f"ERROR: Failed to initialize Google Cloud Vision client using default ADC: {e}")
        return None

# Initialize the Vision API Client once
# Note: We need to import 'vision.credentials' first, which is why we must place this after imports.
try:
    vision_client = initialize_vision_client()
    if vision_client is None:
        raise Exception("Failed to initialize Vision client using any method.")
except Exception as e:
    vision_client = None
    print(f"CRITICAL ERROR: {e}")
# --- END CORE VISION AUTHENTICATION FIX ---


app = FastAPI(
    title="Cloud Vision Image Analysis",
    description="FastAPI endpoint for uploading an image and analyzing it using the Google Cloud Vision API.",
    version="1.0.0"
)

# --- CORS Configuration (Cross-Origin Resource Sharing) ---
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:8000",
    "https://ingredienthealthcheck.netlify.app", 
    "https://healthcheck-backend-g2hd.onrender.com", 
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ------------------------------------------------------------


@app.get("/", tags=["Root"])
def read_root():
    """Simple health check endpoint."""
    auth_status = "Authenticated" if vision_client else "Authentication Failed"
    return {"message": "Cloud Vision API is running.", "auth_status": auth_status}


@app.post("/analyze-image", tags=["Vision Analysis"])
async def analyze_uploaded_image(image_file: UploadFile = File(...)):
    """
    Analyzes an uploaded image using the Cloud Vision API to detect labels and safe search attributes.
    """
    if vision_client is None:
        # Check if the client failed to initialize earlier
        raise HTTPException(
            status_code=500,
            detail="Vision API Client not initialized. Check server logs and GOOGLE_APPLICATION_CREDENTIALS_JSON variable."
        )

    # 1. Read the image file content
    try:
        image_content = await image_file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read image file: {e}")

    # 2. Create the Vision API image object
    image = vision.Image(content=image_content)

    # 3. Configure and make the API request
    try:
        features = [
            vision.Feature(type_=vision.Feature.Type.LABEL_DETECTION),
            vision.Feature(type_=vision.Feature.Type.SAFE_SEARCH_DETECTION),
        ]

        response = vision_client.annotate_image(
            {
                "image": image,
                "features": features
            }
        )
    except Exception as e:
        print(f"Vision API Call Error: {e}")
        raise HTTPException(status_code=500, detail=f"Vision API request failed. Error: {e}")

    # 4. Process the response
    results = MessageToDict(response._pb)

    # Extract the relevant parts for a clean response
    processed_response = {
        "filename": image_file.filename,
        "content_type": image_file.content_type,
        "labels": [],
        "safe_search_attributes": {}
    }
    
    # Extract Labels
    if 'labelAnnotations' in results:
        processed_response["labels"] = [
            {"description": label.get("description"), "score": round(float(label.get("score", 0)), 4)}
            for label in results['labelAnnotations'][:5]
        ]

    # Extract Safe Search Results
    if 'safeSearchAnnotation' in results:
        processed_response["safe_search_attributes"] = results['safeSearchAnnotation']


    # 5. Return the structured results
    return {
        "status": "success",
        "analysis_results": processed_response
    }

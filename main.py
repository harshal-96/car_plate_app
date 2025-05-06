from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np
import easyocr
import io
import base64
import requests
from bs4 import BeautifulSoup
import os
import uvicorn
import logging
import time
import uuid
from datetime import datetime
import sys

# Configure logging
# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

# Configure root logger for general application logs
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"logs/app_{datetime.now().strftime('%Y-%m-%d')}.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

# Create specific loggers for different components
api_logger = logging.getLogger('api')
model_logger = logging.getLogger('model')
ocr_logger = logging.getLogger('ocr')
lookup_logger = logging.getLogger('lookup')

# Set logging levels
api_logger.setLevel(logging.DEBUG)
model_logger.setLevel(logging.INFO)
ocr_logger.setLevel(logging.INFO)
lookup_logger.setLevel(logging.INFO)

# Add file handlers for component-specific logs
api_file_handler = logging.FileHandler(f"logs/api_{datetime.now().strftime('%Y-%m-%d')}.log")
api_file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
api_logger.addHandler(api_file_handler)

model_file_handler = logging.FileHandler(f"logs/model_{datetime.now().strftime('%Y-%m-%d')}.log")
model_file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
model_logger.addHandler(model_file_handler)

ocr_file_handler = logging.FileHandler(f"logs/ocr_{datetime.now().strftime('%Y-%m-%d')}.log")
ocr_file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
ocr_logger.addHandler(ocr_file_handler)

lookup_file_handler = logging.FileHandler(f"logs/lookup_{datetime.now().strftime('%Y-%m-%d')}.log")
lookup_file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
lookup_logger.addHandler(lookup_file_handler)

# Define the FastAPI app
app = FastAPI(
    title="Car Classification & License Plate Detection API",
    description="API for car view classification and license plate detection",
    version="1.0.0"
)

# Middleware for request logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = str(uuid.uuid4())
    start_time = time.time()
    
    # Log request details
    api_logger.info(f"Request {request_id} started: {request.method} {request.url.path}")
    
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        
        # Log response time
        api_logger.info(f"Request {request_id} completed: {response.status_code} in {process_time:.4f}s")
        
        # Add request ID to response headers for tracking
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time"] = str(process_time)
        
        return response
    except Exception as e:
        api_logger.error(f"Request {request_id} failed: {str(e)}")
        raise e

# Set up CORS to allow requests from frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (HTML, CSS, JS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Define model architecture
class CarSegmentationModel(nn.Module):
    def __init__(self, num_classes=3):
        super(CarSegmentationModel, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# Global variables for models
model = None
device = None
reader = None

# Initialize models on startup
@app.on_event("startup")
async def startup_event():
    global model, device, reader
    logging.info("Starting application initialization")
    
    try:
        # Initialize device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_logger.info(f"Using device: {device}")
        
        # Initialize model
        model_logger.info("Initializing car segmentation model")
        model = CarSegmentationModel(num_classes=3).to(device)
        
        # In a real app, you'd load your trained model here
        try:
            model_logger.info("Loading model weights from car_segmentation_model.pth")
            model.load_state_dict(torch.load("car_segmentation_model.pth", map_location=device))
            model_logger.info("Model weights loaded successfully")
        except Exception as e:
            model_logger.warning(f"Failed to load model weights: {str(e)}")
            model_logger.warning("Using untrained model for demo purposes")
        
        # For demo purposes, let's just use the pretrained model
        model.eval()
        model_logger.info("Model set to evaluation mode")
        
        # Initialize OCR reader
        ocr_logger.info("Initializing EasyOCR reader")
        reader = easyocr.Reader(['en'])
        ocr_logger.info("EasyOCR reader initialized")
        
        logging.info("Application initialized successfully")
    except Exception as e:
        logging.error(f"Error during application startup: {str(e)}")
        raise e

@app.on_event("shutdown")
async def shutdown_event():
    logging.info("Application shutting down")

# Helper functions
def predict_car_view(image, model, device):
    model_logger.debug("Starting car view prediction")
    start_time = time.time()
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    image_tensor = transform(image).unsqueeze(0).to(device)
    model_logger.debug("Image transformed and loaded to device")
    
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    class_labels = ["back", "front", "side"]
    predicted_view = class_labels[predicted_class]
    
    process_time = time.time() - start_time
    model_logger.info(f"Car view prediction: {predicted_view} with confidence {confidence:.4f} in {process_time:.4f}s")
    
    return predicted_view

def detect_license_plate(image, reader):
    ocr_logger.debug("Starting license plate detection")
    start_time = time.time()
    
    # Convert PIL Image to OpenCV format
    img_array = np.array(image)
    img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # Convert to grayscale for better OCR
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    ocr_logger.debug("Image converted to grayscale")
    
    # Perform OCR
    ocr_start_time = time.time()
    results = reader.readtext(gray)
    ocr_time = time.time() - ocr_start_time
    ocr_logger.debug(f"OCR completed in {ocr_time:.4f}s. Found {len(results)} text regions")
    
    # Prepare results
    extracted_text = []
    probabilities = {}
    
    # Process results
    for idx, (bbox, text, prob) in enumerate(results):
        cleaned_text = "".join(char.upper() for char in text if char.isalnum())
        
        if cleaned_text:
            extracted_text.append(cleaned_text)
            probabilities[cleaned_text] = prob
            ocr_logger.debug(f"Found text region {idx}: '{text}' -> '{cleaned_text}' with probability {prob:.4f}")
            
            # Get coordinates for drawing
            (top_left, top_right, bottom_right, bottom_left) = bbox
            top_left = tuple(map(int, top_left))
            bottom_right = tuple(map(int, bottom_right))
            
            # Draw rectangle and text
            cv2.rectangle(img_cv, top_left, bottom_right, (0, 255, 0), 2)
            cv2.putText(img_cv, cleaned_text, (top_left[0], top_left[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Convert back to RGB for display
    img_with_detections = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    
    # Convert processed image to base64 for frontend display
    _, buffer = cv2.imencode('.jpg', img_with_detections)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    
    # Filter potential plates
    potential_plates = [text for text in extracted_text if 6 <= len(text) <= 12]
    ocr_logger.debug(f"Filtered to {len(potential_plates)} potential plates")
    
    # Find best plate
    best_plate = None
    if potential_plates:
        best_plate = max(potential_plates, key=lambda p: probabilities[p])
        best_plate = best_plate[:10]  # Limit to 10 characters
        ocr_logger.info(f"Best plate found: {best_plate} with confidence {probabilities[best_plate]:.4f}")
    else:
        ocr_logger.info("No license plate detected")
    
    process_time = time.time() - start_time
    ocr_logger.debug(f"License plate detection completed in {process_time:.4f}s")
    
    return img_base64, best_plate

def correct_text(text, expected_type):
    ocr_logger.debug(f"Correcting text '{text}' with expected type '{expected_type}'")
    correction_dict = {
        '0': 'O', '1': 'I', '2': 'Z', '3': 'B', '4': 'L', '5': 'S', '6': 'G', '7': 'T', '8': 'B', '9': 'P',
        'O': '0', 'I': '1', 'Z': '2', 'B': '3', 'L': '4', 'S': '5', 'G': '6', 'T': '7', 'B': '8', 'P': '9'
    }
    
    corrected_text = ""
    for char in text:
        if expected_type == "alpha" and char.isdigit():
            corrected_text += correction_dict.get(char, char)
        elif expected_type == "numeric" and char.isalpha():
            corrected_text += correction_dict.get(char, char)
        else:
            corrected_text += char
    
    ocr_logger.debug(f"Corrected text: '{text}' -> '{corrected_text}'")
    return corrected_text

def strict_split_number_plate(number_plate):
    ocr_logger.debug(f"Splitting number plate: {number_plate}")
    
    if len(number_plate) < 8:
        ocr_logger.warning(f"Plate '{number_plate}' too short (length {len(number_plate)}), minimum 8 characters required")
        return None, None, None, None
    
    # Extract parts
    part1 = number_plate[:2]
    part2 = number_plate[2:4]
    part4 = number_plate[-4:]
    part3 = number_plate[-6:-4] if len(number_plate) >= 10 else number_plate[-5]
    
    # Apply corrections
    part1 = correct_text(part1, "alpha")
    part2 = correct_text(part2, "numeric")
    part3 = correct_text(part3, "alpha")
    part4 = correct_text(part4, "numeric")
    
    ocr_logger.debug(f"Split plate: {part1}-{part2}-{part3}-{part4}")
    return part1, part2, part3, part4

# Vehicle lookup functions
def get_vehicle_details_paid(plate_number):
    lookup_logger.info(f"Looking up vehicle details (paid method) for plate: {plate_number}")
    start_time = time.time()
    
    try:
        url = "https://rto-vehicle-information-india.p.rapidapi.com/getVehicleInfo"
        
        payload = {
            "vehicle_no": plate_number,
            "consent": "Y",
            "consent_text": "I hereby give my consent for Eccentric Labs API to fetch my information"
        }
        
        headers = {
            "x-rapidapi-key": "83ed10f183mshe1c3f0fe8025d7ap1f9c9bjsn0d1e4be443fc",
            "x-rapidapi-host": "rto-vehicle-information-india.p.rapidapi.com",
            "Content-Type": "application/json"
        }
        
        lookup_logger.debug(f"Sending request to: {url}")
        response = requests.post(url, json=payload, headers=headers)
        
        if response.status_code != 200:
            lookup_logger.warning(f"API returned non-200 status code: {response.status_code}")
            lookup_logger.debug(f"API response: {response.text}")
            return None
        
        data = response.json()
        lookup_logger.debug(f"API response: {data}")
        
        # Check if the API returned valid data
        if data.get("status") and data.get("data"):
            process_time = time.time() - start_time
            lookup_logger.info(f"Vehicle details found in {process_time:.4f}s")
            return data["data"]
        else:
            lookup_logger.warning("API response did not contain expected data structure")
            return None
            
    except Exception as e:
        lookup_logger.error(f"Error in paid vehicle lookup: {str(e)}")
        return None

def get_vehicle_details_free(plate_number):
    lookup_logger.info(f"Looking up vehicle details (free method) for plate: {plate_number}")
    start_time = time.time()
    
    try:
        url = f"https://www.carinfo.app/rto-vehicle-registration-detail/rto-details/{plate_number}"
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        lookup_logger.debug(f"Sending request to: {url}")
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code != 200:
            lookup_logger.warning(f"Website returned non-200 status code: {response.status_code}")
            return None
        
        lookup_logger.debug("Parsing HTML response")
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract vehicle details with error handling
        try:
            make_model = soup.find('p', class_='input_vehical_layout_vehicalModel__1ABTF').text.strip()
            owner_name = soup.find('p', class_='input_vehical_layout_ownerName__NHkpi').text.strip()
            rto_number = soup.find('p', class_='expand_component_itemSubTitle__ElsYf').text.strip()
            
            lookup_logger.debug(f"Found basic details: {make_model}, {owner_name}, {rto_number}")
            
            # Get all subtitle elements
            subtitles = soup.find_all('p', class_='expand_component_itemSubTitle__ElsYf')
            
            # Extract details if available
            rto_address = subtitles[1].text.strip() if len(subtitles) > 1 else "Not available"
            state = subtitles[2].text.strip() if len(subtitles) > 2 else "Not available"
            phone = subtitles[3].text.strip() if len(subtitles) > 3 else "Not available"
            
            # Get website with fallback
            website = "Not available"
            if len(subtitles) > 4 and subtitles[4].find('a'):
                website = subtitles[4].find('a')['href']
            
            vehicle_data = {
                "maker_model": make_model,
                "owner_name": owner_name,
                "registration_no": plate_number,
                "registration_authority": rto_number,
                "rto_address": rto_address,
                "state": state,
                "rto_phone": phone,
                "website": website
            }
            
            process_time = time.time() - start_time
            lookup_logger.info(f"Vehicle details found in {process_time:.4f}s")
            lookup_logger.debug(f"Vehicle data: {vehicle_data}")
            
            return vehicle_data
            
        except (AttributeError, IndexError) as e:
            lookup_logger.error(f"Error parsing vehicle details: {str(e)}")
            return None
            
    except Exception as e:
        lookup_logger.error(f"Error in free vehicle lookup: {str(e)}")
        return None

# API Endpoints
@app.get("/")
async def root():
    api_logger.info("Root endpoint accessed")
    return {"message": "API is running. Access the web interface at /static/index.html"}

# Car classification endpoint
@app.post("/classify-car/")
async def classify_car(file: UploadFile = File(...)):
    request_id = str(uuid.uuid4())
    api_logger.info(f"[{request_id}] Car classification request received: {file.filename}")
    
    global model, device
    
    if not model or not device:
        api_logger.error(f"[{request_id}] Model not initialized")
        raise HTTPException(status_code=500, detail="Model not initialized")
    
    try:
        # Read image
        image_data = await file.read()
        api_logger.debug(f"[{request_id}] Image size: {len(image_data)} bytes")
        
        image = Image.open(io.BytesIO(image_data))
        api_logger.debug(f"[{request_id}] Image loaded: {image.format}, {image.size}")
        
        # Classify car view
        car_view = predict_car_view(image, model, device)
        
        # Convert image to base64 for response
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        api_logger.info(f"[{request_id}] Classification successful: {car_view}")
        
        return {
            "success": True,
            "car_view": car_view,
            "image_base64": img_base64,
            "request_id": request_id
        }
    except Exception as e:
        api_logger.error(f"[{request_id}] Error processing image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

# License plate detection endpoint
@app.post("/detect-license-plate/")
async def detect_license(file: UploadFile = File(...)):
    request_id = str(uuid.uuid4())
    api_logger.info(f"[{request_id}] License plate detection request received: {file.filename}")
    
    global reader
    
    if not reader:
        api_logger.error(f"[{request_id}] OCR reader not initialized")
        raise HTTPException(status_code=500, detail="OCR reader not initialized")
    
    try:
        # Read image
        image_data = await file.read()
        api_logger.debug(f"[{request_id}] Image size: {len(image_data)} bytes")
        
        image = Image.open(io.BytesIO(image_data))
        api_logger.debug(f"[{request_id}] Image loaded: {image.format}, {image.size}")
        
        # Detect license plate
        img_base64, best_plate = detect_license_plate(image, reader)
        
        result = {
            "success": True,
            "license_detected": best_plate is not None,
            "processed_image_base64": img_base64,
            "request_id": request_id
        }
        
        if best_plate:
            # Process the license plate
            part1, part2, part3, part4 = strict_split_number_plate(best_plate)
            
            if part1 and part2 and part3 and part4:
                corrected_plate = part1 + part2 + part3 + part4
                result["raw_plate"] = best_plate
                result["corrected_plate"] = corrected_plate
                api_logger.info(f"[{request_id}] License plate detected and corrected: {best_plate} -> {corrected_plate}")
            else:
                result["raw_plate"] = best_plate
                result["corrected_plate"] = None
                api_logger.info(f"[{request_id}] License plate detected but could not be corrected: {best_plate}")
        else:
            api_logger.info(f"[{request_id}] No license plate detected")
        
        return result
    except Exception as e:
        api_logger.error(f"[{request_id}] Error processing image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

# Full processing endpoint (classification, license detection, vehicle lookup)
@app.post("/process-car-image/")
async def process_car_image(
    file: UploadFile = File(...),
    lookup_method: str = Form("free")
):
    request_id = str(uuid.uuid4())
    api_logger.info(f"[{request_id}] Full car image processing request received: {file.filename}, lookup method: {lookup_method}")
    
    global model, device, reader
    
    if not model or not device or not reader:
        api_logger.error(f"[{request_id}] Models not initialized")
        raise HTTPException(status_code=500, detail="Models not initialized")
    
    try:
        start_time = time.time()
        
        # Read image
        image_data = await file.read()
        api_logger.debug(f"[{request_id}] Image size: {len(image_data)} bytes")
        
        image = Image.open(io.BytesIO(image_data))
        api_logger.debug(f"[{request_id}] Image loaded: {image.format}, {image.size}")
        
        # 1. Classify car view
        car_view = predict_car_view(image, model, device)
        
        # 2. Detect license plate
        img_base64, best_plate = detect_license_plate(image, reader)
        
        result = {
            "success": True,
            "car_view": car_view,
            "processed_image_base64": img_base64,
            "license_detected": best_plate is not None,
            "request_id": request_id
        }
        
        # 3. Process license plate if detected
        if best_plate:
            part1, part2, part3, part4 = strict_split_number_plate(best_plate)
            
            if part1 and part2 and part3 and part4:
                corrected_plate = part1 + part2 + part3 + part4
                result["raw_plate"] = best_plate
                result["corrected_plate"] = corrected_plate
                api_logger.info(f"[{request_id}] License plate detected and corrected: {best_plate} -> {corrected_plate}")
                
                # 4. Lookup vehicle details
                lookup_start_time = time.time()
                if lookup_method.lower() == "paid":
                    vehicle_data = get_vehicle_details_paid(corrected_plate)
                else:
                    vehicle_data = get_vehicle_details_free(corrected_plate)
                
                lookup_time = time.time() - lookup_start_time
                
                if vehicle_data:
                    result["vehicle_details"] = vehicle_data
                    result["vehicle_details_found"] = True
                    api_logger.info(f"[{request_id}] Vehicle details lookup successful in {lookup_time:.4f}s")
                else:
                    result["vehicle_details_found"] = False
                    api_logger.info(f"[{request_id}] Vehicle details lookup failed in {lookup_time:.4f}s")
            else:
                result["raw_plate"] = best_plate
                result["corrected_plate"] = None
                result["vehicle_details_found"] = False
                api_logger.info(f"[{request_id}] License plate could not be properly formatted: {best_plate}")
        else:
            api_logger.info(f"[{request_id}] No license plate detected")
        
        total_time = time.time() - start_time
        api_logger.info(f"[{request_id}] Full processing completed in {total_time:.4f}s")
        
        return result
    except Exception as e:
        api_logger.error(f"[{request_id}] Error processing image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

# Vehicle lookup endpoint
@app.post("/lookup-vehicle/")
async def lookup_vehicle(
    plate_number: str = Form(...),
    lookup_method: str = Form("free")
):
    request_id = str(uuid.uuid4())
    api_logger.info(f"[{request_id}] Vehicle lookup request received: {plate_number}, method: {lookup_method}")
    
    try:
        start_time = time.time()
        
        if lookup_method.lower() == "paid":
            vehicle_data = get_vehicle_details_paid(plate_number)
        else:
            vehicle_data = get_vehicle_details_free(plate_number)
        
        if vehicle_data:
            lookup_time = time.time() - start_time
            api_logger.info(f"[{request_id}] Vehicle lookup successful in {lookup_time:.4f}s")
            
            return {
                "success": True,
                "vehicle_details": vehicle_data,
                "request_id": request_id
            }
        else:
            lookup_time = time.time() - start_time
            api_logger.info(f"[{request_id}] Vehicle lookup failed in {lookup_time:.4f}s")
            
            return {
                "success": False,
                "message": "Vehicle details not found",
                "request_id": request_id
            }
    except Exception as e:
        api_logger.error(f"[{request_id}] Error looking up vehicle: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error looking up vehicle: {str(e)}")

# Run the application
if __name__ == "__main__":
    logging.info("Starting server")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
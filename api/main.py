import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import numpy as np
import os
from PIL import Image
import pickle
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tempfile
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

# Parameters
IMG_SIZE = (144, 256)
max_words = 5000
max_len = 128
CLASSNAME = ['clean', 'defaced']

# Load models and tokenizer at startup
try:
    resnet_model = load_model('models/ResNet50.h5')
    bilstm_model = load_model('models/BiLSTM.h5')
    with open('models/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
except Exception as e:
    logger.error(f"Error loading models or tokenizer: {e}")
    raise Exception("Failed to load models or tokenizer")

# Pydantic model for request body
class URLRequest(BaseModel):
    url: str

# Function to process HTML
def process_html(url: str) -> str:
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        text = soup.get_text(separator=' ', strip=True)
        return text
    except Exception as e:
        logger.error(f"Error fetching HTML from {url}: {e}")
        return None

# Function to capture screenshot
def capture_screenshot(url: str) -> str:
    try:
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        driver = webdriver.Chrome(options=chrome_options)
        driver.set_window_size(1920, 1080)
        driver.get(url)
        
        # Create temporary file for screenshot
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
            screenshot_path = temp_file.name
            driver.save_screenshot(screenshot_path)
        
        driver.quit()
        return screenshot_path
    except Exception as e:
        logger.error(f"Error capturing screenshot from {url}: {e}")
        return None

# Function to process image
def process_image(image_path: str) -> np.ndarray:
    try:
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=IMG_SIZE)
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        logger.error(f"Error processing image {image_path}: {e}")
        return None

# Function to process text
def process_text(text: str, tokenizer) -> np.ndarray:
    try:
        seq = tokenizer.texts_to_sequences([text])
        padded = pad_sequences(seq, maxlen=max_len, padding='post', truncating='post')
        return padded
    except Exception as e:
        logger.error(f"Error processing text: {e}")
        return None

# Prediction function
def predict(url: str, resnet_model, bilstm_model, tokenizer, weights=[0.5, 0.5]):
    html_text = process_html(url)
    if html_text is None:
        return None, None

    screenshot_path = capture_screenshot(url)
    if screenshot_path is None:
        return None, None

    img_array = process_image(screenshot_path)
    if img_array is None:
        return None, None

    text_padded = process_text(html_text, tokenizer)
    if text_padded is None:
        return None, None

    try:
        resnet_prob = resnet_model.predict(img_array, verbose=0)[0][0]
        bilstm_prob = bilstm_model.predict(text_padded, verbose=0)[0][0]

        combined_prob = weights[0] * resnet_prob + weights[1] * bilstm_prob
        prediction = 1 if combined_prob > 0.5 else 0
        label = CLASSNAME[prediction]
        confidence = combined_prob if prediction == 1 else 1 - combined_prob

        # Clean up screenshot file
        if os.path.exists(screenshot_path):
            os.remove(screenshot_path)

        return label, confidence
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return None, None

# API endpoint for prediction
@app.post("/predict")
async def predict_website(request: URLRequest):
    try:
        label, confidence = predict(
            url=request.url,
            resnet_model=resnet_model,
            bilstm_model=bilstm_model,
            tokenizer=tokenizer
        )
        if label is None or confidence is None:
            raise HTTPException(status_code=400, detail="Unable to process the URL")
        
        return {
            "prediction": label,
            "confidence": float(confidence),
            "url": request.url
        }
    except Exception as e:
        logger.error(f"Error in /predict endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}
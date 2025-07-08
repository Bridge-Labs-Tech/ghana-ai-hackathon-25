
import os
import json
import torch
import uvicorn
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from transformers import BertTokenizer
import torchvision.transforms as transforms
from functools import lru_cache
import logging
from pydantic import BaseModel
import hashlib
from model import FoodClassifier

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Food Classifier API")

# Configuration
class Config:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_size = 224
    max_seq_length = 128
    checkpoint_dir = "checkpoints"
    class_mapping_path = "class_mapping.json"


# --- Helper Functions ---
def load_resources():
    """Load model, tokenizer, and class mapping"""
    # Load class mapping
    with open(Config.class_mapping_path, 'r') as f:
        class_mapping = json.load(f)
    idx_to_class = {v: k for k, v in class_mapping.items()}
    
    # Load model
    model = FoodClassifier(len(class_mapping)).to(Config.device)
    checkpoint = torch.load(
        os.path.join(Config.checkpoint_dir, 'best_model.pth'),
        map_location=torch.device(Config.device)
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Image transformation
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(Config.image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    return model, tokenizer, transform, idx_to_class

# Load resources at startup
model, tokenizer, transform, idx_to_class = load_resources()

# --- Caching System ---
class PredictionCache:
    def __init__(self, max_size=100):
        self.cache = {}
        self.max_size = max_size
        
    def get_key(self, image_bytes: bytes, text: str):
        return hashlib.md5(image_bytes + text.encode()).hexdigest()
    
    def get(self, key):
        return self.cache.get(key, None)
    
    def set(self, key, value):
        if len(self.cache) >= self.max_size:
            self.cache.pop(next(iter(self.cache)))
        self.cache[key] = value

prediction_cache = PredictionCache(max_size=100)


# --- LRU Cache for Text Processing ---
@lru_cache(maxsize=1000)
def preprocess_text(text: str):
    return tokenizer(
        text,
        padding='max_length',
        max_length=Config.max_seq_length,
        truncation=True,
        return_tensors="pt"
    )

# --- Request/Response Models ---
class PredictionRequest(BaseModel):
    text: str = "Image of food"

class PredictionResult(BaseModel):
    class_name: str
    probability: float
    confidence: str

class PredictionResponse(BaseModel):
    predictions: list[PredictionResult]
    cache_hit: bool

# --- API Endpoints ---
@app.post("/predict", response_model=PredictionResponse)
async def predict(
    image: UploadFile = File(..., description="Food image to classify"),
    request: PredictionRequest = None
):
    try:
        # Read image and text
        image_bytes = await image.read()
        text = request.text if request else "Image of food"
        
        # Check cache
        cache_key = prediction_cache.get_key(image_bytes, text)
        if cached := prediction_cache.get(cache_key):
            logger.info("Cache hit for prediction")
            return {**cached, "cache_hit": True}
        
        # Process image
        image_tensor = transform(Image.open(image.file).convert('RGB'))
        image_tensor = image_tensor.unsqueeze(0).to(Config.device)
        
        # Process text (uses LRU cache)
        text_tensor = preprocess_text(text)
        text_tensor = {k: v.to(Config.device) for k, v in text_tensor.items()}
        
        # Run prediction
        with torch.no_grad():
            outputs = model(image_tensor, text_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            top_probs, top_classes = torch.topk(probs, 5)
        
        # Format response
        predictions = []
        for i in range(top_probs.shape[1]):
            class_idx = top_classes[0][i].item()
            prob_value = top_probs[0][i].item()
            predictions.append({
                "class_name": idx_to_class[class_idx].replace('_', ' '),
                "probability": prob_value,
                "confidence": f"{prob_value*100:.2f}%"
            })
        
        # Cache result
        response = {"predictions": predictions, "cache_hit": False}
        prediction_cache.set(cache_key, response)
        
        return response
    
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Classification error")

@app.get("/health")
def health_check():
    return {"status": "healthy", "device": Config.device}

@app.get("/classes")
def list_classes():
    return {"classes": list(idx_to_class.values())}


# --- Monitoring Endpoints ---
@app.get("/system_status")
def system_status():
    return {
        "device": Config.device,
        "cache_size": len(prediction_cache.cache),
        "model": "FoodClassifier",
        "num_classes": len(idx_to_class)
    }

# --- Model Reload Endpoint ---
@app.post("/reload_model")
def reload_model():
    global model, tokenizer, transform, idx_to_class
    try:
        model, tokenizer, transform, idx_to_class = load_resources()
        return {"status": "model reloaded"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Main Execution ---
if __name__ == "__main__":
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        timeout_keep_alive=30
    )
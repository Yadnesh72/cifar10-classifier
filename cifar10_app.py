from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18
from PIL import Image
import io
import base64
import uvicorn

app = FastAPI(
    title="CIFAR-10 Image Classifier",
    description="Classifies images into 10 categories using ResNet18 trained on CIFAR-10 dataset",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# CIFAR-10 class names
CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Global variable to store the model
model = None
device = None

def create_model(num_classes=10, use_dropout=True, dropout_rate=0.5):
    """Create ResNet18 model with modified final layer (matching training config)"""
    model = resnet18(pretrained=False)
    num_features = model.fc.in_features
    
    # Model2 from your notebook uses dropout
    if use_dropout:
        model.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, num_classes)
        )
    else:
        model.fc = nn.Linear(num_features, num_classes)
    
    return model

class PredictionOutput(BaseModel):
    predicted_class: str
    confidence: float
    all_probabilities: dict

@app.on_event("startup")
async def load_model():
    """Load the PyTorch model on startup"""
    global model, device
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        # Create model architecture (matching model2 from notebook - with dropout)
        model = create_model(num_classes=10, use_dropout=True, dropout_rate=0.5)
        
        # Load trained weights
        model.load_state_dict(torch.load('cifar10_classifier.pth', map_location=device))
        model.to(device)
        model.eval()
        
        print(f"Model loaded successfully on {device}!")
        print("Using model2 architecture: ResNet18 with dropout (prevention techniques)")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Running without pre-trained model. Please ensure 'cifar10_classifier.pth' is in the working directory.")

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "CIFAR-10 Image Classifier API",
        "description": "Upload an image to classify it into one of 10 categories",
        "categories": CLASS_NAMES,
        "endpoints": {
            "/predict": "POST - Upload image file for classification",
            "/predict/base64": "POST - Send base64 encoded image",
            "/health": "GET - Check API health status",
            "/docs": "GET - Interactive API documentation"
        },
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device),
        "categories": CLASS_NAMES
    }

@app.post("/predict", response_model=PredictionOutput)
async def predict_image(file: UploadFile = File(...)):
    """
    Upload an image file to classify
    
    Args:
        file: Image file (PNG, JPG, etc.)
    
    Returns:
        PredictionOutput with predicted class and confidence scores
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Check file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read and preprocess image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            confidence, predicted_idx = torch.max(probabilities, 0)
        
        # Get predicted class
        predicted_class = CLASS_NAMES[predicted_idx.item()]
        confidence_score = confidence.item()
        
        # Get all probabilities
        all_probs = {
            CLASS_NAMES[i]: float(probabilities[i].item()) 
            for i in range(len(CLASS_NAMES))
        }
        
        return PredictionOutput(
            predicted_class=predicted_class,
            confidence=confidence_score,
            all_probabilities=all_probs
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

class Base64ImageInput(BaseModel):
    image_base64: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "image_base64": "data:image/png;base64,iVBORw0KGgo..."
            }
        }

@app.post("/predict/base64", response_model=PredictionOutput)
async def predict_base64(input_data: Base64ImageInput):
    """
    Send base64 encoded image for classification
    
    Args:
        input_data: Base64ImageInput containing base64 encoded image string
    
    Returns:
        PredictionOutput with predicted class and confidence scores
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Handle data URL format
        image_data = input_data.image_base64
        if 'base64,' in image_data:
            image_data = image_data.split('base64,')[1]
        
        # Decode base64 image
        img_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            confidence, predicted_idx = torch.max(probabilities, 0)
        
        # Get predicted class
        predicted_class = CLASS_NAMES[predicted_idx.item()]
        confidence_score = confidence.item()
        
        # Get all probabilities
        all_probs = {
            CLASS_NAMES[i]: float(probabilities[i].item()) 
            for i in range(len(CLASS_NAMES))
        }
        
        return PredictionOutput(
            predicted_class=predicted_class,
            confidence=confidence_score,
            all_probabilities=all_probs
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

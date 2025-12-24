from fastapi import FastAPI, File, UploadFile, HTTPException, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from transformers import AutoModel, AutoTokenizer
import torch
import os
from pathlib import Path
import time
from datetime import datetime
from typing import Optional, Literal
import shutil
from PIL import Image
import logging
import asyncio
import json
from huggingface_hub import snapshot_download

from config import settings, PROMPTS

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="DeepSeek OCR API",
    description="API for optical character recognition using DeepSeek-OCR",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for the model
model = None
tokenizer = None
model_loaded = False
model_loading = False
model_error = None
download_progress = {"status": "idle", "progress": 0, "message": ""}


def load_model():
    """Load the DeepSeek-OCR model"""
    global model, tokenizer, model_loaded, model_loading, model_error, download_progress
    
    if model_loaded:
        return
    
    if model_loading:
        return
    
    model_loading = True
    model_error = None
    download_progress["status"] = "downloading"
    download_progress["progress"] = 0
    download_progress["message"] = "Starting model download..."
    
    try:
        logger.info(f"Loading model {settings.MODEL_NAME}...")
        
        download_progress["progress"] = 10
        download_progress["message"] = "Downloading tokenizer..."
        
        tokenizer = AutoTokenizer.from_pretrained(
            settings.MODEL_NAME,
            trust_remote_code=True
        )
        
        download_progress["progress"] = 30
        download_progress["message"] = "Tokenizer downloaded. Downloading model..."
        
        # Try first with flash_attention_2, otherwise use eager
        try:
            logger.info("Attempting to load with flash_attention_2...")
            download_progress["message"] = "Loading model with flash_attention_2..."
            model = AutoModel.from_pretrained(
                settings.MODEL_NAME,
                _attn_implementation='flash_attention_2',
                trust_remote_code=True,
                device_map='auto',
                max_memory={'cuda:0': '4GB'},
                use_safetensors=True
            )
            logger.info("✓ Model loaded with flash_attention_2")
        except Exception as e:
            logger.warning(f"Flash attention not available: {e}")
            logger.info("Loading model with eager attention...")
            download_progress["message"] = "Loading model with eager attention..."
            model = AutoModel.from_pretrained(
                settings.MODEL_NAME,
                _attn_implementation='eager',
                trust_remote_code=True,
                use_safetensors=True
            )
            logger.info("✓ Model loaded with eager attention")
        
        download_progress["progress"] = 80
        download_progress["message"] = "Model downloaded. Configuring..."
        
        # Move to GPU if available
        if settings.DEVICE == "cuda" and torch.cuda.is_available():
            logger.info("Moving model to GPU...")
            download_progress["message"] = "Moving model to GPU..."
            # model = model.eval().cuda().to(torch.bfloat16)
            model = model.eval().cuda().to(torch.float16)
            logger.info(f"✓ Model loaded on GPU: {torch.cuda.get_device_name(0)}")
        else:
            model = model.eval()
            logger.info("✓ Model loaded on CPU")
        
        download_progress["progress"] = 100
        download_progress["status"] = "completed"
        download_progress["message"] = "✓ Model fully loaded and ready"
        
        model_loaded = True
        model_loading = False
        logger.info("✓ Model fully loaded and ready")
        
    except Exception as e:
        model_loading = False
        model_error = str(e)
        download_progress["status"] = "error"
        download_progress["progress"] = 0
        download_progress["message"] = f"Error: {str(e)}"
        logger.error(f"Error loading model: {str(e)}")
        raise


@app.on_event("startup")
async def startup_event():
    """Initialization when starting the application"""
    logger.info("Starting DeepSeek OCR API...")
    
    # Create directories
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    os.makedirs(settings.OUTPUT_DIR, exist_ok=True)
    
    # DO NOT load model on startup - it will be loaded on the first request
    logger.info("✓ API ready. The model will be loaded on the first request.")


@app.get("/")
async def root():
    """Endpoint raíz"""
    return {
        "message": "DeepSeek OCR API",
        "version": "1.0.0",
        "model": settings.MODEL_NAME,
        "model_loaded": model_loaded,
        "device": settings.DEVICE,
        "endpoints": {
            "health": "/health",
            "ocr": "/api/ocr",
            "docs": "/docs"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model_loaded,
        "model_loading": model_loading,
        "model_error": model_error,
        "download_progress": download_progress,
        "device": settings.DEVICE,
        "cuda_available": torch.cuda.is_available(),
        "timestamp": datetime.now().isoformat()
    }


@app.post("/api/download-model")
async def download_model(background_tasks: BackgroundTasks):
    """Starts downloading the model in the background"""
    global model_loading, download_progress
    
    if model_loaded:
        return {"status": "already_loaded", "message": "Model already loaded"}
    
    if model_loading:
        return {"status": "downloading", "message": "Download in progress", "progress": download_progress}
    
    # Start loading in background
    background_tasks.add_task(load_model)
    
    return {"status": "started", "message": "Download started"}


@app.get("/api/download-progress")
async def get_download_progress():
    """Gets the model download progress"""
    return {
        "model_loaded": model_loaded,
        "model_loading": model_loading,
        "progress": download_progress
    }


@app.post("/api/ocr")
async def process_ocr(
    file: UploadFile = File(...),
    mode: Literal["free_ocr", "markdown", "grounding", "parse_figure", "detailed"] = Form("markdown"),
    custom_prompt: Optional[str] = Form(None)
):
    """
    Process an image and extract text using OCR
    
    Args:
        file: Image to process (JPG, PNG, PDF, WEBP)
        mode: Predefined processing mode
        custom_prompt: Custom prompt (optional, overrides mode)
    
    Returns:
        JSON with extracted text and metadata
    """
    
    # Check if model is loaded
    if not model_loaded:
        try:
            load_model()
        except Exception as e:
            raise HTTPException(
                status_code=503,
                detail=f"Model not available: {str(e)}"
            )
    
    # Validate file
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"File type not allowed. Use: {settings.ALLOWED_EXTENSIONS}"
        )
    
    # Generar nombres únicos
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = f"{timestamp}_{file.filename}"
    upload_path = os.path.join(settings.UPLOAD_DIR, unique_id)
    
    try:
        # Save uploaded file
        with open(upload_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Validate size
        file_size = os.path.getsize(upload_path)
        if file_size > settings.MAX_FILE_SIZE:
            os.remove(upload_path)
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum: {settings.MAX_FILE_SIZE / 1024 / 1024}MB"
            )
        
        # Validate that it's a valid image
        try:
            img = Image.open(upload_path)
            img_size = img.size
            img.close()
        except Exception as e:
            os.remove(upload_path)
            raise HTTPException(
                status_code=400,
                detail=f"File is not a valid image: {str(e)}"
            )
        
        # Determine prompt
        prompt = custom_prompt if custom_prompt else PROMPTS.get(mode, PROMPTS["markdown"])
        
        # Create unique output directory
        output_dir = os.path.join(settings.OUTPUT_DIR, timestamp)
        os.makedirs(output_dir, exist_ok=True)
        
        # Process image with the model
        logger.info(f"Processing {unique_id} with mode '{mode}'")
        start_time = time.time()
        
        # Check if the model is loaded correctly
        if model is None or tokenizer is None:
            raise HTTPException(
                status_code=503,
                detail="Model not initialized properly"
            )
        
        result = model.infer(
            tokenizer,
            prompt=prompt,
            image_file=upload_path,
            output_path=output_dir,
            base_size=settings.BASE_SIZE,
            image_size=settings.IMAGE_SIZE,
            crop_mode=settings.CROP_MODE,
            save_results=True,
            test_compress=True
        )
        
        processing_time = time.time() - start_time
        logger.info(f"✓ Processed in {processing_time:.2f}s")
        
        # Leer resultado
        result_file = os.path.join(output_dir, "result.mmd")
        text_content = ""
        
        if os.path.exists(result_file):
            with open(result_file, 'r', encoding='utf-8') as f:
                text_content = f.read()
        
        # Respuesta
        response = {
            "success": True,
            "text": text_content or result,
            "mode": mode,
            "prompt": prompt,
            "processing_time": round(processing_time, 2),
            "image_size": img_size,
            "file_size": file_size,
            "timestamp": timestamp,
            "output_dir": output_dir,
            "metadata": {
                "filename": file.filename,
                "unique_id": unique_id,
                "device": settings.DEVICE
            }
        }
        
        return JSONResponse(content=response)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        # Clean up files in case of error
        if os.path.exists(upload_path):
            os.remove(upload_path)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image: {str(e)}"
        )


@app.get("/api/modes")
async def get_modes():
    """Returns the available OCR modes"""
    return {
        "modes": {
            "free_ocr": {
                "description": "OCR fast without structure",
                "speed": "⚡⚡⚡ Fast",
                "use_case": "General text extraction"
            },
            "markdown": {
                "description": "Converts document to Markdown with structure",
                "speed": "⚡⚡ Medium",
                "use_case": "Formatted documents"
            },
            "grounding": {
                "description": "OCR with bounding box coordinates",
                "speed": "⚡ Slow",
                "use_case": "Detailed analysis with locations"
            },
            "parse_figure": {
                "description": "Extracts information from figures and diagrams",
                "speed": "⚡⚡ Medium",
                "use_case": "Graphics, tables, diagrams"
            },
            "detailed": {
                "description": "Detailed description of the image",
                "speed": "⚡⚡⚡ Very fast",
                "use_case": "Visual content analysis"
            }
        }
    }


@app.delete("/api/cleanup")
async def cleanup_old_files(days: int = 7):
    """Clean up old files"""
    try:
        import time
        current_time = time.time()
        days_in_seconds = days * 24 * 60 * 60
        
        cleaned = {"uploads": 0, "outputs": 0}
        
        # Clean up uploads
        for file in Path(settings.UPLOAD_DIR).iterdir():
            if current_time - file.stat().st_mtime > days_in_seconds:
                file.unlink()
                cleaned["uploads"] += 1
        
        # Clean up outputs
        for folder in Path(settings.OUTPUT_DIR).iterdir():
            if folder.is_dir() and current_time - folder.stat().st_mtime > days_in_seconds:
                shutil.rmtree(folder)
                cleaned["outputs"] += 1
        
        return {
            "success": True,
            "cleaned": cleaned,
            "days": days
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

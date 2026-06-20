import pytest
from fastapi.testclient import TestClient
import io
from PIL import Image

# We need to import the app to use TestClient
# Let's see if we can import api.main
try:
    from api.main import app
    client = TestClient(app)
except ImportError:
    client = None

def create_dummy_image():
    # Create a simple white image with some text (optional, just white for now to test connection)
    img = Image.new('RGB', (100, 30), color = (255, 255, 255))
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    return img_byte_arr

def test_paddleocr_connection():
    assert client is not None, "Failed to import FastAPI app"
    
    img_bytes = create_dummy_image()
    
    response = client.post(
        "/v1/claims/ocr",
        data={"engine": "paddleocr", "doc_type": "DEATH_CERTIFICATE"},
        files={"file": ("dummy.png", img_bytes, "image/png")}
    )
    
    # We might expect 422 if no text is found, but 500 means server error (like missing library)
    assert response.status_code in [200, 422], f"PaddleOCR failed: {response.text}"

def test_tesseract_connection():
    assert client is not None, "Failed to import FastAPI app"
    
    img_bytes = create_dummy_image()
    
    response = client.post(
        "/v1/claims/ocr",
        data={"engine": "tesseract", "doc_type": "DEATH_CERTIFICATE"},
        files={"file": ("dummy.png", img_bytes, "image/png")}
    )
    
    assert response.status_code in [200, 422], f"Tesseract failed: {response.text}"

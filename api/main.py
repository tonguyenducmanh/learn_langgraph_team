from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import os
from .image_tool import generate

app = FastAPI()

# Create a directory for static files if it doesn't exist
os.makedirs("api/static", exist_ok=True)

# Mount the static directory to serve images
app.mount("/static", StaticFiles(directory="api/static"), name="static")

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/generate-image/")
async def generate_image(query: str):
    file_name = generate(query, "9:16")
    if file_name:
        # Assuming the server is running at http://127.0.0.1:8000
        image_url = f"http://127.0.0.1:8000/static/{file_name}"
        return JSONResponse(content={"image_url": image_url})
    else:
        raise HTTPException(status_code=500, detail="Image generation failed")

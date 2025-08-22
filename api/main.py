from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List
import os
import uuid
import json
from .image_tool import generate

class Page(BaseModel):
    image_url: str
    content: str

class Book(BaseModel):
    title: str
    author: str
    pages: List[Page]

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

@app.post("/generate-story/")
async def generate_story(book: Book):
    try:
        # Read the template file
        with open("api/template.md", "r") as f:
            template_content = f.read()

        # Replace placeholders
        content = template_content.replace("###bool_title###", book.title)
        content = content.replace("###author###", book.author)
        
        # Convert pages list to a JSON string for embedding in the script
        pages_json = json.dumps([page.dict() for page in book.pages])
        content = content.replace("###content_page###", pages_json)

        # Generate a unique filename
        file_name = f"story_{uuid.uuid4()}.html"
        file_path = os.path.join("api/static", file_name)

        # Write the new HTML file
        with open(file_path, "w") as f:
            f.write(content)

        # Return the URL of the generated file
        # Assuming the server is running at http://127.0.0.1:8000
        story_url = f"http://127.0.0.1:8000/static/{file_name}"
        return JSONResponse(content={"story_url": story_url})

    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="Template file not found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


from dotenv import load_dotenv
from google import genai
import os
from langchain_core.tools import tool
import uuid

load_dotenv()

def generate(query: str, raiton: str = "1:1") -> str | None:
    """
    Generate an image using the Gemini API.
    Args:
        query: The prompt for the image generation.
        raiton: The aspect ratio of the image. ["1:1","9:16","16:9","4:3","3:4"]  Default is "1:1".
    """
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

    try:
        result = client.models.generate_images(
            model="models/imagen-4.0-generate-001",
            prompt=query,
            config={
                "number_of_images": 1,
                "output_mime_type": "image/jpeg",
                "aspect_ratio": raiton,
                "image_size": "1K",
            },
        )

        if not result.generated_images:
            print("No images generated.")
            return None

        generated_image = result.generated_images[0]
        if generated_image.image:
            file_name = f"generated_image_{str(uuid.uuid4())}.jpg"
            file_path = os.path.join("api/static", file_name)
            generated_image.image.save(file_path)
            return file_name

    except Exception as e:
        print(f"An error occurred during image generation: {e}")

    return None

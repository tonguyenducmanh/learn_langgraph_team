
from dotenv import load_dotenv
from google import genai
import os
from langchain_core.tools import tool
import uuid

load_dotenv()

def generate(query: str) -> None:
    '''
        Generate an image using the Gemini API.
        Args:
            query: The prompt for the image generation.
            raiton: The aspect ratio of the image. ["1:1","9:16","16:9","4:3","3:4"]  Default is "1:1".
    '''
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

    result = client.models.generate_images(
        model="models/imagen-4.0-generate-001",
        prompt="""ảnh cố tích cô bé quảng khăn đỏ""",
        config=dict(
            number_of_images=1,
            output_mime_type="image/jpeg",
            aspect_ratio="9:16",
            image_size="1K",
        ),
    )

    if not result.generated_images:
        print("No images generated.")
        return

    if len(result.generated_images) != 1:
        print("Number of images generated does not match the requested number.")

    for n, generated_image in enumerate(result.generated_images):
        file_name = f"generated_image_{str(uuid.uuid4())}.jpg"
        generated_image.image.save(file_name)
        return file_name

generate.invoke({"query": "ảnh cô bé quảng khăn đỏ", "raiton": "1:1"})
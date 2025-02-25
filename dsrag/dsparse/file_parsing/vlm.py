import PIL.Image
import os
import io
from ..utils.imports import genai, vertexai

def make_llm_call_gemini(image_path: str, system_message: str, model: str = "gemini-1.5-pro-002", response_schema: dict = None, max_tokens: int = 4000) -> str:
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    generation_config = {
        "temperature": 0,
        "response_mime_type": "application/json",
        "max_output_tokens": max_tokens
    }
    if response_schema is not None:
        generation_config["response_schema"] = response_schema

    model = genai.GenerativeModel(model)
    try:
        image = PIL.Image.open(image_path)
        # Compress image if needed
        compressed_image = compress_image(image)
        response = model.generate_content(
            [
                compressed_image,
                system_message
            ],
            generation_config=generation_config
        )
        return response.text
    finally:
        # Ensure image is closed even if an error occurs
        if 'image' in locals():
            image.close()

def make_llm_call_vertex(image_path: str, system_message: str, model: str, project_id: str, location: str, response_schema: dict = None, max_tokens: int = 4000) -> str:
    """
    This function calls the Vertex AI Gemini API (not to be confused with the Gemini API) with an image and a system message and returns the response text.
    """
    vertexai.init(project=project_id, location=location)
    model = vertexai.generative_models.GenerativeModel(model)
    
    if response_schema is not None:
        generation_config = vertexai.generative_models.GenerationConfig(temperature=0.0, max_output_tokens=max_tokens, response_mime_type="application/json", response_schema=response_schema)
    else:
        generation_config = vertexai.generative_models.GenerationConfig(temperature=0.0, max_output_tokens=max_tokens)
    
    response = model.generate_content(
        [
            vertexai.generative_models.Part.from_image(vertexai.generative_models.Image.load_from_file(image_path)),
            system_message,
        ],
        generation_config=generation_config,
    )
    return response.text

def compress_image(image: PIL.Image.Image, max_size_bytes: int = 1097152, quality: int = 85) -> tuple[PIL.Image.Image, int]:
    """
    Compress image if it exceeds file size while maintaining aspect ratio.
    
    Args:
        image: PIL Image object
        max_size_bytes: Maximum file size in bytes (default ~1MB)
        quality: Initial JPEG quality (0-100)
    
    Returns:
        Tuple of (compressed PIL Image object, final quality used)
    """
    output = io.BytesIO()
    
    # Initial compression
    image.save(output, format='JPEG', quality=quality)
    
    # Reduce quality if file is too large
    while output.tell() > max_size_bytes and quality > 10:
        output = io.BytesIO()
        quality -= 5
        image.save(output, format='JPEG', quality=quality)
    
    # If reducing quality didn't work, reduce dimensions
    if output.tell() > max_size_bytes:
        while output.tell() > max_size_bytes:
            width, height = image.size
            image = image.resize((int(width*0.9), int(height*0.9)), PIL.Image.Resampling.LANCZOS)
            output = io.BytesIO()
            image.save(output, format='JPEG', quality=quality)
    
    # Convert back to PIL Image
    output.seek(0)
    compressed_image = PIL.Image.open(output)
    compressed_image.load()  # This is important to ensure the BytesIO can be closed
    return compressed_image
import PIL.Image
import io
from .vlm_clients import GeminiVLM, VertexAIVLM

def make_llm_call_gemini(image_path: str, system_message: str, model: str = "gemini-2.0-flash", response_schema: dict = None, max_tokens: int = 4000, temperature: float = 0.5) -> str:
    """
    Backward-compatible free function that delegates to GeminiVLM.

    Signature and behavior are preserved for compatibility.
    """
    client = GeminiVLM(model=model)
    return client.make_llm_call(
        image_path=image_path,
        system_message=system_message,
        response_schema=response_schema,
        max_tokens=max_tokens,
        temperature=temperature,
    )

def make_llm_call_vertex(image_path: str, system_message: str, model: str, project_id: str, location: str, response_schema: dict = None, max_tokens: int = 4000, temperature: float = 0.5) -> str:
    """
    Backward-compatible free function that delegates to VertexAIVLM.

    Signature and behavior are preserved for compatibility.
    """
    client = VertexAIVLM(model=model, project_id=project_id, location=location)
    return client.make_llm_call(
        image_path=image_path,
        system_message=system_message,
        response_schema=response_schema,
        max_tokens=max_tokens,
        temperature=temperature,
    )

def compress_image(image: PIL.Image.Image, max_size_bytes: int = 1097152, quality: int = 95) -> tuple[bytes, int]:
    """
    Compress image if it exceeds file size while maintaining aspect ratio.
    
    Args:
        image: PIL Image object
        max_size_bytes: Maximum file size in bytes (default ~1MB)
        quality: Initial JPEG quality (0-100)
    
    Returns:
        Tuple of (compressed image bytes, final quality used)
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
    
    # Return the bytes directly
    output.seek(0)
    return output.getvalue(), quality
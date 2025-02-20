import os
import instructor
from openai import OpenAI
from anthropic import Anthropic
import google.generativeai as genai
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import warnings
from dsrag.utils.model_names import (
    OPENAI_MODEL_NAMES,
    ANTHROPIC_MODEL_NAMES,
    GEMINI_MODEL_NAMES
)

def get_response(
    messages: Optional[List[Dict]] = None,
    prompt: Optional[str] = None,
    model_name: str = "claude-3-5-sonnet-20241022",
    response_model: Optional[BaseModel] = None,
    temperature: float = 0.0,
    max_tokens: int = 4000
) -> Any:
    """
    Unified LLM response handler supporting:
    - Text and multimodal inputs
    - Structured output (via Instructor)
    - Multiple providers (OpenAI, Anthropic, Gemini)
    
    Args:
        messages: List of message dicts (preferred if both messages and prompt provided)
            Format: [
                {
                    "role": "user"|"assistant"|"system",
                    "content": [
                        text: str | 
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg"|"image/png"|etc,
                                "data": "base64-encoded-string"
                            }
                        }
                    ]
                }
            ]
        prompt: Single prompt string (converted to messages if used)
        model_name: Model identifier string
        response_model: Pydantic model for structured output
        temperature: Generation temperature (0.0-1.0)
        max_tokens: Maximum tokens to generate
        
    Returns:
        str if no response_model provided, otherwise instance of response_model
        
    Example:
        >>> messages = [{
        >>>     "role": "user",
        >>>     "content": [
        >>>         "Describe this image",
        >>>         {
        >>>             "type": "image",
        >>>             "source": {
        >>>                 "type": "base64",
        >>>                 "media_type": "image/jpeg",
        >>>                 "data": "/9j/4AAQSkZJRgABAQ..."
        >>>             }
        >>>         }
        >>>     ]
        >>> }]
        >>> response = get_response(messages=messages, model_name="gpt-4o")
        
    Note: The function automatically handles provider-specific formatting for:
        - OpenAI: Converts images to base64 URLs
        - Anthropic: Uses native image format
        - Gemini: Converts to inline_data format
    """
    # Validate input
    if not messages and not prompt:
        raise ValueError("Either messages or prompt must be provided")
    if messages and prompt:
        warnings.warn("Both messages and prompt provided - using messages")
    
    # Convert prompt to messages if needed
    final_messages = messages or [{"role": "user", "content": prompt}]
    
    # Determine processing mode
    if response_model:
        return _handle_instructor_mode(final_messages, model_name, response_model, temperature, max_tokens)
    return _handle_standard_mode(final_messages, model_name, temperature, max_tokens)

def _handle_instructor_mode(messages: List[Dict], model_name: str, response_model: BaseModel, temperature: float, max_tokens: int) -> BaseModel:
    """Handle structured output using Instructor"""
    if model_name in ANTHROPIC_MODEL_NAMES:
        return _handle_anthropic_instructor(messages, model_name, response_model, temperature, max_tokens)
    if model_name in OPENAI_MODEL_NAMES:
        return _handle_openai_instructor(messages, model_name, response_model, temperature, max_tokens)
    if model_name in GEMINI_MODEL_NAMES:
        return _handle_gemini_instructor(messages, model_name, response_model, temperature, max_tokens)
    raise ValueError(f"Unsupported model for instructor: {model_name}")

def _handle_standard_mode(messages: List[Dict], model_name: str, temperature: float, max_tokens: int) -> str:
    """Handle standard text output"""
    if model_name in ANTHROPIC_MODEL_NAMES:
        return _handle_anthropic_standard(messages, model_name, temperature, max_tokens)
    if model_name in OPENAI_MODEL_NAMES:
        return _handle_openai_standard(messages, model_name, temperature, max_tokens)
    if model_name in GEMINI_MODEL_NAMES:
        return _handle_gemini_standard(messages, model_name, temperature, max_tokens)
    raise ValueError(f"Unsupported model: {model_name}")

# OpenAI Handlers
def _handle_openai_instructor(messages, model_name, response_model, temperature, max_tokens):
    client = instructor.from_openai(OpenAI(api_key=os.environ["OPENAI_API_KEY"]))
    formatted = _format_openai_messages(messages)
    return client.chat.completions.create(
        model=model_name,
        messages=formatted,
        response_model=response_model,
        temperature=temperature,
        max_tokens=max_tokens
    )

def _handle_openai_standard(messages, model_name, temperature, max_tokens):
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    formatted = _format_openai_messages(messages)
    response = client.chat.completions.create(
        model=model_name,
        messages=formatted,
        temperature=temperature,
        max_tokens=max_tokens
    )
    return response.choices[0].message.content

def _format_openai_messages(messages):
    """Handle OpenAI's message format with base64 images only"""
    formatted = []
    for msg in messages:
        content = []
        for part in msg["content"] if isinstance(msg["content"], list) else [msg["content"]]:
            if isinstance(part, dict) and part.get("type") == "image":
                # Now only handles base64
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{part['source']['media_type']};base64,{part['source']['data']}"}
                })
            else:
                content.append({"type": "text", "text": str(part)})
        formatted.append({"role": msg["role"], "content": content})
    return formatted

# Anthropic Handlers (with multimodal support)
def _handle_anthropic_instructor(messages, model_name, response_model, temperature, max_tokens):
    client = instructor.from_anthropic(Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"]))
    
    # Extract system message if present
    system = None
    filtered_messages = []
    for msg in messages:
        if msg["role"] == "system":
            system = msg["content"]
        else:
            filtered_messages.append(msg)
    
    formatted = _format_anthropic_messages(filtered_messages)
    
    # Only include system parameter if we have a system message
    kwargs = {
        "model": model_name,
        "messages": formatted,
        "response_model": response_model,
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    if system is not None:
        kwargs["system"] = system
    
    return client.messages.create(**kwargs)

def _handle_anthropic_standard(messages, model_name, temperature, max_tokens):
    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    
    # Extract system message if present
    system = None
    filtered_messages = []
    for msg in messages:
        if msg["role"] == "system":
            system = msg["content"]
        else:
            filtered_messages.append(msg)
    
    formatted = _format_anthropic_messages(filtered_messages)
    
    # Only include system parameter if we have a system message
    kwargs = {
        "model": model_name,
        "messages": formatted,
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    if system is not None:
        kwargs["system"] = system
        
    response = client.messages.create(**kwargs)
    return response.content[0].text

def _format_anthropic_messages(messages):
    """Handle Anthropic's message format including images with validation"""

    ANTHROPIC_SUPPORTED_IMAGE_FORMATS = ["jpeg", "png", "gif", "webp"]

    # Magic numbers for different image formats in base64
    IMAGE_HEADERS = {
        'image/jpeg': '/9j/',  # JPEG - only check for the consistent part
        'image/png': 'iVBORw0KGgo',  # PNG
        'image/gif': 'R0lGOD',  # GIF
        'image/webp': 'UklGR'  # WebP
    }
    
    formatted = []
    for msg in messages:
        content = []
        for part in msg["content"] if isinstance(msg["content"], list) else [msg["content"]]:
            if isinstance(part, dict) and part.get("type") == "image":
                # Validate image structure according to Anthropic requirements
                if "source" not in part:
                    raise ValueError("Image content must contain 'source' field")
                
                source = part["source"]
                if source.get("type") != "base64":
                    raise ValueError("Anthropic only supports base64 image encoding")
                
                media_type = source.get("media_type", "")
                if not media_type:
                    raise ValueError("Media type must be specified for image")
                
                # Extract format from media_type (e.g., 'image/jpeg' -> 'jpeg')
                image_format = media_type.split('/')[-1]
                if image_format not in ANTHROPIC_SUPPORTED_IMAGE_FORMATS:
                    raise ValueError(
                        f"Unsupported image format: {image_format}. "
                        f"Supported formats are: {', '.join(ANTHROPIC_SUPPORTED_IMAGE_FORMATS)}"
                    )
                
                # Check for data URI prefix
                data = source.get("data", "")
                if data.startswith(('data:', 'http')):
                    raise ValueError("Image data must be raw base64 without URI prefix")
                
                # Validate image data starts with correct header
                expected_header = IMAGE_HEADERS.get(media_type)
                if expected_header and not data.startswith(expected_header):
                    actual_header = data[:20]  # Show first 20 chars of the data
                    raise ValueError(
                        f"Invalid base64 data for {media_type}. "
                        f"Expected header starting with '{expected_header}', "
                        f"but got '{actual_header}...'"
                    )
                
                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": data
                    }
                })
            else:
                content.append({"type": "text", "text": str(part)})
        formatted.append({"role": msg["role"], "content": content})
    return formatted

# Gemini Handlers
def _handle_gemini_instructor(messages, model_name, response_model, temperature, max_tokens):
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    client = instructor.from_gemini(
        client=genai.GenerativeModel(model_name=f"models/{model_name}"),
        mode=instructor.Mode.GEMINI_JSON
    )
    formatted = _format_gemini_messages(messages)
    return client.messages.create(
        messages=formatted,
        response_model=response_model,
        generation_config={"temperature": temperature, "max_output_tokens": max_tokens}
    )

def _handle_gemini_standard(messages, model_name, temperature, max_tokens):
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    model = genai.GenerativeModel(model_name=f"models/{model_name}")
    formatted = _format_gemini_messages(messages)
    response = model.generate_content(formatted)
    return response.text

def _format_gemini_messages(messages):
    """Convert to Gemini's message format with image support and validation"""
    formatted = []
    for msg in messages:
        parts = []
        for part in msg["content"] if isinstance(msg["content"], list) else [msg["content"]]:
            if isinstance(part, dict) and part.get("type") == "image":
                # Validate Gemini image requirements
                if "source" not in part:
                    raise ValueError("Image content must contain 'source' field")
                
                # Check MIME type compliance
                valid_mime_types = {
                    "image/png", "image/jpeg", 
                    "image/webp", "image/heic", "image/heif"
                }
                if part["source"]["media_type"] not in valid_mime_types:
                    raise ValueError(f"Invalid MIME type for Gemini: {part['source']['media_type']}")
                
                # Verify base64 data format
                if part["source"].get("type") == "base64":
                    if part["source"]["data"].startswith("http"):
                        raise ValueError("Gemini requires raw base64 without URI prefix")
                
                parts.append({
                    "inline_data": {
                        "mime_type": part["source"]["media_type"],
                        "data": part["source"]["data"]
                    }
                })
            else:
                parts.append({"text": str(part)})
        formatted.append({"role": msg["role"], "parts": parts})
    return formatted

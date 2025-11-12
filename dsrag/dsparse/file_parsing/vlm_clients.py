from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type
import os

from ..utils.imports import genai_new, vertexai  # lazy loaders


class VLM(ABC):
    """Abstract base class for Visual Language Model clients.

    Subclasses should implement the make_llm_call method to perform the actual
    provider-specific API call and return the response text.

    Subclass registration
    ---------------------
    Each subclass is automatically registered by class name using
    __init_subclass__. This allows simple construction from a configuration
    dictionary via from_dict.
    """

    # Registry of subclass name -> subclass type
    _subclasses: Dict[str, Type["VLM"]] = {}

    def __init_subclass__(cls, **kwargs):  # type: ignore[override]
        super().__init_subclass__(**kwargs)
        # Register subclass by its class name for from_dict factory construction
        VLM._subclasses[cls.__name__] = cls

    def to_dict(self) -> Dict[str, Any]:
        """Serialize this VLM instance to a dictionary.

        Returns a dict containing the subclass name and public fields
        required to reconstruct the instance with from_dict.
        """
        # Default implementation serializes __dict__ (public fields) plus subclass name
        data = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
        data["subclass_name"] = self.__class__.__name__
        return data

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "VLM":
        """Construct a VLM instance from a serialized config dictionary.

        The config must contain a "subclass_name" key identifying the
        registered subclass. Remaining keys are forwarded to the subclass
        constructor as keyword arguments.
        """
        subclass_name = config.get("subclass_name")
        if not subclass_name:
            raise ValueError("config must include 'subclass_name'")
        subclass = cls._subclasses.get(subclass_name)
        if subclass is None:
            raise ValueError(f"Unknown VLM subclass: {subclass_name}")
        kwargs = {k: v for k, v in config.items() if k != "subclass_name"}
        return subclass(**kwargs)  # type: ignore[arg-type]

    @abstractmethod
    def make_llm_call(
        self,
        image_path: str,
        system_message: str,
        response_schema: Optional[Dict[str, Any]] = None,
        max_tokens: int = 4000,
        temperature: float = 0.5,
    ) -> str:
        """Perform a VLM call and return the raw response text."""
        raise NotImplementedError


class GeminiVLM(VLM):
    """VLM client for Google Gemini via the new google-genai SDK.

    Fields
    ------
    - model: Gemini model name, default "gemini-2.0-flash".

    Behavior
    --------
    - Uses dsrag.dsparse.utils.imports.genai_new (lazy) to construct a
      Client(api_key=os.environ["GEMINI_API_KEY"]).
    - Builds GenerateContentConfig with response_mime_type="application/json",
      including optional response_schema when provided.
    - For models starting with "gemini-2.5", sets a ThinkingConfig with
      thinking_budget=0 to disable thinking as per existing behavior.
    - Compresses images using compress_image from vlm.py.
    """

    def __init__(self, model: str = "gemini-2.0-flash") -> None:
        self.model = model

    def make_llm_call(
        self,
        image_path: str,
        system_message: str,
        response_schema: Optional[Dict[str, Any]] = None,
        max_tokens: int = 4000,
        temperature: float = 0.5,
    ) -> str:
        # Local import to avoid circular dependency at module import time
        from .vlm import compress_image  # noqa: WPS433 (allow local import)
        import PIL.Image  # used only when this method is executed
        import io

        # Create client using lazy loader with explicit API key check
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set; required for GeminiVLM")
        client = genai_new.Client(api_key=api_key)  # type: ignore[attr-defined]

        # Base generation config
        config = genai_new.types.GenerateContentConfig(  # type: ignore[attr-defined]
            temperature=temperature,
            max_output_tokens=max_tokens,
            response_mime_type="application/json",
        )
        if response_schema is not None:
            config.response_schema = response_schema

        image = None
        try:
            # Open and compress the image
            image = PIL.Image.open(image_path)
            compressed_image_bytes, _ = compress_image(image)

            # Close the original image (safe to call multiple times)
            if image:
                image.close()

            # Create content parts using bytes
            image_part = genai_new.types.Part.from_bytes(  # type: ignore[attr-defined]
                data=compressed_image_bytes,
                mime_type="image/jpeg",
            )
            content_parts = [image_part, system_message]

            # For Gemini 2.5 models, disable thinking
            if self.model.startswith("gemini-2.5"):
                gemini25_config = genai_new.types.GenerateContentConfig(  # type: ignore[attr-defined]
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                    response_mime_type="application/json",
                    thinking_config=genai_new.types.ThinkingConfig(thinking_budget=0),  # type: ignore[attr-defined]
                )
                if response_schema is not None:
                    gemini25_config.response_schema = response_schema

                response = client.models.generate_content(  # type: ignore[attr-defined]
                    model=self.model,
                    contents=content_parts,
                    config=gemini25_config,
                )
            else:
                response = client.models.generate_content(  # type: ignore[attr-defined]
                    model=self.model,
                    contents=content_parts,
                    config=config,
                )
            return response.text
        finally:
            if image is not None:
                try:
                    image.close()
                except Exception:
                    pass


class VertexAIVLM(VLM):
    """VLM client for Vertex AI Gemini API.

    Fields
    ------
    - model: Vertex AI model name (required)
    - project_id: GCP project id (required)
    - location: GCP location/region (required)

    Behavior
    --------
    - Uses dsrag.dsparse.utils.imports.vertexai (lazy) to initialize and call
      the vertexai.generative_models API.
    - Builds GenerationConfig with optional response_schema and returns
      response.text
    """

    def __init__(self, model: str, project_id: str, location: str) -> None:
        self.model = model
        self.project_id = project_id
        self.location = location

    def make_llm_call(
        self,
        image_path: str,
        system_message: str,
        response_schema: Optional[Dict[str, Any]] = None,
        max_tokens: int = 4000,
        temperature: float = 0.5,
    ) -> str:
        # Initialize Vertex AI client
        vertexai.init(project=self.project_id, location=self.location)  # type: ignore[attr-defined]
        model = vertexai.generative_models.GenerativeModel(self.model)  # type: ignore[attr-defined]

        if response_schema is not None:
            generation_config = vertexai.generative_models.GenerationConfig(  # type: ignore[attr-defined]
                temperature=temperature,
                max_output_tokens=max_tokens,
                response_mime_type="application/json",
                response_schema=response_schema,
            )
        else:
            generation_config = vertexai.generative_models.GenerationConfig(  # type: ignore[attr-defined]
                temperature=temperature,
                max_output_tokens=max_tokens,
            )

        response = model.generate_content(
            [
                vertexai.generative_models.Part.from_image(  # type: ignore[attr-defined]
                    vertexai.generative_models.Image.load_from_file(image_path)  # type: ignore[attr-defined]
                ),
                system_message,
            ],
            generation_config=generation_config,
        )
        return response.text

from .vlm import make_llm_call_gemini, make_llm_call_vertex
from ..models.types import ElementType, Element, VLMConfig
from .file_system import FileSystem
from .vlm_clients import VLM, GeminiVLM, VertexAIVLM
from typing import Optional
from .element_types import (
    get_visual_elements_as_str, 
    get_non_visual_elements_as_str, 
    get_element_description_block, 
    default_element_types,
    get_num_visual_elements,
    get_num_non_visual_elements,
)
from pdf2image import convert_from_path
import json
import time
import logging
import concurrent.futures
from pypdf import PdfReader

# Get the dsparse logger
logger = logging.getLogger("dsrag.dsparse.vlm_file_parsing")

"""
pip install pdf2image
brew install poppler
"""

SYSTEM_MESSAGE = """
You are a PDF -> MD file parser. Your task is to analyze the provided PDF page (provided as an image) and return a structured JSON response containing all of the elements on the page. Each element must be represented using Markdown formatting.

There are two categories of elements you need to identify: text elements and visual elements. Text elements are those that can be accurately represented using plain text. Visual elements are those that need to be represented as images to fully capture their content. For text elements, you must provide the exact text content. For visual elements, you must provide a detailed description of the content.

There are {num_visual_elements} types of visual elements: {visual_elements_as_str}.
There are {num_non_visual_elements} types of text elements: {non_visual_elements_as_str}.

Every element on the page should be classified as one of these types. There should be no overlap between elements. You should use the smallest number of elements possible while still accurately representing and categorizing the content on the page. For example, if the page contains a couple paragraphs of text, followed by a large figure, followed by a few more paragraphs of text, you should use three elements: NarrativeText, Figure, and NarrativeText. With that said, you should never combine two different types of elements into a single element.

Here are detailed descriptions of the element types you can use:
{element_description_block}

For visual elements ({visual_elements_as_str}), you must provide a detailed description of the element in the "content" field. Do not just transcribe the actual text contained in the element. For textual elements ({non_visual_elements_as_str}), you must provide the exact text content of the element.

If there is any sensitive information in the document, YOU MUST IGNORE IT. This could be a SSN, bank information, etc. Names and DOBs are not sensitive information.

Output format
- Your output should be an ordered (from top to bottom) list of elements on the page, where each element is a dictionary with the following keys:
    - type: str - the type of the element
    - content: str - the content of the element. For visual elements, this should be a detailed description of the visual content, rather than a transcription of the actual text contained in the element. You can use Markdown formatting for text content.

Complex and multi-part figures or images should be represented as a single element. For example, if a figure consists of a main chart and a smaller inset chart, these should be described together in a single Figure element. If there are two separate graphs side by side, these should be represented as a single Figure element with a bounding box that encompasses both graphs. DO NOT create separate elements for each part of a complex figure or image.
"""

response_schema = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "type": {
                "type": "string",
            },
            "content": {
                "type": "string",
            },
        },
        "required": ["type", "content"],
    },
}

def get_page_count(file_path: str, kb_id: str = "", doc_id: str = ""):
    # Create base logging context with identifiers
    base_extra = {}
    if kb_id:
        base_extra["kb_id"] = kb_id
    if doc_id:
        base_extra["doc_id"] = doc_id
        
    try:
        with open(file_path, "rb") as pdf_file:
            pdf_reader = PdfReader(pdf_file)
            return len(pdf_reader.pages)
    except Exception as e:
        logger.error(f"Error getting page count: {e}", extra={
            **base_extra,
            "file_path": file_path
        })
        return None

def pdf_to_images(pdf_path: str, kb_id: str, doc_id: str, file_system: FileSystem, dpi=100, max_workers: int=2, max_pages: int=10) -> list[str]:
    """
    Convert a PDF to images and save them to a folder. Uses pdf2image (which relies on poppler).

    Inputs:
    - pdf_path: str - the path to the PDF file.
    - page_images_path: str - the path to the folder where the images will be saved.
    - thread_count: int - the number of threads to use for converting the PDF to images.
    - max_workers: int - the number of workers to use for saving the images.

    Returns:
    - image_file_paths: list[str] - a list of the paths to the saved images.
    """
    
    # Create base logging context with identifiers
    base_extra = {"kb_id": kb_id, "doc_id": doc_id}
    
    if (max_pages < 1):
        logger.error("max_pages must be greater than 0", extra=base_extra)
        raise ValueError("max_pages must be greater than 0")
    
    # Create the folder
    file_system.create_directory(kb_id, doc_id)

    def save_single_image(args):
        i, image = args
        file_system.save_image(kb_id, doc_id, f'page_{i+1}.jpg', image)
        return f'/{kb_id}/{doc_id}/page_{i+1}.jpg'

    # Convert PDF to images in batches of max_pages
    page_count = get_page_count(pdf_path, kb_id, doc_id)
    all_image_paths = []
    
    for i in range(1, page_count + 1, max_pages):
        logger.debug(f"Converting pages {i} to {i + max_pages-1}", extra=base_extra)
        last_page = min(i + max_pages-1, page_count)
        images = convert_from_path(pdf_path, dpi=dpi, thread_count=max_workers, 
                                 first_page=i, last_page=last_page)
        
        # Save batch of images in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            batch_paths = list(executor.map(save_single_image, enumerate(images, start=i-1)))
            all_image_paths.extend(batch_paths)
        
        logger.debug(f"Converted pages {i} to {last_page}", extra=base_extra)

    logger.info(f"Converted total {len(all_image_paths)} pages to images", extra=base_extra)
    return all_image_paths

def parse_page(kb_id: str, doc_id: str, file_system: FileSystem, page_number: int, vlm_config: VLMConfig, element_types: list[ElementType], vlm_client: Optional[VLM] = None, vlm_fallback_client: Optional[VLM] = None) -> tuple[int, list[Element]]:
    """
    Given an image of a page, use LLM to extract the content of the page.
    This function includes retry logic and a fallback model mechanism.

    Inputs:
    - page_image_path: str, path to the image of the page
    - vlm_config: dict, configuration for the VLM
    - element_types: list[ElementType], list of element types that the VLM can output
    
    Outputs:
    - tuple[int, list[Element]]: A tuple containing the page number and a list of Elements.
    """
    base_extra = {"kb_id": kb_id, "doc_id": doc_id, "page_number": page_number}
    tries = 0
    max_retries = 10

    primary_provider = vlm_config.get("provider", "gemini")
    primary_model = vlm_config.get("model")
    if not primary_model:
        if primary_provider == "gemini":
            primary_model = "gemini-2.0-flash"
        else:
            raise ValueError("Non-default VLM provider specified without specifying a model")

    fallback_provider = vlm_config.get("fallback_provider")
    fallback_model = vlm_config.get("fallback_model")

    # Determine primary and fallback clients
    primary_client: Optional[VLM] = vlm_client
    fallback_client: Optional[VLM] = vlm_fallback_client

    if primary_client is None:
        if primary_provider == "gemini":
            primary_client = GeminiVLM(model=primary_model)
        elif primary_provider == "vertex_ai":
            project_id = vlm_config.get("project_id")
            location = vlm_config.get("location")
            if project_id and location:
                primary_client = VertexAIVLM(model=primary_model, project_id=project_id, location=location)
            else:
                primary_client = None  # fall back to legacy path if insufficient config
        # else: invalid provider handled later in loop

    if fallback_client is None and fallback_provider and fallback_model:
        if fallback_provider == "gemini":
            fallback_client = GeminiVLM(model=fallback_model)
        elif fallback_provider == "vertex_ai":
            project_id_fb = vlm_config.get("project_id")
            location_fb = vlm_config.get("location")
            if project_id_fb and location_fb:
                fallback_client = VertexAIVLM(model=fallback_model, project_id=project_id_fb, location=location_fb)

    while tries < max_retries:
        # Determine which model to use for this attempt
        current_vlm_config = vlm_config.copy()
        attempt_number = tries + 1

        # Alternate between primary and fallback model after the first few attempts
        if attempt_number > 3 and fallback_provider and fallback_model:
            if attempt_number % 2 != 0:  # Odd attempts (5, 7, 9) use fallback
                logger.info(f"Attempt {attempt_number}: Switching to fallback model {fallback_model}", extra=base_extra)
                current_vlm_config["provider"] = fallback_provider
                current_vlm_config["model"] = fallback_model
            else: # Even attempts (4, 6, 8, 10) use primary
                logger.info(f"Attempt {attempt_number}: Switching back to primary model {primary_model}", extra=base_extra)
                current_vlm_config["provider"] = primary_provider
                current_vlm_config["model"] = primary_model
        else: # Attempts 1-3 use primary, or all attempts if no fallback is configured
            current_vlm_config["provider"] = primary_provider
            current_vlm_config["model"] = primary_model

        # format system message
        system_message = SYSTEM_MESSAGE.format(
            num_visual_elements=get_num_visual_elements(element_types),
            num_non_visual_elements=get_num_non_visual_elements(element_types),
            visual_elements_as_str=get_visual_elements_as_str(element_types),
            non_visual_elements_as_str=get_non_visual_elements_as_str(element_types),
            element_description_block=get_element_description_block(element_types)
        )

        page_image_path = file_system.get_files(kb_id, doc_id, page_number, page_number)[0]
        
        llm_output = None
        rate_limited = False

        # Choose instance client if available based on attempt
        use_fallback = attempt_number > 3 and (fallback_provider and fallback_model)
        current_client = None
        if use_fallback and fallback_client is not None:
            current_client = fallback_client
            logger.debug("Using instance-based VLM fallback client", extra=base_extra)
        elif primary_client is not None:
            current_client = primary_client
            logger.debug("Using instance-based VLM primary client", extra=base_extra)

        if current_client is not None:
            try:
                temperature = current_vlm_config.get("temperature", 0.5)
                llm_output = current_client.make_llm_call(
                    image_path=page_image_path,
                    system_message=system_message,
                    response_schema=response_schema,
                    max_tokens=current_vlm_config.get("max_tokens", 4000),
                    temperature=temperature,
                )
            except Exception as e:
                if "429" in str(e):
                    logger.warning(f"Rate limit exceeded with instance VLM: {e}", extra=base_extra)
                    rate_limited = True
                else:
                    logger.error(f"Error with instance VLM: {e}", extra=base_extra)
                    llm_output = ""
        elif current_vlm_config["provider"] == "vertex_ai":
            try:
                temperature = current_vlm_config.get("temperature", 0.5)
                llm_output = make_llm_call_vertex(
                    image_path=page_image_path, 
                    system_message=system_message, 
                    model=current_vlm_config["model"], 
                    project_id=current_vlm_config["project_id"], 
                    location=current_vlm_config["location"],
                    response_schema=response_schema,
                    max_tokens=current_vlm_config.get("max_tokens", 4000),
                    temperature=temperature
                )
            except Exception as e:
                if "429" in str(e):
                    logger.warning(f"Rate limit exceeded with Vertex AI: {e}", extra=base_extra)
                    rate_limited = True
                else:
                    logger.error(f"Error with Vertex AI: {e}", extra=base_extra)
                    llm_output = "" # Force retry on other errors

        elif current_vlm_config["provider"] == "gemini":
            try:
                temperature = current_vlm_config.get("temperature", 0.5)
                llm_output = make_llm_call_gemini(
                    image_path=page_image_path, 
                    system_message=system_message, 
                    model=current_vlm_config["model"],
                    response_schema=response_schema,
                    max_tokens=current_vlm_config.get("max_tokens", 4000),
                    temperature=temperature
                )
            except Exception as e:
                if "429" in str(e):
                    logger.warning(f"Rate limit exceeded with Gemini: {e}", extra=base_extra)
                    rate_limited = True
                else:
                    logger.error(f"Error with Gemini: {e}", extra=base_extra)
                    llm_output = "" # Force retry on other errors
        else:
            raise ValueError(f"Invalid provider specified: {current_vlm_config['provider']}")

        # Handle rate limit errors
        if rate_limited:
            if tries == max_retries - 1:
                logger.error(f"Rate limit exceeded on final retry attempt {tries+1}/{max_retries}", extra={**base_extra, "retry_attempt": tries+1, "final_attempt": True})
            else:
                logger.warning(f"Rate limit exceeded. Sleeping for 10 seconds before retrying...", extra={**base_extra, "retry_attempt": tries+1})
            time.sleep(10)
            tries += 1
            continue

        # Handle JSON parsing and other errors
        page_content = []
        try:
            if llm_output:
                page_content = json.loads(llm_output)
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON on attempt {tries+1}: {e}", extra=base_extra)
            logger.debug("Full problematic model output:", extra={"full_model_output": llm_output, **base_extra})
        
        if page_content:
            for element in page_content:
                element["page_number"] = page_number
            return page_number, page_content
        else: # Empty content or JSON error, so retry
            if tries == max_retries - 1:
                logger.error(f"Empty or invalid content on final retry attempt {tries+1}/{max_retries}", extra={**base_extra, "retry_attempt": tries+1, "final_attempt": True})
            else:
                logger.warning(f"Empty or invalid content received. Retrying...", extra={**base_extra, "retry_attempt": tries+1})
            tries += 1
            continue

    # If all retries fail, return a minimal valid result
    logger.error(f"Failed to process page after {max_retries} attempts", extra=base_extra)
    return page_number, [{"type": "NarrativeText", "content": "Failed to process page after multiple attempts", "page_number": page_number}]


def parse_file(pdf_path: str, kb_id: str, doc_id: str, vlm_config: VLMConfig, file_system: FileSystem, vlm_client: Optional[VLM] = None, vlm_fallback_client: Optional[VLM] = None) -> list[Element]:
    """
    Given a PDF file, extract the content of each page using a VLM model.
    
    Inputs
    - pdf_path: str, path to the PDF file - can be an empty string if images_already_exist is True
    - kb_id: str, knowledge base ID
    - doc_id: str, document ID
    - vlm_config: dict, configuration for the VLM model. For Vertex this should include project_id and location.
    - file_system: FileSystem, object for interacting with the file system where the images are stored
    
    Outputs
    - all_page_content: list of Elements

    Saves
    - images of each page of the PDF (if images_already_exist is False)
    - JSON files of the content of each page
    """
    max_pages = vlm_config.get("max_pages", 10)
    max_workers = vlm_config.get("max_workers", 2)
    images_already_exist = vlm_config.get("images_already_exist", False)
    vlm_max_concurrent_requests = vlm_config.get("vlm_max_concurrent_requests", 5)
    dpi = vlm_config.get("dpi", 100)

    if images_already_exist:
        image_file_paths = file_system.get_all_jpg_files(kb_id, doc_id)
    else:
        image_file_paths = pdf_to_images(pdf_path, kb_id, doc_id, file_system, dpi=dpi, max_workers=max_workers, max_pages=max_pages)
    
    all_page_content_dict = {}

    element_types = vlm_config.get("element_types", default_element_types)
    if len(element_types) == 0:
        element_types = default_element_types

    base_extra = {"kb_id": kb_id, "doc_id": doc_id}
    logger.debug(f"Starting VLM page processing with up to {vlm_max_concurrent_requests} concurrent requests", extra=base_extra)
    
    # Use ThreadPoolExecutor to process pages in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=vlm_max_concurrent_requests) as executor:
        futures = {
            executor.submit(
                parse_page, 
                kb_id, 
                doc_id, 
                file_system, 
                i + 1, 
                vlm_config, 
                element_types,
                vlm_client,
                vlm_fallback_client,
            ): i 
            for i in range(len(image_file_paths))
        }
        for future in concurrent.futures.as_completed(futures):
            # Add the page content to the dictionary, keyed on the page number
            page_number, page_content = future.result()
            all_page_content_dict[page_number] = page_content
            logger.debug(f"Processed page {page_number}", 
                         extra={**base_extra, "page_number": page_number})

    all_page_content = []
    for key in sorted(all_page_content_dict.keys()):
        all_page_content.extend(all_page_content_dict[key])

    # Save the extracted content to a JSON file
    file_system.save_json(kb_id, doc_id, 'elements.json', all_page_content)
    
    logger.info(f"Finished parsing file with {len(all_page_content)} elements from {len(all_page_content_dict)} pages", 
               extra={"kb_id": kb_id, "doc_id": doc_id})

    return all_page_content

def elements_to_markdown(elements: list[Element]) -> str:
    """
    Given a list of elements extracted from a PDF, convert them to a markdown string.
    
    Inputs
    - elements: list of dictionaries, each containing information about an element on a page
    
    Outputs
    - markdown_string: str, a markdown string representing the elements
    """
    markdown_string = ""
    for element in elements:
        markdown_string += f"{element['content']}\n\n"

    return markdown_string
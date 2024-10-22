from dsrag.dsparse.vlm import make_llm_call_gemini, make_llm_call_vertex
from dsrag.dsparse.types import ElementType, Element, VLMConfig
from dsrag.dsparse.element_types import (
    get_visual_elements_as_str, 
    get_non_visual_elements_as_str, 
    get_element_description_block, 
    default_element_types,
    get_num_visual_elements,
    get_num_non_visual_elements,
)
import os
from pdf2image import convert_from_path
import json
import time
import concurrent.futures

"""
pip install pdf2image
brew install poppler
"""

SYSTEM_MESSAGE = """
You are a PDF -> MD file parser. Your task is to analyze the provided PDF page (provided as an image) and return a structured JSON response containing all of the elements on the page. Each element must be represented using Markdown formatting.

There are two categories of elements you need to identify: text elements and visual elements. Text elements are those that can be accurately represented using plain text. Visual elements are those that need to be represented as images to fully capture their content. For text elements, you must provide the exact text content. For visual elements, you must provide a detailed description of the content.

There are {num_visual_elements} types of visual elements: {visual_elements_as_str}.
There are {num_non_visual_elements} types of text elements: {non_visual_elements_as_str}.

Every element on the page should be classified as one of these types. There should be no overlap between elements. You should use the smallest number of elements possible while still accurately representing the content on the page. For example, if the page contains a couple paragraphs of text, followed by a large figure, followed by a few more paragraphs of text, you should use three elements: NarrativeText, Figure, and NarrativeText.

Here are detailed descriptions of the element types you can use:
{element_description_block}

For visual elements ({visual_elements_as_str}), you must provide a detailed description of the element in the "content" field. Do not just transcribe the actual text contained in the element. For textual elements ({non_visual_elements_as_str}), you must provide the exact text content of the element.

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

def pdf_to_images(pdf_path: str, page_images_path: str, dpi=150) -> list[str]:
    """
    Convert a PDF to images and save them to a folder. Uses pdf2image (which relies on poppler).

    Inputs:
    - pdf_path: str - the path to the PDF file.
    - page_images_path: str - the path to the folder where the images will be saved.

    Returns:
    - image_file_paths: list[str] - a list of the paths to the saved images.
    """
    # Delete the folder if it already exists
    if os.path.exists(page_images_path):
        for file in os.listdir(page_images_path):
            os.remove(os.path.join(page_images_path, file))
        os.rmdir(page_images_path)

    # Create the folder
    os.makedirs(page_images_path, exist_ok=False)

    # Convert PDF to images
    images = convert_from_path(pdf_path, dpi=dpi)

    # Save each image
    image_file_paths = []
    for i, image in enumerate(images):
        image_file_path = os.path.join(page_images_path, f'page_{i+1}.png')
        image.save(image_file_path, 'PNG')
        image_file_paths.append(image_file_path)

    print(f"Converted {len(images)} pages to images in {page_images_path}")
    return image_file_paths

def parse_page(page_image_path: str, vlm_config: VLMConfig, element_types: list[ElementType]) -> list[Element]:
    """
    Given an image of a page, use LLM to extract the content of the page.

    Inputs:
    - page_image_path: str, path to the image of the page
    - vlm_config: dict, configuration for the VLM
    - element_types: list[ElementType], list of element types that the VLM can output
    
    Outputs:
    - page_content: list of Elements
    """

    # format system message
    system_message = SYSTEM_MESSAGE.format(
        num_visual_elements=get_num_visual_elements(element_types),
        num_non_visual_elements=get_num_non_visual_elements(element_types),
        visual_elements_as_str=get_visual_elements_as_str(element_types),
        non_visual_elements_as_str=get_non_visual_elements_as_str(element_types),
        element_description_block=get_element_description_block(element_types)
    )

    print(f"System message: {system_message}")

    if vlm_config["provider"] == "vertex_ai":
        try:
            llm_output = make_llm_call_vertex(
                image_path=page_image_path, 
                system_message=system_message, 
                model=vlm_config["model"], 
                project_id=vlm_config["project_id"], 
                location=vlm_config["location"],
                response_schema=response_schema,
                max_tokens=4000
            )
        except Exception as e:
            if "429 Online prediction request quota exceeded" in str(e):
                print (f"Error in make_llm_call_gemini: {e}")
                return 429
    elif vlm_config["provider"] == "gemini":
        try:
            llm_output = make_llm_call_gemini(
                image_path=page_image_path, 
                system_message=system_message, 
                model=vlm_config["model"],
                response_schema=response_schema,
                max_tokens=4000
            )
        except Exception as e:
            if "429 Online prediction request quota exceeded" in str(e):
                print (f"Error in make_llm_call_gemini: {e}")
                return
            else:
                print (f"Error in make_llm_call_gemini: {e}")
    else:
        raise ValueError("Invalid provider specified in the VLM config. Only 'vertex_ai' is supported for now.")
    
    try:
        page_content = json.loads(llm_output)
    except json.JSONDecodeError:
        print(f"Error decoding JSON response from LLM for {page_image_path}")
        print(llm_output)
        page_content = []

    return page_content

def parse_file(pdf_path: str, save_path: str, vlm_config: VLMConfig, element_types: list[ElementType] = default_element_types) -> list[Element]:
    """
    Given a PDF file, extract the content of each page using a VLM model.
    
    Inputs
    - pdf_path: str, path to the PDF file
    - save_path: str, path to the base directory where everything is saved (i.e. {user_id}/{job_id})
    - vlm_config: dict, configuration for the VLM model. For Gemini this should include project_id and location.
    
    Outputs
    - all_page_content: list of Elements
    """
    page_images_path = f"{save_path}/page_images"
    image_file_paths = pdf_to_images(pdf_path, page_images_path)
    all_page_content_dict = {}

    def process_page(image_path, page_number):
        tries = 0
        while tries < 20:
            content = parse_page(page_image_path=image_path, vlm_config=vlm_config, element_types=element_types)
            if content == 429:
                print(f"Rate limit exceeded. Sleeping for 10 seconds before retrying...")
                time.sleep(10)
                tries += 1
                continue
            else:
                print ("Successfully processed page")
                return page_number, content

    # Use ThreadPoolExecutor to process pages in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_page, image_path, i + 1): image_path for i, image_path in enumerate(image_file_paths)}
        for future in concurrent.futures.as_completed(futures):
            page_content = future.result()
            # Add the page content to the dictionary, keyed on the page number
            page_number, page_content = future.result()
            all_page_content_dict[page_number] = page_content

    all_page_content = []
    for key in sorted(all_page_content_dict.keys()):
        all_page_content.extend(all_page_content_dict[key])

    # Save the extracted content to a JSON file
    output_file_path = f"{save_path}/elements.json"
    with open(output_file_path, "w") as f:
        json.dump(all_page_content, f, indent=2)

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
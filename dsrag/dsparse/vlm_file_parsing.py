from dsrag.dsparse.bounding_box_retry import get_improved_bounding_box
from dsrag.dsparse.vertex_ai import make_llm_call_gemini
#from dsrag.dsparse.types import Element

import os
from pdf2image import convert_from_path
from PIL import Image
import json
import base64
import time

"""
pip install pdf2image
brew install poppler
"""

SYSTEM_MESSAGE = """
You are a PDF parser. Your task is to analyze the provided PDF page (provided as an image) and return a structured JSON response containing all of the elements on the page.

Here are the element types you can use:
- NarrativeText
    - This is the main text content of the page, including paragraphs, lists, titles, and any other text content that is not part of a header, footer, figure, table, or image. Not all pages have narrative text, but most do.
- Figure
    - This covers charts, graphs, diagrams, etc. Associated titles, legends, axis titles, etc. should be considered to be part of the figure. Be sure your descriptions and bounding boxes fully capture these associated items, as they are essential for providing context to the figure. Not all pages have figures.
- Image
    - This is any visual content on the page that isn't a figure. Not all pages have images.
- Table
    - This is a table on the page. If the table can be represented accurately using Markdown, then it should be included as a Table element. If not, it should be included as an Image element to ensure accuracy.
- Header
    - This is the header of the page. You should never user more than one header element per page. Not all pages have a header.
- Footnote
    - This is a footnote on the page. Footnotes should always be included as a separate element from the main text content as they aren't part of the main linear reading flow of the page. Not all pages have footnotes.
- Footer
    - This is the footer of the page. You should never user more than one footer element per page. Not all pages have a footer, but when they do it is always the very last element on the page.

For Image and Figure elements ONLY, you must provide a detailed description of the image or figure. Do not transcribe the actual text contained in the image or figure. For all other element types, you must provide the text content.

Output format
- Your output should be an ordered (from top to bottom) list of elements on the page, where each element is a dictionary with the following keys:
    - type: str - the type of the element (e.g. "NarrativeText", "Figure", "Image", "Table", "Header", "Footnote", or "Footer")
    - content: str - the content of the element (ONLY include when type is "NarrativeText", "Table", "Header", "Footnote", or "Footer". For other element types, just use an empty string here). You can use Markdown formatting for text content. Always use Markdown for tables.
    - description: str (ONLY include when type is "Image" or "Figure". For other element types, just use an empty string here) - a detailed description of the image or figure.
    - bounding_box: list[int] (ONLY include when type is "Image" or "Figure". For other element types, just use an empty list here) - a bounding box around the image or figure, in the format [ymin, xmin, ymax, xmax].

Additional instructions
- Ignore background images or other images that don't convey any information.
- The element types described above are the only ones you are allowed to use.
- Be sure to include all page content in your response.
- Image and Figure elements MUST have accurate bounding boxes.
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
            "description": {
                "type": "string",
            },
            "bounding_box": {
                "type": "array",
                "items": {
                    "type": "number",
                },
            },
        },
        "required": ["type", "content", "description", "bounding_box"],
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

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def extract_image(page_image_path: str, bounding_box: list[int], extracted_image_path: str, padding: int = 100):
    """
    Given a page image and a bounding box, extract the image from the bounding boxes by cropping the page image.
    - Leave a bit of extra padding around the provided bounding box to ensure that the entire content is captured.

    Inputs:
    - page_image_file_path: str, path to the page image
    - bounding_box: list[int], bounding box in the format [ymin, xmin, ymax, xmax], where (xmin, ymin) is the top-left corner and (xmax, ymax) is the bottom-right corner and the top-left corner is (0, 0)
    - extracted_image_path: str, path where the extracted images will be saved - should include the image name and extension
    - padding: int, padding around the VLM-generated bounding boxes to include extra content, provided in 1000x1000 coordinate space

    Outputs:
    - Saves the extracted images to the output folder
    """
    ymin, xmin, ymax, xmax = bounding_box
    
    # Add some padding to the bounding box
    xmin = max(0, xmin - padding)
    ymin = max(0, ymin - padding)
    xmax = min(1000, xmax + padding)
    ymax = min(1000, ymax + padding)

    # Open the image
    with Image.open(page_image_path) as img:
        width, height = img.size
        print(f"Original image size: {width}x{height}")

        # Calculate actual pixel coordinates
        ymin_scaled, xmin_scaled, ymax_scaled, xmax_scaled = bounding_box
        actual_ymin = int(ymin_scaled / 1000 * height)
        actual_xmin = int(xmin_scaled / 1000 * width)
        actual_ymax = int(ymax_scaled / 1000 * height)
        actual_xmax = int(xmax_scaled / 1000 * width)

        # Ensure coordinates are within image bounds
        actual_ymin = max(0, actual_ymin)
        actual_xmin = max(0, actual_xmin)
        actual_ymax = min(height, actual_ymax)
        actual_xmax = min(width, actual_xmax)

        # Crop the image
        cropped_img = img.crop((actual_xmin, actual_ymin, actual_xmax, actual_ymax)) # the order is (left, top, right, bottom) for the crop function

        # Save the cropped image
        cropped_img.save(extracted_image_path)
        print(f"Cropped image saved to: {extracted_image_path}")

def parse_page(page_image_path: str, page_number: int, save_path: str, vlm_config: dict) -> list[dict]:
    """
    Given an image of a page, use LLM to extract the content of the page.

    Inputs:
    - page_image_path: str, path to the image of the page
    - page_number: int, the page number
    - save_path: str, path to the base directory where everything is saved (i.e. {user_id}/{job_id})
    
    Outputs:
    - page_content: list of dictionaries, each containing information about an element on the page
    """
    if vlm_config["provider"] == "vertex_ai":
        llm_output = make_llm_call_gemini(
            image_path=page_image_path, 
            system_message=SYSTEM_MESSAGE, 
            model=vlm_config["model"], 
            project_id=vlm_config["project_id"], 
            location=vlm_config["location"],
            response_schema=response_schema,
            max_tokens=4000
            )
    else:
        raise ValueError("Invalid provider specified in the VLM config. Only 'vertex_ai' is supported for now.")
    
    try:
        page_content = json.loads(llm_output)
    except json.JSONDecodeError:
        print(f"Error decoding JSON response from LLM for {page_image_path}")
        print(llm_output)
        page_content = []

    # save images for each bounding box
    i = 0 # counter for the number of images extracted
    for element in page_content:
        element["page_number"] = page_number
        if element["type"] in ["Image", "Figure"]:
            bounding_box = element["bounding_box"]

            # run the bounding box through the bounding box retry function to improve accuracy
            bounding_box = get_improved_bounding_box(page_image_path, bounding_box, vlm_config, i)
            element["improved_bounding_box"] = bounding_box

            # create the directory to save the extracted images if it doesn't exist
            if not os.path.exists(f"{save_path}/extracted_images"):
                os.makedirs(f"{save_path}/extracted_images")

            print(f"Extracting image from bounding box: {bounding_box}")
            extracted_image_path = os.path.join(save_path, f"extracted_images/page_{page_number}_image_{i}.png")
            extract_image(page_image_path, bounding_box, extracted_image_path)

            # add image path to the element
            element["image_path"] = extracted_image_path
            i += 1

    return page_content

def parse_file(pdf_path: str, save_path: str, vlm_config: dict) -> list[dict]:
    """
    Given a PDF file, extract the content of each page using a VLM model.

    Inputs
    - pdf_path: str, path to the PDF file
    - save_path: str, path to the base directory where everything is saved (i.e. {user_id}/{job_id})
    - config: dict, configuration for the VLM model. For Gemini this should include project_id and location.

    Outputs
    - all_page_content: list of dictionaries, each containing information about an element on a page, for all pages in the PDF, in order
    """
    page_images_path = f"{save_path}/page_images"
    image_file_paths = pdf_to_images(pdf_path, page_images_path)
    all_page_content = []
    for i, image_path in enumerate(image_file_paths[0:5]):
        print (f"Processing {image_path}")
        page_content = parse_page(image_path, page_number=i+1, save_path=save_path, vlm_config=vlm_config)
        all_page_content.extend(page_content)
        time.sleep(3) # sleep for a few seconds to avoid rate limit issues with the Gemini API

    # save the extracted content to a JSON file
    output_file_path = f"{save_path}/elements.json"
    with open(output_file_path, "w") as f:
        json.dump(all_page_content, f, indent=2)

    return all_page_content


if __name__ == "__main__":
    user_id = "zmcc"

    pdf_path = '/Users/zach/Code/dsRAG/tests/data/levels_of_agi.pdf'
    file_id = "levels_of_agi"
    
    #pdf_path = "/Users/zach/Code/mck_energy.pdf"
    #file_id = "mck_energy"

    save_path = f"{user_id}/{file_id}" # base directory to save the page images, pages with bounding boxes, and extracted images

    vlm_config = {
        "provider": "vertex_ai",
        "model": "gemini-1.5-pro-002",
        "project_id": os.environ["VERTEX_PROJECT_ID"],
        "location": "us-central1"
    }
    
    parse_file(pdf_path, save_path, vlm_config)
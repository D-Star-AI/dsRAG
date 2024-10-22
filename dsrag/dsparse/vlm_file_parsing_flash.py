from dsrag.dsparse.vertex_ai import make_llm_call_gemini
import concurrent.futures
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
You are a PDF -> MD file parser. Your task is to analyze the provided PDF page (provided as an image) and return a structured JSON response containing all of the elements on the page. Each element must be represented using Markdown formatting.

There are two categories of elements you need to identify: text elements and visual elements. Text elements are those that can be accurately represented using plain text. Visual elements are those that need to be represented as images to fully capture their content. For text elements, you must provide the full text content. For visual elements, you must provide a detailed description of the content and a bounding box around the content.

There are four types of visual elements: Figure, Image, Table, and Equation.
There are four types of text elements: NarrativeText, Header, Footnote, and Footer.

Every element on the page should be classified as one of these types. There should be no overlap between elements. You should use the smallest number of elements possible while still accurately representing the content on the page. For example, if the page contains a couple paragraphs of text, followed by a large figure, followed by a few more paragraphs of text, you should use three elements: NarrativeText, Figure, and NarrativeText.

Here are detailed descriptions of the seven element types you can use:
- NarrativeText
    - This is the main text content of the page, including paragraphs, lists, titles, and any other text content that is not part of a header, footer, figure, table, or image. Not all pages have narrative text, but most do. Be sure to use Markdown formatting for the text content. This includes using tags like # for headers, * for lists, etc. Make sure your header tags are properly nested and that your lists are properly formatted.
- Figure
    - This covers charts, graphs, diagrams, complex tables, etc. Associated titles, legends, axis titles, etc. should be considered to be part of the figure. Be sure your descriptions and bounding boxes fully capture these associated items, as they are essential for providing context to the figure. 
- Image
    - This is any visual content on the page that isn't a figure. This could include photos, illustrations, etc. Any title or captions associated with the image should be considered part of the image. Be sure your descriptions and bounding boxes fully capture these associated items, as they are essential for providing context to the image. Do not include background images or other images that don't convey any information.
- Table
    - If a table can be represented accurately using Markdown, then it should be included as a Table element. Proper Markdown formatting must be used for the content. If not, it should be included as a Figure element instead. Be especially careful with multi-column tables: it's critical that you use proper Markdown formatting to ensure the table is accurately represented.
- Header
    - This is the header of the page, which would be located at the very top of the page and may include things like a document titles and page number. You should never user more than one header element per page. Not all pages have a header.
- Footnote
    - Footnotes should always be included as a separate element from the main text content as they aren't part of the main linear reading flow of the page. Not all pages have footnotes.
- Footer
    - This is the footer of the page, which would be located at the very bottom of the page. You should never user more than one footer element per page. Not all pages have a footer, but when they do it is always the very last element on the page.

For Image and Figure elements ONLY, you must provide a detailed description of the image or figure in the "content" field. Do not just transcribe the actual text contained in the image or figure. For all other element types, you must provide the text content of the element.

Output format
- Your output should be an ordered (from top to bottom) list of elements on the page, where each element is a dictionary with the following keys:
    - type: str - the type of the element (e.g. "NarrativeText", "Figure", "Image", "Table", "Header", "Footnote", or "Footer")
    - content: str - the content of the element. For "Figure" and "Image" elements, this should be a detailed description of the visual content, rather than a transcription of the actual text contained in the element. You can use Markdown formatting for text content. Always use Markdown for tables.
    - bounding_box: list[int] (ONLY include when type is "Image" or "Figure". For other element types, just use an empty list here) - a bounding box around the image or figure, in the format [ymin, xmin, ymax, xmax]. Be sure the bounding box fully captures the content of the image or figure, including any associated titles, legends, axis titles, etc.

Complex and multi-part figures or images should be represented as a single element. For example, if a figure consists of a main chart and a smaller inset chart, these should be described together in a single Figure element. If there are two separate graphs side by side, these should be represented as a single Figure element with a bounding box that encompasses both graphs. DO NOT create separate elements for each part of a complex figure or image.

Additional instructions
- Ignore background images or other images that don't convey any information.
- The element types described above are the only ones you are allowed to use.
- Be sure to include all page content in your response.
- Image and Figure elements MUST have accurate bounding boxes.
- For mathematical expressions and equations, you must use LaTeX formatting.
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
            "bounding_box": {
                "type": "array",
                "items": {
                    "type": "number",
                },
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

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def extract_image(page_image_path: str, bounding_box: list[int], extracted_image_path: str, padding: int = 150):
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
    ymin = max(0, ymin - padding)
    xmin = max(0, xmin - padding)
    ymax = min(1000, ymax + padding)
    xmax = min(1000, xmax + padding)
    
    # Open the image
    with Image.open(page_image_path) as img:
        width, height = img.size
        print(f"Original image size: {width}x{height}")

        # Calculate actual pixel coordinates
        actual_ymin = int((ymin/1000) * height)
        actual_xmin = int((xmin/1000) * width)
        actual_ymax = int((ymax/1000) * height)
        actual_xmax = int((xmax/1000) * width)

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
        try:
            llm_output = make_llm_call_gemini(
                image_path=page_image_path, 
                system_message=SYSTEM_MESSAGE, 
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

            # check if the bounding box is valid - should be a list of 4 integers all between 0 and 1000
            if not isinstance(bounding_box, list) or len(bounding_box) != 4 or not all(isinstance(x, int) for x in bounding_box) or not all(0 <= x <= 1000 for x in bounding_box):
                print(f"Invalid bounding box for element {element} on page {page_number}. Using the entire page instead.")
                bounding_box = [0, 0, 1000, 1000] # use a bounding box that represents the entire page

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
    - vlm_config: dict, configuration for the VLM model. For Gemini this should include project_id and location.
    Outputs
    - all_page_content: list of dictionaries, each containing information about an element on a page, for all pages in the PDF, in order
    """
    page_images_path = f"{save_path}/page_images"
    image_file_paths = pdf_to_images(pdf_path, page_images_path)
    all_page_content_dict = {}

    def process_page(image_path, page_number):
        tries = 0
        while tries < 20:
            content = parse_page(image_path, page_number=page_number, save_path=save_path, vlm_config=vlm_config)
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

def elements_to_markdown(elements: list[dict]) -> str:
    """
    Given a list of elements extracted from a PDF, convert them to a markdown string.
    Inputs
    - elements: list of dictionaries, each containing information about an element on a page
    Outputs
    - markdown_string: str, a markdown string representing the elements
    """
    markdown_string = ""
    for element in elements:
        if element["type"] in ["Figure", "Image"]:
            markdown_string += f"![Image {element['page_number']}](./{element['image_path']})\n\n"
        else:
            markdown_string += f"{element['content']}\n\n"

    return markdown_string
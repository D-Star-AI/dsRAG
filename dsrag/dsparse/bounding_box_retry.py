from PIL import Image, ImageDraw
import json
from vertex_ai import make_llm_call_gemini

SYSTEM_MESSAGE = """
Your task is to improve the accuracy of the bounding box shown in the image. This bounding box was generated by an AI model, and it may not accurately capture the content of the image or figure.

The bounding box shown in the image has coordinates ({ymin}, {xmin}, {ymax}, {xmax}).

Here is a description of the image or figure that the bounding box is supposed to capture:
{description}

Here is what a good bounding box should look like:
- The bounding box should capture the entire image or figure described above.
- It should capture all relevant elements of the image or figure, including legends, labels, figure and axis titles, and other important features that are necessary to understand the content.
- The bounding box should not include any unnecessary elements that are not relevant to the image or figure described above.
- If you only see what's inside the bounding box, you should be able to understand the content of the image or figure without any additional context.

Please provide a new set of coordinates for the bounding box that more accurately captures the image or figure described above. The new coordinates should be in the format [ymin, xmin, ymax, xmax].

The coordinate system works as follows:
- The origin (0, 0) is the top-left corner of the image.
- The y-axis is vertical, with increasing values going down.
- The x-axis is horizontal, with increasing values going to the right.
- The bottom-right corner of the image has coordinates (1000, 1000).
- For example, the coordinates [0, 0, 1000, 1000] would represent a bounding box that covers the entire image. The coordinates [0, 0, 500, 500] would represent a bounding box that covers the top-left quadrant of the image.

Remember, the original bounding box may not be very good, so don't be afraid to make significant changes to the coordinates to improve its accuracy. You may need to adjust the coordinates by as much as 20% (200 points) in any direction to capture the entire image or figure.

Your response should have two parts:
1. A one sentence description of the changes that need to be made to the bounding box to improve its accuracy.
2. A list of 4 integers, separated by commas, that represent the new coordinates of the bounding box. Include the opening and closing square brackets, and do not include any additional text after the closing bracket.
""".strip()


def add_box_to_image(page_image_path: str, bounding_box: list[int], save_path: str, color: str = "red"):
    """
    This function adds a bounding box to an image and saves the image with the bounding box drawn on it.
    
    Inputs:
    - page_image_path: str, path to the image file
    - bounding_box: list[int], list of integers representing the bounding box coordinates in the format [ymin, xmin, ymax, xmax]
    - save_path: str, path to save the image with the bounding box drawn on it
    """

    with Image.open(page_image_path) as img:
        width, height = img.size
        print(f"Original image size: {width}x{height}")

        # Calculate actual pixel coordinates
        ymin_scaled, xmin_scaled, ymax_scaled, xmax_scaled = bounding_box
        actual_xmin = int(xmin_scaled / 1000 * width)
        actual_ymin = int(ymin_scaled / 1000 * height)
        actual_xmax = int(xmax_scaled / 1000 * width)
        actual_ymax = int(ymax_scaled / 1000 * height)

        # Draw bounding box on the image
        draw = ImageDraw.Draw(img)
        draw.rectangle([actual_xmin, actual_ymin, actual_xmax, actual_ymax], outline=color, width=3)

        # Save the image with the bounding box
        img.save(save_path)

def get_improved_bounding_box(page_image_path: str, bounding_box: list[int], vlm_config: dict, counter: int = 0):
    """
    This function takes in the path to the page image and the bounding box coordinates, and returns an improved bounding box.

    Inputs:
    - page_image_path: str, path to the image file
    - bounding_box: list[int], list of integers representing the bounding box coordinates in the format [ymin, xmin, ymax, xmax]
    - counter: int, the counter for the number of times this function has been called for a given page; used to ensure file uniqueness

    Returns:
    - new_bbox: list[int], list of integers representing the improved bounding box coordinates in the format [ymin, xmin, ymax, xmax]
    """
    image_path_w_bbox = f"{page_image_path}_with_box_{counter}.png"
    add_box_to_image(page_image_path, bounding_box, image_path_w_bbox, color="red")
    llm_response = make_llm_call_gemini(
        image_path=image_path_w_bbox, 
        system_message=SYSTEM_MESSAGE, 
        model=vlm_config["model"], 
        project_id=vlm_config["project_id"], 
        location=vlm_config["location"],
        max_tokens=500
        )
    llm_response_parts = llm_response.split("\n")
    new_bbox = json.loads(llm_response_parts[-1]) # TODO: add error handling for this
    return new_bbox


# Example usage
if __name__ == "__main__":
    page_number = 24
    image_path = f"/Users/zach/Code/pdf_to_images/mck_energy/page_{page_number}.png"
    #bounding_box = [298, 330, 540, 925] # [ymin, xmin, ymax, xmax]
    bounding_box = [315, 432, 577, 703]

    # get an improved bounding box
    new_bounding_box = get_improved_bounding_box(image_path, bounding_box)

    # add a new bounding box to the image
    image_path_w_bbox = f"{image_path}_with_box.png"
    add_box_to_image(image_path_w_bbox, new_bounding_box, image_path_w_bbox, color="green")
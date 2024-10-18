import os
import sys
from typing import Dict, List

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from dsrag.knowledge_base import KnowledgeBase


def create_kb_and_add_document():
    kb = KnowledgeBase(kb_id="mck_energy", exists_ok=True)
    kb.delete()

    file_path = "/Users/zach/Code/mck_energy.pdf"
    kb = KnowledgeBase(kb_id="mck_energy")
    kb.add_document(
        doc_id="mck_energy_report",
        file_path=file_path,
        document_title="McKinsey Energy Report",
        file_parsing_config={
            "use_vlm": True,
            "vlm_config": {
                "provider": "vertex_ai",
                "model": "gemini-1.5-pro-002",
                "project_id": os.environ["VERTEX_PROJECT_ID"],
                "location": "us-central1",
                "save_path": "zmcc/mck_energy"
            }
        }
    )

    return kb

def visualize_multi_modal_results(search_results: List[Dict], output_file: str = "visualization.html"):
    """
    Writes search results to an HTML file for visualization.

    Inputs:
    - search_results (List[Dict]): A list of dictionaries, ordered by relevance, that each contain:
        - doc_id (str): The document ID of the document that the segment is from
        - chunk_start (int): The start index of the segment in the document
        - chunk_end (int): The (non-inclusive) end index of the segment in the document
        - content (List[Dict]): A list of dictionaries, where each dictionary has the following keys:
            - type (str): Either "text" or "image"
            - content (str): The text content (if 'type' is "text") or image path (if 'type' is "image")
        - segment_page_start (int): The page number that the segment starts on
        - segment_page_end (int): The page number that the segment ends on

    - output_file (str): The filename for the output HTML file.

    Saves:
    - An HTML file that contains the text and images from the search results.
    """

    # HTML boilerplate with simple CSS for styling
    html_header = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Search Results Visualization</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 20px;
            }
            .result {
                border: 1px solid #ccc;
                padding: 15px;
                margin-bottom: 20px;
                border-radius: 5px;
            }
            .metadata {
                font-size: 0.9em;
                color: #555;
                margin-bottom: 10px;
            }
            .content p {
                line-height: 1.6;
            }
            .content img {
                max-width: 100%;
                height: auto;
                margin-top: 10px;
                margin-bottom: 10px;
            }
        </style>
    </head>
    <body>
        <h1>Search Results Visualization</h1>
    """

    html_footer = """
    </body>
    </html>
    """

    # Start writing the HTML content
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_header)

        # Iterate through each search result
        for idx, result in enumerate(search_results, start=1):
            doc_id = result.get('doc_id', 'N/A')
            chunk_start = result.get('chunk_start', 'N/A')
            chunk_end = result.get('chunk_end', 'N/A')
            segment_page_start = result.get('segment_page_start', 'N/A')
            segment_page_end = result.get('segment_page_end', 'N/A')
            content = result.get('content', [])

            # Write the result container
            f.write(f'<div class="result">\n')

            # Write metadata
            f.write(f'<div class="metadata">\n')
            f.write(f'<strong>Result {idx}:</strong><br>\n')
            f.write(f'Document ID: {doc_id}<br>\n')
            f.write(f'Chunk: {chunk_start} to {chunk_end}<br>\n')
            f.write(f'Pages: {segment_page_start} to {segment_page_end}\n')
            f.write(f'</div>\n')  # Close metadata

            # Write content
            f.write(f'<div class="content">\n')
            for item in content:
                content_type = item.get('type')
                content_data = item.get('content', '')

                if content_type == 'text':
                    f.write(f'<p>{content_data}</p>\n')
                elif content_type == 'image':
                    # Check if the image path exists
                    if os.path.isfile(content_data):
                        # Use relative paths for portability
                        img_path = os.path.relpath(content_data, os.path.dirname(output_file))
                        f.write(f'<img src="{img_path}" alt="Image from {doc_id}">\n')
                    else:
                        f.write(f'<p><em>Image not found: {content_data}</em></p>\n')
                else:
                    f.write(f'<p><em>Unknown content type: {content_type}</em></p>\n')
            f.write('</div>\n')  # Close content

            f.write('</div>\n')  # Close result

        f.write(html_footer)

    print(f"Visualization has been saved to {output_file}")

def get_response(user_input: str, search_results: List[Dict]):
    """
    Given a user input and potentially multimodal search results, generate a response
    """
    pass

# create or load KB
#kb = create_kb_and_add_document()
kb = KnowledgeBase(kb_id="mck_energy")

# print the number of chunks in the KB
num_chunks = len(kb.chunk_db.data["mck_energy_report"])
print(f"Number of chunks in the KB: {num_chunks}")

#query = "What is the McKinsey Energy Report about?"
query = "How does the oil and gas industry compare to other industries in terms of its value prop to employees?"

rse_params = {
    "minimum_value": 0.4,
    "irrelevant_chunk_penalty": 0.2,
}

search_results = kb.query(search_queries=[query], rse_params=rse_params, return_images=True)
print (search_results)

visualize_multi_modal_results(search_results=search_results, output_file="mck_energy_search_results.html")
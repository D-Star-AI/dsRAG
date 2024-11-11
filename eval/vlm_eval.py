import os
import sys
from typing import Dict, List
import PIL.Image
import google.generativeai as genai

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from dsrag.knowledge_base import KnowledgeBase


def create_kb_and_add_document(kb_id: str, file_path: str, document_title: str):
    kb = KnowledgeBase(kb_id=kb_id, exists_ok=True)
    kb.delete()

    kb = KnowledgeBase(kb_id=kb_id, exists_ok=False)
    kb.add_document(
        doc_id="test_doc_1",
        file_path=file_path,
        document_title=document_title,
        file_parsing_config={
            "use_vlm": True,
            "vlm_config": {
                "provider": "gemini",
                "model": "gemini-1.5-flash-002",
            }
        }
    )

    return kb

def get_response(user_input: str, search_results: List[Dict], model_name: str = "gemini-1.5-pro-002"):
    """
    Given a user input and potentially multimodal search results, generate a response.

    We'll use the Gemini API since we already have code for it.
    """
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    generation_config = {
        "temperature": 0.0,
        "max_output_tokens": 2000
    }

    model = genai.GenerativeModel(model_name)
    
    all_image_paths = []
    for search_result in search_results:
        if type(search_result["content"]) == list:
            all_image_paths += search_result["content"]

    images = [PIL.Image.open(image_path) for image_path in all_image_paths]
    system_message = "Please cite the page number of the document used to answer the question."
    generation_input = images + [user_input, system_message] # user input is the last element
    response = model.generate_content(
        contents=generation_input,
        generation_config=generation_config
    )
    [image.close() for image in images]
    return response.text


kb_id = "state_of_ai_flash"

"""
# create or load KB
file_path = "/Users/zach/Code/test_docs/state_of_ai_report_2024.pdf"
document_title = "State of AI Report 2024"
kb = create_kb_and_add_document(kb_id=kb_id, file_path=file_path, document_title=document_title)

"""

kb = KnowledgeBase(kb_id=kb_id, exists_ok=True)

#query = "What is the McKinsey Energy Report about?"
#query = "How does the oil and gas industry compare to other industries in terms of its value prop to employees?"

query = "Who is Nathan Benaich?" # page 2
#query = "Which country had the most AI publications in 2024?" # page 84
#query = "Did frontier labs increase or decrease publications in 2024? By how much?" # page 84
#query = "Who has the most H100s? How many do they have?" # page 92
#query = "How many H100s does Lambda have?" # page 92

rse_params = {
    "minimum_value": 0.5,
    "irrelevant_chunk_penalty": 0.2,
}

search_results = kb.query(search_queries=[query], rse_params=rse_params, return_mode="page_images")
print (search_results)

response = get_response(user_input=query, search_results=search_results)
print(response)
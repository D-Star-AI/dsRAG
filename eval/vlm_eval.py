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
                "model": "gemini-1.5-pro-002",
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
    generation_input = images + [user_input] # user input is the last element
    response = model.generate_content(
        contents=generation_input,
        generation_config=generation_config
    )
    [image.close() for image in images]
    return response.text


# create or load KB
kb_id = "mck_energy"
file_path = "/Users/zach/Code/test_docs/mck_energy.pdf"
document_title = "McKinsey Energy Report"
#kb = create_kb_and_add_document(kb_id=kb_id, file_path=file_path, document_title=document_title)

kb = KnowledgeBase(kb_id=kb_id, exists_ok=True)

# print the number of chunks in the KB
doc_id = "test_doc_1"
num_chunks = len(kb.chunk_db.data[doc_id])
print(f"Number of chunks in the KB: {num_chunks}")

#query = "What is the McKinsey Energy Report about?"
query = "How does the oil and gas industry compare to other industries in terms of its value prop to employees?"

rse_params = {
    "minimum_value": 0.5,
    "irrelevant_chunk_penalty": 0.2,
}

search_results = kb.query(search_queries=[query], rse_params=rse_params, return_mode="page_images")
#print (search_results)

response = get_response(user_input=query, search_results=search_results)
print(response)
import os
import sys

# add ../../ to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# import the necessary modules
from sprag.knowledge_base import KnowledgeBase
from sprag.embedding import CohereEmbedding

document_directory = "" # path to the directory containing the documents, links to which are contained in the `tests/data/financebench_sample_150.csv` file
kb_id = "finance_bench"

"""
Note: to run this, you'll need to first download all of the documents (~80 of them, mostly 10-Ks and 10-Qs) and convert them to txt files. You'll want to make sure they all have consistent and meaningful file names, such as costco_2023_10k.txt, to match the performance we quote.
"""

# create a new KnowledgeBase object
embedding_model = CohereEmbedding()
kb = KnowledgeBase(kb_id, embedding_model=embedding_model, exists_ok=False)

for file_name in os.listdir(document_directory):
    with open(os.path.join(document_directory, file_name), "r") as f:
        text = f.read()
        kb.add_document(doc_id=file_name, text=text)
        print (f"Added {file_name} to the knowledge base.")
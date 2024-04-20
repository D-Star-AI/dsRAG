import sys
import os

# add ../ to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sprag.knowledge_base import create_kb_from_file

file_path = "../sample_data/levels_of_agi.pdf"
file_path = os.path.abspath(file_path) # convert to absolute path
kb_id = "ai_papers"
kb = create_kb_from_file(kb_id, file_path)
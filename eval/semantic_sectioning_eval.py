import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

#from dsrag.semantic_sectioning import get_sections
from dsrag.sectioning_and_chunking.semantic_sectioning import get_sections
from dsrag.document_parsing import extract_text_from_pdf

# Get the absolute path of the script file
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the absolute path to the dataset file
file_path = os.path.join(script_dir, "../tests/data/levels_of_agi.pdf")
#file_path = os.path.join(script_dir, "../tests/data/nike_2023_annual_report.txt")
#file_path = os.path.join(script_dir, "../tests/data/les_miserables.txt")

file_name = file_path.split("/")[-1].split(".")[0]
print(f"Processing {file_name}...")

if file_path.endswith(".pdf"):
    document_text = extract_text_from_pdf(file_path)
elif file_path.endswith(".txt"):
    with open(file_path, "r") as f:
        document_text = f.read()
else:
    raise ValueError("File must be a PDF or TXT file")

llm_provider = "openai"
model = "gpt-4o-mini"
chunk_size = 1000
min_length_for_chunking = 2000
chunking_method = "semantic"

sections, document_lines = get_sections(document_text, max_characters=20000, llm_provider="openai", model=model, language="en")

# write sections to file
with open(f"{file_name}_sections.txt", "w") as f:
    for s in sections:
        f.write(f"\n\nStart: {s['start']}\n")
        f.write(f"End: {s['end']}\n")
        f.write(f"Title: {s['title']}\n")
        f.write(f"\n{s['content']}\n")
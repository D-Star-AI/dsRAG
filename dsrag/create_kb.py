from dsrag.document_parsing import extract_text_from_pdf, extract_text_from_docx
from dsrag.knowledge_base import KnowledgeBase
import os
import time

def create_kb_from_directory(kb_id: str, directory: str, title: str = None, description: str = "", language: str = 'en'):
    """
    - kb_id is the name of the knowledge base
    - directory is the absolute path to the directory containing the documents
    - no support for manually defined chunk headers here, because they would have to be defined for each file in the directory

    Supported file types: .docx, .md, .txt, .pdf
    """
    if not title:
        title = kb_id
    
    # create a new KB
    kb = KnowledgeBase(kb_id, title=title, description=description, language=language, exists_ok=False)

    # add documents
    for root, dirs, files in os.walk(directory):
        for file_name in files:
            if file_name.endswith(('.docx', '.md', '.txt', '.pdf')):
                try:
                    file_path = os.path.join(root, file_name)
                    clean_file_path = file_path.replace(directory, "")
                    
                    if file_name.endswith('.docx'):
                        text = extract_text_from_docx(file_path)
                    elif file_name.endswith('.pdf'):
                        text, _ = extract_text_from_pdf(file_path)
                    elif file_name.endswith('.md') or file_name.endswith('.txt'):
                        with open(file_path, 'r') as f:
                            text = f.read()

                    kb.add_document(doc_id=clean_file_path, text=text)
                    time.sleep(1) # pause for 1 second to avoid hitting API rate limits
                except:
                    print (f"Error reading {file_name}")
                    continue
            else:
                print (f"Unsupported file type: {file_name}")
                continue
    
    return kb

def create_kb_from_file(kb_id: str, file_path: str, title: str = None, description: str = "", language: str = 'en'):
    """
    - kb_id is the name of the knowledge base
    - file_path is the absolute path to the file containing the documents

    Supported file types: .docx, .md, .txt, .pdf
    """
    if not title:
        title = kb_id
    
    # create a new KB
    kb = KnowledgeBase(kb_id, title=title, description=description, language=language, exists_ok=False)
    
    print (f'Creating KB with id {kb_id}...')

    file_name = os.path.basename(file_path)

    # add document
    if file_path.endswith(('.docx', '.md', '.txt', '.pdf')):
        # define clean file path as just the file name here since we're not using a directory
        clean_file_path = file_name
        
        if file_path.endswith('.docx'):
            text = extract_text_from_docx(file_path)
        elif file_name.endswith('.pdf'):
            text, _ = extract_text_from_pdf(file_path)
        elif file_path.endswith('.md') or file_path.endswith('.txt'):
            with open(file_path, 'r') as f:
                text = f.read()

        kb.add_document(doc_id=clean_file_path, text=text)
    else:
        print (f"Unsupported file type: {file_name}")
        return
    
    return kb
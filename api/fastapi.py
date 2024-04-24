from fastapi import FastAPI, HTTPException
import os
from sprag.knowledge_base import KnowledgeBase

app = FastAPI()

# Load all databases located in the ~/.sprag folder into a dictionary
kb_path = os.path.join(os.path.expanduser("~"), ".sprag")
if not (os.path.exists(kb_path)):
    os.mkdir(kb_path)
kb_ids = os.listdir(kb_path)

# Create a dictionary of databases keyed on name
knowledge_bases = {kb_id: KnowledgeBase(kb_id, storage_directory=kb_path) for kb_id in kb_ids}

# API routes
@app.get("/health")
def read_root():
    return {"status": "healthy"}

@app.post("/db/create")
def create_db(create_db_input: CreateDBInput):
    if create_db_input.name in knowledge_bases:
        raise HTTPException(
            status_code=400, detail="Knowledge base with this name already exists")
    kb = KnowledgeBase(kb_id=create_db_input.kb_id)
    knowledge_bases[create_db_input.name] = kb
    return {"message": "Database created successfully"}
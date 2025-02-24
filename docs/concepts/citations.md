# Citations

dsRAG's citation system ensures that responses are grounded in your knowledge base content and provides transparency about information sources.

## Overview

Citations in dsRAG:
- Track the source of information in responses
- Include document IDs and page numbers
- Provide exact quoted text from sources
- Link back to original documents

## Citation Structure

Each citation includes:
- `doc_id`: Unique identifier of the source document
- `page_number`: Page where the information was found (if available)
- `cited_text`: Exact text containing the cited information
- `kb_id`: ID of the knowledge base containing the document

## Working with Citations

Citations are automatically included in chat responses:

```python
from dsrag.chat.chat import get_chat_thread_response
from dsrag.chat.chat_types import ChatResponseInput
from dsrag.database.chat_thread.sqlite_db import SQLiteChatThreadDB

# Create the knowledge base instances
knowledge_bases = {
    "my_knowledge_base": KnowledgeBase(kb_id="my_knowledge_base")
}

# Initialize database and get response
chat_thread_db = SQLiteChatThreadDB()
response = get_chat_thread_response(
    thread_id=thread_id,
    get_response_input=ChatResponseInput(
        user_input="What is the system architecture?"
    ),
    chat_thread_db=chat_thread_db,
    knowledge_bases=knowledge_bases
)

# Access citations
citations = response["model_response"]["citations"]
for citation in citations:
    print(f"""
Source: {citation['doc_id']}
Page: {citation['page_number']}
Text: {citation['cited_text']}
Knowledge Base: {citation['kb_id']}
""")
```

## Technical Details

Citations are managed through the `ResponseWithCitations` model:

```python
from dsrag.chat.citations import ResponseWithCitations, Citation

response = ResponseWithCitations(
    response="The system architecture is...",
    citations=[
        Citation(
            doc_id="arch-doc-v1",
            page_number=12,
            cited_text="The system uses a microservices architecture"
        )
    ]
)
```

This structured format ensures consistent citation handling throughout the system.
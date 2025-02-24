# Chat

The Chat functionality in dsRAG provides a powerful way to interact with your knowledge bases through a conversational interface. It handles message history, knowledge base searching, and citation tracking automatically.

## Overview

The chat system works by:
1. Maintaining a chat thread with message history
2. Automatically generating relevant search queries based on user input
3. Searching knowledge bases for relevant information
4. Generating responses with citations to source materials

## Defining Where to Store Chat History

Chat threads in dsRAG need to be persisted somewhere so that conversations can continue across multiple interactions. You'll need to provide your own implementation of the `ChatThreadDB` class to handle this storage. This allows you to store chat threads in whatever database or storage system works best for your application.

The `ChatThreadDB` interface defines methods for:
- Creating new chat threads
- Retrieving existing threads
- Adding interactions to threads
- Managing thread metadata and configuration

You'll need to initialize your chat thread storage before creating any new chat threads or retrieving responses.

There are two implementations of the `ChatThreadDB` interface included:
- `BasicChatThreadDB`: A basic implementation that stores chat threads in a JSON file
- `SQLiteChatThreadDB`: A SQLite implementation that stores chat threads in a SQLite database

## Creating a Chat Thread

To start a conversation, first create a chat thread:

```python
from dsrag.chat.chat import create_new_chat_thread
from dsrag.database.chat_thread.sqlite_db import SQLiteChatThreadDB

# Configure chat parameters
chat_params = {
    "kb_ids": ["my_knowledge_base"],  # List of knowledge base IDs to use
    "model": "gpt-4o",                 # LLM model to use
    "temperature": 0.2,               # Response creativity (0.0-1.0)
    "system_message": "You are a helpful assistant specialized in technical documentation",
    "target_output_length": "medium"  # "short", "medium", or "long"
}

# Initialize chat thread database (SQLite in this case)
chat_thread_db = SQLiteChatThreadDB()

# Create the thread
thread_id = create_new_chat_thread(chat_params, chat_thread_db)
```

## Getting Responses

Once you have a thread, you can send messages and get responses:

```python
from dsrag.chat.chat import get_chat_thread_response
from dsrag.chat.chat_types import ChatResponseInput

# Create input with optional metadata filter
response_input = ChatResponseInput(
    user_input="What are the key features of XYZ product?",
    metadata_filter={
        "document_type": "technical_spec",
        "version": "latest"
    }
)

# Create the knowledge base instances
knowledge_bases = {
    "my_knowledge_base": KnowledgeBase(kb_id="my_knowledge_base")
}

# Get response
response = get_chat_thread_response(
    thread_id=thread_id,
    get_response_input=response_input,
    chat_thread_db=chat_thread_db,
    knowledge_bases=knowledge_bases  # Dictionary of your knowledge base instances
)

# Access the response content and citations
print(response["model_response"]["content"])
for citation in response["model_response"]["citations"]:
    print(f"Source: {citation['doc_id']}, Page: {citation['page_number']}")
```

## Chat Thread Parameters

The chat thread parameters dictionary supports several configuration options:

- `kb_ids`: List of knowledge base IDs to search
- `model`: LLM model to use (e.g., "gpt-4")
- `temperature`: Controls response randomness (0.0-1.0)
- `system_message`: Custom instructions for the LLM
- `auto_query_model`: Model to use for generating search queries
- `auto_query_guidance`: Custom guidance for query generation
- `target_output_length`: Desired response length ("short", "medium", "long")
- `max_chat_history_tokens`: Maximum tokens to keep in chat history

## Response Structure

Chat responses include:

- User input with timestamp
- Model response with citations
- Search queries used
- Relevant segments found
- File names and types for citations

## Metadata Filtering

You can filter knowledge base searches using metadata:

```python
metadata_filter = {
    "document_type": "manual",
    "department": "engineering",
    "status": "approved"
}

response_input = ChatResponseInput(
    user_input="How do I configure the system?",
    metadata_filter=metadata_filter
)
```
## Best Practices

1. Set appropriate `target_output_length` based on your use case
2. Use `system_message` to guide the LLM's behavior
3. Configure `max_chat_history_tokens` based on your needs
4. Use metadata filters to focus searches on relevant documents
5. Monitor and adjust `temperature` based on desired response creativity
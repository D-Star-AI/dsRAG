# Chat

The Chat module provides functionality for managing chat-based interactions with knowledge bases. It handles chat thread creation, message history, and generating responses using knowledge base search.

### Public Methods

#### create_new_chat_thread

```python
def create_new_chat_thread(chat_thread_params: ChatThreadParams, chat_thread_db: ChatThreadDB) -> str
```

Create a new chat thread in the database.

**Arguments**:
- `chat_thread_params`: Parameters for the chat thread. Example:
  ```python
  {
      # Knowledge base IDs to use
      "kb_ids": ["kb1", "kb2"],
      
      # LLM model to use
      "model": "gpt-4o-mini",
      
      # Temperature for LLM sampling
      "temperature": 0.2,
      
      # System message for LLM
      "system_message": "You are a helpful assistant",
      
      # Model for auto-query generation
      "auto_query_model": "gpt-4o-mini",
      
      # Guidance for auto-query generation
      "auto_query_guidance": "",
      
      # Target response length (short/medium/long)
      "target_output_length": "medium",
      
      # Maximum tokens in chat history
      "max_chat_history_tokens": 8000,
      
      # Optional supplementary ID
      "supp_id": ""
  }
  ```
- `chat_thread_db`: Database instance for storing chat threads.

**Returns**: Unique identifier (str) for the created chat thread.

#### get_chat_thread_response

```python
def get_chat_thread_response(thread_id: str, get_response_input: ChatResponseInput, 
                           chat_thread_db: ChatThreadDB, knowledge_bases: dict)
```

Get a response for a chat thread using knowledge base search.

**Arguments**:
- `thread_id`: Unique identifier for the chat thread.
- `get_response_input`: Input parameters containing:
  - `user_input`: User's message text
  - `chat_thread_params`: Optional parameter overrides
  - `metadata_filter`: Optional search filter
- `chat_thread_db`: Database instance for chat threads.
- `knowledge_bases`: Dictionary mapping knowledge base IDs to instances.

**Returns**: Formatted interaction containing:
- `user_input`: User message with content and timestamp
- `model_response`: Model response with content and timestamp
- `search_queries`: Generated search queries
- `relevant_segments`: Retrieved relevant segments with file names and types
- `message`: Error message if something went wrong (optional)

### Chat Types

#### ChatThreadParams

Type definition for chat thread parameters.

```python
ChatThreadParams = TypedDict('ChatThreadParams', {
    'kb_ids': List[str],              # List of knowledge base IDs to use
    'model': str,                     # LLM model name (e.g., "gpt-4o-mini")
    'temperature': float,             # Temperature for LLM sampling (0.0-1.0)
    'system_message': str,            # Custom system message for LLM
    'auto_query_model': str,          # Model for generating search queries
    'auto_query_guidance': str,       # Custom guidance for query generation
    'target_output_length': str,      # Response length ("short", "medium", "long")
    'max_chat_history_tokens': int,   # Maximum tokens in chat history
    'thread_id': str,                 # Unique thread identifier (auto-generated)
    'supp_id': str,                   # Optional supplementary identifier
})
```

#### ChatResponseInput

Input parameters for getting a chat response.

```python
ChatResponseInput = TypedDict('ChatResponseInput', {
    'user_input': str,                          # User's message text
    'chat_thread_params': Optional[ChatThreadParams],  # Optional parameter overrides
    'metadata_filter': Optional[MetadataFilter], # Optional search filter
})
```

#### MetadataFilter

Filter criteria for knowledge base searches.

```python
MetadataFilter = TypedDict('MetadataFilter', {
    'field_name': str,       # Name of the metadata field to filter on
    'field_value': Any,      # Value to match against
    'comparison_type': str,  # Type of comparison ("equals", "contains", etc.)
}, total=False)
```

You can use these types to ensure type safety when working with the chat functions. For example:

```python
# Create chat thread parameters
params: ChatThreadParams = {
    "kb_ids": ["kb1"],
    "model": "gpt-4o-mini",
    "temperature": 0.2,
    "system_message": "You are a helpful assistant",
    "auto_query_model": "gpt-4o-mini",
    "auto_query_guidance": "",
    "target_output_length": "medium",
    "max_chat_history_tokens": 8000,
    "supp_id": ""
}

# Create chat response input
response_input: ChatResponseInput = {
    "user_input": "What is the capital of France?",
    "chat_thread_params": None,
    "metadata_filter": None
}
```
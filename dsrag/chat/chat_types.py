from typing import Optional, Union, Literal
from typing_extensions import TypedDict
from pydantic import BaseModel

class ChatThreadParams(TypedDict):
    kb_ids: Optional[list[str]]
    model: Optional[str]
    temperature: Optional[float]
    system_message: Optional[str]
    auto_query_model: Optional[str]
    auto_query_guidance: Optional[str]
    rse_params: Optional[dict]
    target_output_length: Optional[str]
    max_chat_history_tokens: Optional[int]

class ChatResponseOutput(TypedDict):
    response: str
    metadata: dict

class MetadataFilter(TypedDict):
    field: str
    operator: Literal['equals', 'not_equals', 'in', 'not_in', 'greater_than', 'less_than', 'greater_than_equals', 'less_than_equals']
    value: Union[str, int, float, list[str], list[int], list[float]]

class ChatResponseInput(BaseModel):
    user_input: str
    chat_thread_params: Optional[ChatThreadParams] = None
    metadata_filter: Optional[MetadataFilter] = None
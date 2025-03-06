import os
from pydantic import BaseModel
from typing import List
from ..utils.llm import get_response

SYSTEM_MESSAGE = """
You are a query generation system. Please generate one or more search queries (up to a maximum of {max_queries}) based on the provided user input. DO NOT generate the answer, just queries. You must specify the knowledge base you want to use for each query by providing the knowledge_base_id.

Each of the queries you generate will be used to search a knowledge base for information that can be used to respond to the user input. Make sure each query is specific enough to return relevant information. If multiple pieces of information would be useful, you should generate multiple queries, one for each specific piece of information needed.

{auto_query_guidance}

Here are the knowledge bases you can search:
{knowledge_base_descriptions}
""".strip()

class Query(BaseModel):
    query: str
    knowledge_base_id: str

class Queries(BaseModel):
    queries: List[Query]


def get_knowledge_base_descriptions_str(kb_info: list[dict]):
    kb_descriptions = []
    for kb in kb_info:
        kb_descriptions.append(f"kb_id: {kb['id']}\ndescription: {kb['title']} - {kb['description']}")
    return "\n\n".join(kb_descriptions)

def validate_queries(queries: List[Query], kb_info: list[dict]) -> List[dict]:
    """
    Validates and potentially modifies search queries based on knowledge base availability.
    
    Args:
        queries: List of Query objects to validate
        kb_info: List of available knowledge base information
        
    Returns:
        List of validated query dictionaries
    """
    valid_kb_ids = {kb["id"] for kb in kb_info}
    validated_queries = []
    
    for query in queries:
        if query.knowledge_base_id in valid_kb_ids:
            validated_queries.append({"query": query.query, "kb_id": query.knowledge_base_id})
        else:
            # If invalid KB ID is found
            if len(kb_info) == 1:
                # If only one KB exists, use that
                validated_queries.append({"query": query.query, "kb_id": kb_info[0]["id"]})
            else:
                # If multiple KBs exist, create a query for each KB
                for kb in kb_info:
                    validated_queries.append({"query": query.query, "kb_id": kb["id"]})

    return validated_queries

def get_search_queries(chat_messages: list[dict], kb_info: list[dict], auto_query_guidance: str = "", max_queries: int = 5, auto_query_model: str = "gpt-4o-mini") -> List[dict]:
    """
    Input:
    - chat_messages: list of dictionaries, where each dictionary has the keys "role" and "content". This should include the current user input as the final message.
    - kb_info: list of dictionaries, where each dictionary has the keys "id", "title", and "description". This should include information about the available knowledge bases.
    - auto_query_guidance: str, optional additional instructions for the auto_query system
    - max_queries: int, maximum number of queries to generate
    - auto_query_model: str, the model to use for generating queries

    Returns
    - queries: list of dictionaries, where each dictionary has the keys "query" and "knowledge_base_id"
    """
    knowledge_base_descriptions = get_knowledge_base_descriptions_str(kb_info)
    
    system_message = SYSTEM_MESSAGE.format(
        max_queries=max_queries,
        auto_query_guidance=auto_query_guidance,
        knowledge_base_descriptions=knowledge_base_descriptions
    )
    
    messages = [{"role": "system", "content": system_message}] + chat_messages

    # Try with initial model
    try:
        queries = get_response(
            messages=messages,
            model_name=auto_query_model,
            response_model=Queries,
            max_tokens=600,
            temperature=0.0
        )
    except:
        # Fallback to a stronger model
        fallback_model = "claude-3-5-sonnet-20241022" if "gpt" in auto_query_model else "gpt-4o"
        queries = get_response(
            messages=messages,
            model_name=fallback_model,
            response_model=Queries,
            max_tokens=600,
            temperature=0.0
        )

    # Validate and potentially modify the queries
    validated_queries = validate_queries(queries.queries[:max_queries], kb_info)[:max_queries]
    return validated_queries
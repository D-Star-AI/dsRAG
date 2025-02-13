import os
import instructor
from pydantic import BaseModel
from typing import List

SYSTEM_MESSAGE = """
You are a query generation system. Please generate one or more search queries (up to a maximum of {max_queries}) based on the provided user input. DO NOT generate the answer, just queries. You must specify the knowledge base you want to use for each query by providing the knowledge_base_id.

Each of the queries you generate will be used to search a knowledge base for information that can be used to respond to the user input. Make sure each query is specific enough to return relevant information. If multiple pieces of information would be useful, you should generate multiple queries, one for each specific piece of information needed.

{auto_query_guidance}

Here are the knowledge bases you can search:
{knowledge_base_descriptions}
""".strip()

OPENAI_MODELS = ["gpt-4o-mini", "gpt-4o", "gpt-4o-2024-08-06"]
ANTHROPIC_MODELS = ["claude-3-haiku-20240307", "claude-3-5-sonnet-20240620"]

class Query(BaseModel):
    query: str
    knowledge_base_id: str

class Queries(BaseModel):
    queries: List[Query]


def get_knowledge_base_descriptions_str(kb_info: list[dict]):
    kb_descriptions = []
    for kb in kb_info:
        kb_descriptions.append(f"kb_id: {kb['id']}\ndescription:{kb['description']}")
    return "\n\n".join(kb_descriptions)

def make_llm_call(chat_messages: list[dict], auto_query_model: str, max_queries: str, auto_query_guidance: str, knowledge_base_descriptions: str) -> Queries:
    if auto_query_model in OPENAI_MODELS:
        from openai import OpenAI
        base_url = os.getenv("DSRAG_OPENAI_BASE_URL", None)
        if base_url is not None:
            client = instructor.from_openai(OpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url=base_url))
        else:
            client = instructor.from_openai(OpenAI(api_key=os.environ["OPENAI_API_KEY"]))
        system_message = [{"role": "system", "content": SYSTEM_MESSAGE.format(max_queries=max_queries, auto_query_guidance=auto_query_guidance, knowledge_base_descriptions=knowledge_base_descriptions)}]
        chat_messages = system_message + chat_messages
        resp = client.chat.completions.create(
            model=auto_query_model,
            messages=chat_messages,
            max_tokens=600,
            temperature=0.0,
            response_model=Queries,
        )
        queries = resp.queries[:max_queries]
    elif auto_query_model in ANTHROPIC_MODELS:
        from anthropic import Anthropic
        base_url = os.getenv("DSRAG_ANTHROPIC_BASE_URL", None)
        if base_url is not None:
            client = instructor.from_anthropic(Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"], base_url=base_url))
        else:
            client = instructor.from_anthropic(Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"]))
        resp = client.messages.create(
            model=auto_query_model,
            max_tokens=600,
            temperature=0.0,
            system=SYSTEM_MESSAGE.format(max_queries=max_queries, auto_query_guidance=auto_query_guidance, knowledge_base_descriptions=knowledge_base_descriptions),
            messages=chat_messages,
            response_model=Queries,
        )
        queries = resp.queries[:max_queries]
    else:
        raise ValueError(f"Invalid auto_query_model: {auto_query_model}")
    
    return queries

def get_search_queries(chat_messages: list[dict], kb_info: list[dict], auto_query_guidance: str = "", max_queries: int = 5, auto_query_model: str = "gpt-4o-mini") -> List[dict]:
    """
    Input:
    - chat_messages: list of dictionaries, where each dictionary has the keys "role" and "content". This should include the current user input as the final message.
    - kb_info: list of dictionaries, where each dictionary has the keys "id" and "description". This should include information about the available knowledge bases.
    - auto_query_guidance: str, optional additional instructions for the auto_query system
    - max_queries: int, maximum number of queries to generate
    - auto_query_model: str, the model to use for generating queries

    Returns
    - queries: list of dictionaries, where each dictionary has the keys "query" and "knowledge_base_id"
    """

    # make sure the auto_query_model is valid
    try:
        assert auto_query_model in OPENAI_MODELS or auto_query_model in ANTHROPIC_MODELS
    except AssertionError:
        raise ValueError(f"Unsupported auto_query_model: {auto_query_model}")

    knowledge_base_descriptions = get_knowledge_base_descriptions_str(kb_info)

    # make the call to the LLM, retrying with a different model if the first one fails
    try:
        queries = make_llm_call(chat_messages, auto_query_model, max_queries, auto_query_guidance, knowledge_base_descriptions)
    except:
        # use a strong model if the initial model fails to minimize the chance of a second failure
        if auto_query_model in OPENAI_MODELS:
            auto_query_model = "claude-3-5-sonnet-20240620"
        else:
            auto_query_model = "gpt-4o"

        queries = make_llm_call(chat_messages, auto_query_model, max_queries, auto_query_guidance, knowledge_base_descriptions)

    return [{"query": query.query, "kb_id": query.knowledge_base_id} for query in queries]

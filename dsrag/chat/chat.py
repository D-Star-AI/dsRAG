import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__))))

from database.chat_thread.db import ChatThreadDB
from chat.chat_types import ChatThreadParams, MetadataFilter
from auto_query import get_search_queries
from utils.llm import get_response
import tiktoken
from datetime import datetime
import uuid


MAIN_SYSTEM_MESSAGE = """
INSTRUCTIONS AND GUIDANCE
{user_configurable_message}

CAPABILITIES
You are an AI assistant. Specifically, you are a large language model (LLM) that has been trained on a huge corpus of text from the internet and other sources. This means you have a wide range of knowledge about the world. You also have very strong reasoning and critical thinking skills. 

You have been given access to one or more searchable knowledge bases to help you better respond to user inputs. Here are the titles and descriptions of the knowledge bases you have access to:

{knowledge_base_descriptions}

Each time the user sends a message, you can choose to run a knowledge base search to find information that could be helpful in responding to the user. If you have gotten to the point where you are reading this, then that means you have already run a knowledge base search (or chosen not to) for the current user input, and now it's time to respond to the user. The results of this search can be found below in the "RELEVANT KNOWLEDGE" section.

Capabilities you DO NOT have:
- You DO NOT have access to the internet or any other real-time information sources aside from the specific knowledge bases listed above.
- You DO NOT have the ability to view images, videos, or any other non-textual information.
- You DO NOT have the ability to use any other software or tools aside from the specific knowledge bases listed above.
- As a language model, you don't have the ability to perform precise mathematical calculations. However, you are extremely good at "mental math," so you can get exact answers to simple calculations and approximate answers to more complex calculations, just like a highly talented human could.

RELEVANT KNOWLEDGE
The following pieces of text are search results from searches you chose to run. You can use them to help you write your response. But be careful: not all search results will be relevant, and sometimes you won't need to use any of them, so use your best judgement when deciding what to include in your response. Ignore any information here that is not highly relevant to the user's input.

{relevant_knowledge_str}

NEVER MAKE THINGS UP
If you do not have sufficient information to respond to the user, then you should either ask the user to clarify their question or tell the user you don't know the answer. DO NOT make things up.

{response_length_guidance}
""".strip()

SHORT_OUTPUT = """
RESPONSE LENGTH GUIDANCE
Please keep your response extremely short and concise. Answer in a single concise sentence whenever possible. If the question is more complex, then you may use up to a few sentences. IN NO CIRCUMSTANCE should your response be longer than one paragraph.
""".strip()

LONG_OUTPUT = """
RESPONSE LENGTH GUIDANCE
Please provide as much detail as possible in your response. If the question is very simple, then you may only need one paragraph, but most of the time you will need to use multiple paragraphs to provide a detailed response. Feel free to write up to a few pages if necessary. The most important thing is that you provide a detailed, thorough, and accurate response.
""".strip()

    
def create_new_chat_thread(chat_thread_params: ChatThreadParams, chat_thread_db: ChatThreadDB) -> str:
    thread_id = str(uuid.uuid4())
    chat_thread_params["thread_id"] = thread_id
    if "supp_id" not in chat_thread_params:
        chat_thread_params["supp_id"] = ""
    chat_thread_params = set_chat_thread_params(chat_thread_params)
    chat_thread_db.create_chat_thread(chat_thread_params=chat_thread_params)
    
    return thread_id

def get_knowledge_base_descriptions_str(kb_info: list[dict]):
    kb_descriptions = []
    for kb in kb_info:
        kb_descriptions.append(f"kb_id: {kb['id']}\ndescription:{kb['description']}")

    if len(kb_descriptions) == 0:
        return "No knowledge bases available."
    else:
        return "\n\n".join(kb_descriptions)

def format_relevant_knowledge_str(relevant_segments: list[dict]):
    relevant_knowledge_str = ""
    for segment in relevant_segments:
        relevant_knowledge_str += f"\n\n{segment['text']}"
    return relevant_knowledge_str.strip()

def count_tokens(text: str) -> int:
    """Count the number of tokens in a string."""
    encoding = tiktoken.encoding_for_model("gpt-4o")
    return len(encoding.encode(text))

def limit_chat_messages(chat_messages: list[dict], max_tokens: int = 8000) -> list[dict]:
    """Limit the total number of tokens in chat_messages."""
    total_tokens = 0
    limited_messages = []
    
    # Count tokens from the end (most recent messages)
    for message in reversed(chat_messages):
        message_tokens = count_tokens(message['content'])
        if total_tokens + message_tokens <= max_tokens:
            limited_messages.insert(0, message) # Insert at the beginning since we are iterating in reverse
            total_tokens += message_tokens
        else:
            break
    
    return limited_messages

def set_chat_thread_params(chat_thread_params: ChatThreadParams, kb_ids: list[str] = None, model: str = None, temperature: float = None, system_message: str = None, auto_query_model: str = None, auto_query_guidance: str = None, target_output_length: str = None, max_chat_history_tokens: int = None) -> ChatThreadParams:
    # set parameters - override if provided
    if kb_ids is not None:
        chat_thread_params['kb_ids'] = kb_ids
    elif 'kb_ids' not in chat_thread_params:
        chat_thread_params['kb_ids'] = []
    
    if model is not None:
        chat_thread_params['model'] = model
    elif 'model' not in chat_thread_params:
        chat_thread_params['model'] = "gpt-4o-mini"
    
    if temperature is not None:
        chat_thread_params['temperature'] = temperature
    elif 'temperature' not in chat_thread_params:
        chat_thread_params['temperature'] = 0.2

    if system_message is not None:
        chat_thread_params['system_message'] = system_message
    elif 'system_message' not in chat_thread_params:
        chat_thread_params['system_message'] = ""

    if auto_query_model is not None:
        chat_thread_params['auto_query_model'] = auto_query_model
    elif 'auto_query_model' not in chat_thread_params:
        chat_thread_params['auto_query_model'] = "gpt-4o-mini"

    if auto_query_guidance is not None:
        chat_thread_params['auto_query_guidance'] = auto_query_guidance
    elif 'auto_query_guidance' not in chat_thread_params:
        chat_thread_params['auto_query_guidance'] = ""

    if target_output_length is not None:
        chat_thread_params['target_output_length'] = target_output_length
    elif 'target_output_length' not in chat_thread_params:
        chat_thread_params['target_output_length'] = "medium"

    if max_chat_history_tokens is not None:
        chat_thread_params['max_chat_history_tokens'] = max_chat_history_tokens
    elif 'max_chat_history_tokens' not in chat_thread_params:
        chat_thread_params['max_chat_history_tokens'] = 8000

    return chat_thread_params

def get_chat_response(input: str, kbs: dict, chat_thread_params: ChatThreadParams, chat_thread_interactions: list[dict], metadata_filter: MetadataFilter = None) -> dict:
    # make note of the timestamp of the request
    request_timestamp = datetime.now().isoformat()

    kb_ids = chat_thread_params['kb_ids']

    # set parameters - override if provided
    chat_thread_params = set_chat_thread_params(chat_thread_params, kb_ids, chat_thread_params["model"], chat_thread_params["temperature"], chat_thread_params["system_message"], chat_thread_params["auto_query_model"], chat_thread_params["auto_query_guidance"], chat_thread_params["target_output_length"], chat_thread_params["max_chat_history_tokens"])

    kb_info = []
    for kb_id in chat_thread_params['kb_ids']:
        kb = kbs.get(kb_id)
        kb_info.append(
            {
                "id": kb_id,
                "description": kb.kb_metadata["description"],
            }
        )

    knowledge_base_descriptions = get_knowledge_base_descriptions_str(kb_info)

    # construct chat messages from interactions and input
    chat_messages = []
    for interaction in chat_thread_interactions:
        chat_messages.append({"role": "user", "content": interaction['user_input']['content']})
        chat_messages.append({"role": "assistant", "content": interaction['model_response']['content']})
    chat_messages.append({"role": "user", "content": input})

    # limit total number of tokens in chat_messages
    chat_messages = limit_chat_messages(chat_messages, chat_thread_params['max_chat_history_tokens'])

    if kb_info != []:
        # generate search queries
        try:
            search_queries = get_search_queries(chat_messages=chat_messages, kb_info=kb_info, auto_query_guidance=chat_thread_params['auto_query_guidance'], max_queries=5, auto_query_model=chat_thread_params['auto_query_model'])
        except:
            search_queries = []

        # group search queries by kb
        search_queries_by_kb = {}
        for search_query in search_queries:
            kb_id = search_query["kb_id"]
            query = search_query["query"]
            if kb_id not in search_queries_by_kb:
                search_queries_by_kb[kb_id] = []
            search_queries_by_kb[kb_id].append(query)

        print (f"Search queries by KB: {search_queries_by_kb}")

        # run searches
        search_results = {}
        for kb_id, queries in search_queries_by_kb.items():
            print (f"Running search for KB: {kb_id}")
            print (f"Queries: {queries}")
            kb = kbs.get(kb_id)
            search_results[kb_id] = kb.query(search_queries=queries, metadata_filter=metadata_filter)

        # unpack search results into a list and then combine them into a single string
        all_relevant_segments = []
        for kb_id, results in search_results.items():
            for result in results:
                result["kb_id"] = kb_id
                all_relevant_segments.append(result)
        relevant_knowledge_str = format_relevant_knowledge_str(all_relevant_segments)
    else:
        relevant_knowledge_str = "No knowledge bases provided, therefore no relevant knowledge to display."
        search_queries = []
        all_relevant_segments = []

    # deal with target_output_length
    if chat_thread_params['target_output_length'] == "short":
        response_length_guidance = SHORT_OUTPUT
    elif chat_thread_params['target_output_length'] == "medium":
        response_length_guidance = ""
    elif chat_thread_params['target_output_length'] == "long":
        response_length_guidance = LONG_OUTPUT
    else:
        response_length_guidance = ""
        print (f"ERROR: target_output_length {chat_thread_params['target_output_length']} not recognized. Using medium length output.")

    # format system message and add to chat messages
    formatted_system_message = MAIN_SYSTEM_MESSAGE.format(user_configurable_message=chat_thread_params['system_message'], knowledge_base_descriptions=knowledge_base_descriptions, relevant_knowledge_str=relevant_knowledge_str, response_length_guidance=response_length_guidance)
    chat_messages = [{"role": "system", "content": formatted_system_message}] + chat_messages

    # get LLM response
    llm_output = get_response(messages=chat_messages, model_name=chat_thread_params['model'], temperature=chat_thread_params['temperature'])

    # add interaction to chat thread
    interaction = {
        "user_input": {
            "content": input,
            "timestamp": request_timestamp
        },
        "model_response": {
            "content": llm_output,
            "timestamp": datetime.now().isoformat()
        },
        "search_queries": search_queries,
        "relevant_segments": all_relevant_segments
    }

    return interaction
import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__))))

from dsrag.database.chat_thread.db import ChatThreadDB
from dsrag.chat.chat_types import ChatThreadParams, MetadataFilter, ChatResponseInput
from dsrag.chat.auto_query import get_search_queries
from dsrag.chat.citations import format_sources_for_context, ResponseWithCitations
from dsrag.utils.llm import get_response
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
- You DO NOT have the ability to use any other software or tools aside from the specific knowledge bases listed above.
- As a language model, you don't have the ability to perform precise mathematical calculations. However, you are extremely good at "mental math," so you can get exact answers to simple calculations and approximate answers to more complex calculations, just like a highly talented human could.

RELEVANT KNOWLEDGE
The following pieces of text are search results from searches you chose to run. You can use them to help you write your response. But be careful: not all search results will be relevant, and sometimes you won't need to use any of them, so use your best judgement when deciding what to include in your response. Ignore any information here that is not highly relevant to the user's input.

{relevant_knowledge_str}

NEVER MAKE THINGS UP
If you do not have sufficient information to respond to the user, then you should either ask the user to clarify their question or tell the user you don't know the answer. DO NOT make things up.

CITATION FORMAT
When providing your response, you must cite your sources. For each piece of information you use, provide a citation that includes:
1. The document ID (doc_id)
2. The page number where the information was found (if available)
3. The relevant text that supports your response

Your response must be a valid ResponseWithCitations object. It must include these two fields:
1. response: Your complete response text
2. citations: An array of citation objects, each containing:
   - doc_id: The source document ID where the information used to generate the response was found
   - page_number: The page number where the information used to generate the response was found (or null if not available)
   - cited_text: The exact text containing the information used to generate the response

Note that an individual citation may only be associated with one page number. If the information used to generate the response was found on multiple pages, then you must provide multiple citations.

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
    print ("chat_thread_params: ", chat_thread_params)
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
    if max_tokens is None:
        max_tokens = 8000
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
    print ("chat_thread_params: ", chat_thread_params)
    # set parameters - override if provided
    if kb_ids is not None:
        chat_thread_params['kb_ids'] = kb_ids
    elif 'kb_ids' not in chat_thread_params or chat_thread_params['kb_ids'] is None:
        chat_thread_params['kb_ids'] = []
    
    if model is not None:
        chat_thread_params['model'] = model
    elif 'model' not in chat_thread_params or chat_thread_params['model'] is None:
        chat_thread_params['model'] = "gpt-4o-mini"
    
    if temperature is not None:
        chat_thread_params['temperature'] = temperature
    elif 'temperature' not in chat_thread_params or chat_thread_params['temperature'] is None:
        chat_thread_params['temperature'] = 0.2

    if system_message is not None:
        chat_thread_params['system_message'] = system_message
    elif 'system_message' not in chat_thread_params or chat_thread_params['system_message'] is None:
        chat_thread_params['system_message'] = ""

    if auto_query_model is not None:
        chat_thread_params['auto_query_model'] = auto_query_model
    elif 'auto_query_model' not in chat_thread_params or chat_thread_params['auto_query_model'] is None:
        chat_thread_params['auto_query_model'] = "gpt-4o-mini"

    if auto_query_guidance is not None:
        chat_thread_params['auto_query_guidance'] = auto_query_guidance
    elif 'auto_query_guidance' not in chat_thread_params or chat_thread_params['auto_query_guidance'] is None:
        chat_thread_params['auto_query_guidance'] = ""

    if target_output_length is not None:
        chat_thread_params['target_output_length'] = target_output_length
    elif 'target_output_length' not in chat_thread_params or chat_thread_params['target_output_length'] is None:
        chat_thread_params['target_output_length'] = "medium"

    if max_chat_history_tokens is not None:
        chat_thread_params['max_chat_history_tokens'] = max_chat_history_tokens
    elif 'max_chat_history_tokens' not in chat_thread_params or chat_thread_params['max_chat_history_tokens'] is None:
        chat_thread_params['max_chat_history_tokens'] = 8000

    return chat_thread_params

def get_chat_response(input: str, kbs: dict, chat_thread_params: ChatThreadParams, chat_thread_interactions: list[dict], metadata_filter: MetadataFilter = None) -> dict:
    # make note of the timestamp of the request
    request_timestamp = datetime.now().isoformat()

    kb_ids = chat_thread_params['kb_ids']

    # set parameters - override if provided
    chat_thread_params = set_chat_thread_params(
        chat_thread_params=chat_thread_params, 
        kb_ids=kb_ids,
        model=chat_thread_params.get("model"),
        temperature=chat_thread_params.get("temperature"),
        system_message=chat_thread_params.get("system_message"),
        auto_query_model=chat_thread_params.get("auto_query_model"),
        auto_query_guidance=chat_thread_params.get("auto_query_guidance"),
        target_output_length=chat_thread_params.get("target_output_length"),
        max_chat_history_tokens=chat_thread_params.get("max_chat_history_tokens")
    )

    kb_info = []
    for kb_id in chat_thread_params['kb_ids']:
        kb = kbs.get(kb_id)
        if not kb:
            continue
        kb_info.append(
            {
                "id": kb_id,
                "title": kb.kb_metadata.get("title", "No title available"),
                "description": kb.kb_metadata.get("description", "No description available"),
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

    if kb_info:
        # generate search queries
        try:
            search_queries = get_search_queries(chat_messages=chat_messages, kb_info=kb_info, auto_query_guidance=chat_thread_params['auto_query_guidance'], max_queries=5, auto_query_model=chat_thread_params['auto_query_model'])
        except Exception as e:
            print(f"Error generating search queries: {str(e)}")
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

        # unpack search results into a list
        formatted_relevant_segments = {}
        for kb_id, results in search_results.items():
            formatted_relevant_segments[kb_id] = []
            for result in results:
                result["kb_id"] = kb_id
                formatted_relevant_segments[kb_id].append(result)
        
        # Format search results into citation-friendly context
        relevant_knowledge_str = ""
        all_doc_ids = {}
        for kb_id, result in formatted_relevant_segments.items():
            kb = kbs.get(kb_id)
            # Format search results into citation-friendly context
            [relevant_knowledge_str, doc_ids] = format_sources_for_context(
                search_results=result,
                kb_id=kb_id,
                file_system=kb.file_system
            )
            for doc_id in doc_ids:
                all_doc_ids[doc_id] = kb_id
    else:
        relevant_knowledge_str = "No knowledge bases provided, therefore no relevant knowledge to display."
        search_queries = []

    # DEBUG
    print (f"Relevant knowledge str:\n{relevant_knowledge_str}")

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
    formatted_system_message = MAIN_SYSTEM_MESSAGE.format(
        user_configurable_message=chat_thread_params['system_message'],
        knowledge_base_descriptions=knowledge_base_descriptions,
        relevant_knowledge_str=relevant_knowledge_str,
        response_length_guidance=response_length_guidance
    )
    chat_messages = [{"role": "system", "content": formatted_system_message}] + chat_messages

    # get LLM response with structured output
    response = get_response(
        messages=chat_messages,
        model_name=chat_thread_params['model'],
        temperature=chat_thread_params['temperature'],
        max_tokens=4000,
        response_model=ResponseWithCitations
    )
    
    citations = response.citations
    # For each citation, add the kb_id to the citation
    formatted_citations = []
    for citation in citations:
        citation = citation.model_dump()
        # Add error handling for unknown doc_ids
        if citation["doc_id"] in all_doc_ids:
            citation["kb_id"] = all_doc_ids[citation["doc_id"]]
        else:
            # Skip citations with unknown doc_ids
            continue
        formatted_citations.append(citation)
        
    all_relevant_segments = []
    for kb_id, results in formatted_relevant_segments.items():
        for result in results:
            result["kb_id"] = kb_id
            all_relevant_segments.append(result)

    # add interaction to chat thread
    interaction = {
        "user_input": {
            "content": input,
            "timestamp": request_timestamp
        },
        "model_response": {
            "content": response.response,
            "citations": formatted_citations,
            "timestamp": datetime.now().isoformat()
        },
        "search_queries": search_queries,
        "relevant_segments": all_relevant_segments
    }

    return interaction

def get_filenames_and_types(interaction: dict, kbs: dict):
    ranked_results = interaction.get("relevant_segments", [])
    formatted_results = []
    for result in ranked_results:
        kb_id = result["kb_id"]
        doc_id = result["doc_id"]

        kb = kbs.get(kb_id, None)
        if kb is None:
            # Should never happen
            file_name = ""
            document_type = "text"
        else:
            document = kb.chunk_db.get_document(doc_id, include_content=False)
            if document is None:
                file_name = ""
                document_type = "text"
            else:
                file_name = document.get("metadata", {}).get("file_name", "")
                document_type = document.get("metadata", {}).get("document_type", "text")
        
        result["file_name"] = file_name
        result["document_type"] = document_type
        formatted_results.append(result)
    
    interaction["relevant_segments"] = formatted_results
    return interaction

def get_chat_thread_response(thread_id: str, get_response_input: ChatResponseInput, chat_thread_db: ChatThreadDB, knowledge_bases: dict):
    """
    - knowledge_bases: dict of knowledge base id to knowledge base object
    """
    user_input = get_response_input.user_input
    chat_thread_params_override = get_response_input.chat_thread_params
    metadata_filter = get_response_input.metadata_filter
    thread = chat_thread_db.get_chat_thread(thread_id)
    if chat_thread_params_override is not None:
        chat_thread_params = chat_thread_params_override
    else:
        chat_thread_params = thread["params"]
    chat_thread_interactions = thread['interactions']
    # Get the kbs from the chat_thread_params
    missing_kbs = [kb_id for kb_id in chat_thread_params["kb_ids"] if kb_id not in knowledge_bases]
    if missing_kbs:
        return {"message": f"Missing knowledge bases: {', '.join(missing_kbs)}"}

    kbs = {kb_id: knowledge_bases[kb_id] for kb_id in chat_thread_params["kb_ids"]}

    # Get the chat response
    interaction = get_chat_response(user_input, kbs, chat_thread_params, chat_thread_interactions, metadata_filter)
    formatted_interaction = get_filenames_and_types(interaction, knowledge_bases)

    # Add this interaction to the chat thread db
    response = chat_thread_db.add_interaction(thread_id, interaction)
    message_id = response["message_id"]
    formatted_interaction["message_id"] = message_id

    return formatted_interaction
from dsrag.database.chat_thread.db import ChatThreadDB
from dsrag.chat.chat_types import ChatThreadParams, MetadataFilter, ChatResponseInput, ExaSearchResult
from dsrag.chat.auto_query import get_search_queries, get_exa_search_queries
from dsrag.chat.citations import format_sources_for_context, format_exa_search_results, ResponseWithCitations
from dsrag.utils.llm import get_response
import tiktoken
from datetime import datetime
import uuid
from typing import Optional
import os

MAIN_SYSTEM_MESSAGE = """
INSTRUCTIONS AND GUIDANCE
{user_configurable_message}

CAPABILITIES
You are an AI assistant. Specifically, you are a large language model (LLM) that has been trained on a huge corpus of text from the internet and other sources. This means you have a wide range of knowledge about the world. You also have very strong reasoning and critical thinking skills. 

You have been given access to one or more searchable knowledge bases to help you better respond to user inputs. Here are the titles and descriptions of the knowledge bases you have access to:

{knowledge_base_descriptions}

Each time the user sends a message, you can choose to run a knowledge base search to find information that could be helpful in responding to the user. If you have gotten to the point where you are reading this, then that means you have already run a knowledge base search (or chosen not to) for the current user input, and now it's time to respond to the user. The results of this search can be found below in the "RELEVANT KNOWLEDGE" section.

Capabilities you DO NOT have:
- You DO NOT have access to the internet or any other real-time information sources aside from the specific knowledge bases listed above and the web search results below (if provided).
    - There may be content from websites included in the "RELEVANT KNOWLEDGE" section, but you DO NOT have the ability to access the internet or any other real-time information sources.
- You DO NOT have the ability to use any other software or tools aside from the specific knowledge bases listed above and the web search results below (if provided).
- As a language model, you don't have the ability to perform precise mathematical calculations. However, you are extremely good at "mental math," so you can get exact answers to simple calculations and approximate answers to more complex calculations, just like a highly talented human could.

RELEVANT KNOWLEDGE
The following pieces of text are search results from searches you chose to run. You can use them to help you write your response. But be careful: not all search results will be relevant, and sometimes you won't need to use any of them, so use your best judgement when deciding what to include in your response. Ignore any information here that is not highly relevant to the user's input.

{relevant_knowledge_str}

NEVER MAKE THINGS UP
If you do not have sufficient information to respond to the user, then you should either ask the user to clarify their question or tell the user you don't know the answer. DO NOT make things up.

CITATION FORMAT
When providing your response, you must cite your sources. For each piece of information you use, provide a citation that includes:
1. The document ID (doc_id) or URL (url) where the information was found
2. The page number where the information was found (if available)
3. The relevant text or content that supports your response

The doc_id and page_number will be in the following format:
<doc_id: some_random_long_id>
<page_N>
Text content from the page
</page_N>
</doc_id: some_random_long_id>
The page number is N in the above format.

Web search results will be in the following format:
<id: url_of_the_web_page>
Title: title of the web page
URL: url of the web page
Content: text content of the web page
</id: url_of_the_web_page>

Your response must be a valid ResponseWithCitations object. It must include these two fields:
1. response: Your complete response text
2. citations: An array of citation objects, each containing:
   - doc_id: The source document ID where the information used to generate the response was found (if available)
   - url: The URL where the information used to generate the response was found (if available)
   - title: The title of the web page where the information used to generate the response was found (if available)
   - page_number: The page number where the information used to generate the response was found (or null if not available)
   - cited_text: The exact text containing the information used to generate the response

Note that an individual citation may only be associated with one page number (if citing a document) or one URL (if citing a website). If the information used to generate the response was found on multiple pages or URLs, then you must provide multiple citations.

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
    """Create a new chat thread in the database.

    Args:
        chat_thread_params (ChatThreadParams): Parameters for the chat thread. Example:
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
        chat_thread_db (ChatThreadDB): Database instance for storing chat threads.

    Returns:
        str: Unique identifier for the created chat thread.
    """
    thread_id = str(uuid.uuid4())
    chat_thread_params["thread_id"] = thread_id
    if "supp_id" not in chat_thread_params:
        chat_thread_params["supp_id"] = ""
    chat_thread_params = _set_chat_thread_params(chat_thread_params)
    print ("chat_thread_params: ", chat_thread_params)
    chat_thread_db.create_chat_thread(chat_thread_params=chat_thread_params)
    return thread_id

def get_knowledge_base_descriptions_str(kb_info: list[dict]):
    kb_descriptions = []
    for kb in kb_info:
        kb_descriptions.append(f"kb_id: {kb['id']}\ndescription:{kb['description']}")

    if len(kb_descriptions) == 0:
        return "No knowledge bases available at this time. Please respond using only your built-in knowledge or the results from the web search if provided, but be sure not to make things up. Keep in mind that you may still have access to web search results to use for your response."
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
        if message["content"] is None:
            continue
        message_tokens = count_tokens(message['content'])
        if total_tokens + message_tokens <= max_tokens:
            limited_messages.insert(0, message) # Insert at the beginning since we are iterating in reverse
            total_tokens += message_tokens
        else:
            break
    
    return limited_messages

def _set_chat_thread_params(
    chat_thread_params: ChatThreadParams,
    kb_ids: list[str] = None,
    model: str = None,
    temperature: float = None,
    system_message: str = None,
    auto_query_model: str = None,
    auto_query_guidance: str = None,
    target_output_length: str = None,
    max_chat_history_tokens: int = None,
    rse_params: dict = None
) -> ChatThreadParams:
    """Set and validate chat thread parameters.

    Internal method to ensure all required parameters are set with appropriate defaults.

    Args:
        chat_thread_params (ChatThreadParams): Base parameters dictionary
        kb_ids (list[str], optional): Knowledge base IDs. Defaults to None.
        model (str, optional): LLM model name. Defaults to None.
        temperature (float, optional): LLM temperature. Defaults to None.
        system_message (str, optional): System message for LLM. Defaults to None.
        auto_query_model (str, optional): Model for query generation. Defaults to None.
        auto_query_guidance (str, optional): Guidance for query generation. Defaults to None.
        target_output_length (str, optional): Target response length. Defaults to None.
        max_chat_history_tokens (int, optional): Maximum chat history tokens. Defaults to None.
        rse_params (dict, optional): Parameters for response evaluation. Defaults to None.

    Returns:
        ChatThreadParams: Updated parameters with defaults filled in.
    """
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

    if rse_params is not None:
        chat_thread_params['rse_params'] = rse_params
    elif 'rse_params' not in chat_thread_params or chat_thread_params['rse_params'] is None:
        chat_thread_params['rse_params'] = {}

    return chat_thread_params

async def _run_exa_search(
    exa_search_queries: list[str],
    exa_include_domains: Optional[list[str]] = None
) -> list[dict]:
    """Run an EXA search."""
    import asyncio
    from exa_py import Exa
    from exa_py.api import SearchResponse

    async def search_single_query(query: str) -> list[SearchResponse]:
        print ("\n")
        print ("EXA search query: ", query)
        print ("\n")
        exa = Exa(api_key=os.getenv("EXA_API_KEY"))
        return exa.search_and_contents(
            query,
            text=True
        )

    # Run all searches in parallel
    tasks = [search_single_query(query.query) for query in exa_search_queries]
    results = await asyncio.gather(*tasks)
    
    return results
    

async def _prepare_chat_context(
    input: str,
    kbs: dict,
    chat_thread_params: ChatThreadParams,
    chat_thread_interactions: list[dict],
    metadata_filter: MetadataFilter = None,
    run_exa_search: bool = False,
    exa_search_query: str = None,
    exa_include_domains: Optional[list[str]] = None
) -> tuple:
    """Prepare the chat context for generating a response.
    
    This function handles the common setup work needed for both streaming and non-streaming responses.
    It runs EXA search and knowledge base searches in parallel when both are requested.

    Args:
        input (str): User input text to respond to.
        kbs (dict): Dictionary of knowledge base instances keyed by ID.
        chat_thread_params (ChatThreadParams): Chat thread configuration parameters.
        chat_thread_interactions (list[dict]): Previous chat interactions.
        metadata_filter (MetadataFilter, optional): Filter for knowledge base search. Defaults to None.
        run_exa_search (bool, optional): Whether to run EXA search. Defaults to False.
        exa_search_query (str, optional): Query for EXA search. Defaults to None.
        exa_include_domains (list[str], optional): Domains to include in EXA search. Defaults to None.

    Returns:
        tuple: A tuple containing:
            - request_timestamp (str): Timestamp of the request
            - chat_messages (list): Formatted chat messages including system message
            - search_queries (list): Generated search queries
            - all_relevant_segments (list): All relevant segments from search
            - formatted_relevant_segments (dict): Relevant segments organized by KB
            - all_doc_ids (dict): Mapping of doc_ids to kb_ids
            - chat_thread_params (ChatThreadParams): Updated chat thread parameters
    """
    import asyncio

    # make note of the timestamp of the request
    request_timestamp = datetime.now().isoformat()

    kb_ids = chat_thread_params['kb_ids']

    # set parameters - override if provided
    chat_thread_params = _set_chat_thread_params(
        chat_thread_params=chat_thread_params, 
        kb_ids=kb_ids,
        model=chat_thread_params.get("model"),
        temperature=chat_thread_params.get("temperature"),
        system_message=chat_thread_params.get("system_message"),
        auto_query_model=chat_thread_params.get("auto_query_model"),
        auto_query_guidance=chat_thread_params.get("auto_query_guidance"),
        target_output_length=chat_thread_params.get("target_output_length"),
        max_chat_history_tokens=chat_thread_params.get("max_chat_history_tokens"),
        rse_params=chat_thread_params.get("rse_params")
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

    formatted_relevant_segments = {}
    all_doc_ids = {}
    all_relevant_segments = []
    search_queries = []
    relevant_knowledge_str = ""

    async def run_kb_search():
        nonlocal search_queries, formatted_relevant_segments, all_doc_ids, all_relevant_segments, relevant_knowledge_str
        
        if not kb_info:
            return "No knowledge bases provided, therefore no relevant knowledge to display. You may still have web search results to display."

        # generate search queries
        try:
            search_queries = get_search_queries(
                chat_messages=chat_messages, 
                kb_info=kb_info, 
                auto_query_guidance=chat_thread_params['auto_query_guidance'], 
                max_queries=5, 
                auto_query_model=chat_thread_params['auto_query_model']
            )
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

        print(f"Search queries by KB: {search_queries_by_kb}")

        rse_params = chat_thread_params['rse_params']

        # run searches in parallel
        async def search_kb(kb_id: str, queries: list[str]):
            kb = kbs.get(kb_id)
            return kb_id, kb.query(search_queries=queries, rse_params=rse_params, metadata_filter=metadata_filter)

        search_tasks = [
            search_kb(kb_id, queries) 
            for kb_id, queries in search_queries_by_kb.items()
        ]
        
        if search_tasks:
            search_results = dict(await asyncio.gather(*search_tasks))
        else:
            search_results = {}

        # unpack search results into a list
        for kb_id, results in search_results.items():
            formatted_relevant_segments[kb_id] = []
            for result in results:
                result["kb_id"] = kb_id
                formatted_relevant_segments[kb_id].append(result)
        
        # Format search results into citation-friendly context
        kb_knowledge_str = ""
        for kb_id, result in formatted_relevant_segments.items():
            kb = kbs.get(kb_id)
            # Format search results into citation-friendly context
            [formatted_knowledge, doc_ids] = format_sources_for_context(
                search_results=result,
                kb_id=kb_id,
                file_system=kb.file_system
            )
            kb_knowledge_str += formatted_knowledge
            for doc_id in doc_ids:
                all_doc_ids[doc_id] = kb_id

        # Prepare all relevant segments list
        for kb_id, results in formatted_relevant_segments.items():
            for result in results:
                result["kb_id"] = kb_id
                all_relevant_segments.append(result)

        return kb_knowledge_str

    # Run both searches in parallel if EXA search is requested
    relevant_web_search_segments = []
    if run_exa_search:
        exa_search_queries = get_exa_search_queries(
            chat_messages=chat_messages, 
            auto_query_guidance=chat_thread_params['auto_query_guidance'], 
            max_queries=3, 
            auto_query_model=chat_thread_params['auto_query_model']
        )
        kb_search_task = run_kb_search()
        exa_search_task = _run_exa_search(exa_search_queries, exa_include_domains)
        kb_knowledge_str, exa_results = await asyncio.gather(kb_search_task, exa_search_task)
        
        formatted_exa_results = format_exa_search_results(exa_results)
        
        """for results in exa_results:
            for result in results.results:
                relevant_web_search_segments.append(result.__dict__)"""
        
        # Combine knowledge from both sources
        relevant_knowledge_str = kb_knowledge_str
        if exa_results:
            # Format EXA results and append to relevant knowledge
            # TODO: Implement proper formatting for EXA results
            relevant_knowledge_str += "\n\nWeb Search Results:\n" + str(formatted_exa_results)
    else:
        # Just run KB search
        relevant_knowledge_str = await run_kb_search()

    # deal with target_output_length
    if chat_thread_params['target_output_length'] == "short":
        response_length_guidance = SHORT_OUTPUT
    elif chat_thread_params['target_output_length'] == "medium":
        response_length_guidance = ""
    elif chat_thread_params['target_output_length'] == "long":
        response_length_guidance = LONG_OUTPUT
    else:
        response_length_guidance = ""
        print(f"ERROR: target_output_length {chat_thread_params['target_output_length']} not recognized. Using medium length output.")

    # format system message and add to chat messages
    formatted_system_message = MAIN_SYSTEM_MESSAGE.format(
        user_configurable_message=chat_thread_params['system_message'],
        knowledge_base_descriptions=knowledge_base_descriptions,
        relevant_knowledge_str=relevant_knowledge_str,
        response_length_guidance=response_length_guidance
    )
    print ("formatted_system_message", formatted_system_message)
    chat_messages = [{"role": "system", "content": formatted_system_message}] + chat_messages
    
    return (
        request_timestamp,
        chat_messages,
        search_queries,
        all_relevant_segments,
        relevant_web_search_segments,
        formatted_relevant_segments,
        all_doc_ids,
        chat_thread_params
    )

async def _get_chat_response_streaming(
    input: str,
    kbs: dict,
    chat_thread_params: ChatThreadParams,
    chat_thread_interactions: list[dict],
    metadata_filter: MetadataFilter = None,
    run_exa_search: bool = False,
    exa_search_query: str = None,
    exa_include_domains: Optional[list[str]] = None
):
    """Generate a streaming response to a chat input using knowledge base search.
    
    This function is a generator that yields partial responses.

    Args:
        input (str): User input text to respond to.
        kbs (dict): Dictionary of knowledge base instances keyed by ID.
        chat_thread_params (ChatThreadParams): Chat thread configuration parameters.
        chat_thread_interactions (list[dict]): Previous chat interactions.
        metadata_filter (MetadataFilter, optional): Filter for knowledge base search.
        run_exa_search (bool, optional): Whether to run EXA search. Defaults to False.
        exa_search_query (str, optional): Query for EXA search. Defaults to None.
        exa_include_domains (list[str], optional): Domains to include in EXA search. Defaults to None.

    Yields:
        dict: Partial interaction dictionaries during generation.
        
    Returns:
        dict: Final complete interaction when generation is complete.
    """
    # Import here to avoid circular imports
    from dsrag.chat.citations import PartialResponseWithCitations
    
    # First, await the async context preparation
    context = await _prepare_chat_context(
        input, 
        kbs, 
        chat_thread_params, 
        chat_thread_interactions, 
        metadata_filter,
        run_exa_search,
        exa_search_query,
        exa_include_domains
    )
    
    (
        request_timestamp,
        chat_messages,
        search_queries,
        all_relevant_segments,
        relevant_web_search_segments,
        formatted_relevant_segments,
        all_doc_ids,
        chat_thread_params
    ) = context
    
    # Create a response stream
    response_stream = get_response(
        messages=chat_messages,
        model_name=chat_thread_params['model'],
        temperature=chat_thread_params['temperature'],
        max_tokens=4000,
        response_model=ResponseWithCitations,
        stream=True
    )
    
    # Create a base interaction that we'll update with each partial response
    interaction_base = {
        "user_input": {
            "content": input,
            "timestamp": request_timestamp
        },
        "search_queries": search_queries,
        "relevant_segments": [],
        "relevant_web_search_segments": relevant_web_search_segments
    }
    
    # Keep track of the final response for later saving
    final_response = None
    final_citations = []
    
    # Change this section to use regular for loop since response_stream is a regular generator
    for partial_response in response_stream:
        # Create a streaming response with what we have so far
        current_interaction = interaction_base.copy()
        
        # Store the latest partial response for final saving
        final_response = partial_response
        
        # Format the partial response for streaming
        content = ""
        if hasattr(partial_response, 'response'):
            content = partial_response.response
        elif hasattr(partial_response, 'model_fields_set') and 'response' in partial_response.model_fields_set:
            content = getattr(partial_response, 'response', "")
        elif isinstance(partial_response, dict) and 'response' in partial_response:
            content = partial_response['response']
            
        current_interaction["model_response"] = {
            "content": content,
            "citations": [],
            "timestamp": datetime.now().isoformat()
        }
        
        # Handle citations
        citations_list = []
        if hasattr(partial_response, 'citations') and partial_response.citations:
            citations_list = partial_response.citations
        elif hasattr(partial_response, 'model_fields_set') and 'citations' in partial_response.model_fields_set:
            citations_list = getattr(partial_response, 'citations', [])
        elif isinstance(partial_response, dict) and 'citations' in partial_response:
            citations_list = partial_response['citations']
            
        print ("\n\n")
        print ("citations_list", citations_list)
        print ("\n\n")
        if citations_list:
            formatted_stream_citations = []
            for citation in citations_list:
                citation_dict = (citation.model_dump() if hasattr(citation, 'model_dump') 
                               else citation.dict() if hasattr(citation, 'dict')
                               else citation if isinstance(citation, dict)
                               else None)
                
                if citation_dict and citation_dict.get("doc_id") in all_doc_ids:
                    citation_dict["kb_id"] = all_doc_ids[citation_dict["doc_id"]]
                    formatted_stream_citations.append(citation_dict)
                elif citation_dict and citation_dict.get("url"):
                    citation_dict["kb_id"] = "web_search"
                    formatted_stream_citations.append(citation_dict)
                    
            current_interaction["model_response"]["citations"] = formatted_stream_citations
            final_citations = formatted_stream_citations
        
        yield current_interaction
    
    # Prepare final interaction with all relevant segments
    final_interaction = {
        "user_input": {
            "content": input,
            "timestamp": request_timestamp
        },
        "model_response": {
            "content": (final_response.response if hasattr(final_response, 'response')
                       else getattr(final_response, 'response', "") if hasattr(final_response, 'model_fields_set')
                       else final_response['response'] if isinstance(final_response, dict)
                       else ""),
            "citations": final_citations,
            "timestamp": datetime.now().isoformat()
        },
        "search_queries": search_queries,
        "relevant_segments": all_relevant_segments,
        "relevant_web_search_segments": relevant_web_search_segments
    }
    
    yield final_interaction

def _get_chat_response(
    input: str,
    kbs: dict,
    chat_thread_params: ChatThreadParams,
    chat_thread_interactions: list[dict],
    metadata_filter: MetadataFilter = None,
    run_exa_search: bool = False,
    exa_search_query: str = None,
    exa_include_domains: Optional[list[str]] = None
) -> dict:
    """Generate a response to a chat input using knowledge base search.

    Args:
        input (str): User input text to respond to.
        kbs (dict): Dictionary of knowledge base instances keyed by ID.
        chat_thread_params (ChatThreadParams): Chat thread configuration parameters.
        chat_thread_interactions (list[dict]): Previous chat interactions.
        metadata_filter (MetadataFilter, optional): Filter for knowledge base search. Defaults to None.
        run_exa_search (bool, optional): Whether to run EXA search. Defaults to False.
        exa_search_query (str, optional): Query for EXA search. Defaults to None.
        exa_include_domains (list[str], optional): Domains to include in EXA search. Defaults to None.

    Returns:
        dict: Interaction dictionary containing:
            - user_input (dict): User message with content and timestamp
            - model_response (dict): Model response with content and timestamp
            - search_queries (list): Generated search queries
            - relevant_segments (list): Retrieved relevant segments
    """
    # Prepare context
    (
        request_timestamp,
        chat_messages,
        search_queries,
        all_relevant_segments,
        formatted_relevant_segments,
        all_doc_ids,
        chat_thread_params
    ) = _prepare_chat_context(
        input, 
        kbs, 
        chat_thread_params, 
        chat_thread_interactions, 
        metadata_filter,
        run_exa_search,
        exa_search_query,
        exa_include_domains
    )
    
    # Non-streaming case - get complete response
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

def _get_filenames_and_types(interaction: dict, kbs: dict) -> dict:
    """Add file names and types to relevant segments.

    Internal method to enrich search results with file metadata.

    Args:
        interaction (dict): Chat interaction containing relevant segments.
        kbs (dict): Dictionary of knowledge base instances.

    Returns:
        dict: Updated interaction with file names and types added to segments.
    """
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

async def get_chat_thread_response_streaming(thread_id: str, get_response_input: ChatResponseInput, chat_thread_db: ChatThreadDB, knowledge_bases: dict):
    """Get a streaming response for a chat thread using knowledge base search.

    Args:
        thread_id (str): Unique identifier for the chat thread.
        get_response_input (ChatResponseInput): Input parameters containing user input and optional overrides.
        chat_thread_db (ChatThreadDB): Database instance for chat threads.
        knowledge_bases (dict): Dictionary mapping knowledge base IDs to instances.

    Yields:
        dict: Partial formatted interactions during response generation.
    """
    print("STREAMING RESPONSE")
    user_input = get_response_input.user_input
    chat_thread_params_override = get_response_input.chat_thread_params
    metadata_filter = get_response_input.metadata_filter
    run_exa_search = get_response_input.run_exa_search
    exa_search_query = get_response_input.exa_search_query
    exa_include_domains = get_response_input.exa_include_domains
    
    thread = chat_thread_db.get_chat_thread(thread_id)
    
    if chat_thread_params_override is not None:
        # Complete override of chat thread params
        chat_thread_params = chat_thread_params_override
    else:
        chat_thread_params = thread["params"]
    
    chat_thread_interactions = thread['interactions']
    
    # Get the kbs from the chat_thread_params
    missing_kbs = [kb_id for kb_id in chat_thread_params["kb_ids"] if kb_id not in knowledge_bases]
    if missing_kbs:
        yield {"message": f"Missing knowledge bases: {', '.join(missing_kbs)}"}
        return

    kbs = {kb_id: knowledge_bases[kb_id] for kb_id in chat_thread_params["kb_ids"]}

    print ("Getting response generator")
    # Get a generator for chat responses using the streaming function
    response_generator = _get_chat_response_streaming(
        user_input, 
        kbs, 
        chat_thread_params, 
        chat_thread_interactions, 
        metadata_filter,
        run_exa_search,
        exa_search_query,
        exa_include_domains
    )
    
    print ("\n\n")
    print (type(response_generator))
    print ("\n\n")
    
    try:
        print ("Getting initial response")
        # Get the first response
        try:
            initial_response = await anext(response_generator)
        except Exception as e:
            print ("No initial response", e)
            return
        print ("Initial response received")
        
        # Set initial status to "pending"
        if "model_response" in initial_response:
            initial_response["model_response"]["status"] = "pending"
        
        formatted_initial = _get_filenames_and_types(initial_response, knowledge_bases)
        print ("Adding initial response to db")
        db_response = chat_thread_db.add_interaction(thread_id, initial_response)
        print ("Initial response added to db")
        message_id = db_response["message_id"]
        formatted_initial["message_id"] = message_id
        
        yield formatted_initial
        
        # Process remaining responses
        async for partial_response in response_generator:
            if "model_response" in partial_response:
                partial_response["model_response"]["status"] = "streaming"
            
            formatted_partial = _get_filenames_and_types(partial_response, knowledge_bases)
            formatted_partial["message_id"] = message_id
            
            chat_thread_db.update_interaction(
                thread_id,
                message_id,
                {
                    "model_response": partial_response["model_response"]
                }
            )
            
            yield formatted_partial
        
        # Final status update
        final_update = {
            "model_response": {
                "status": "finished",
                "content": partial_response["model_response"]["content"],
                "timestamp": partial_response["model_response"]["timestamp"]
            }
        }
        
        if "citations" in partial_response["model_response"]:
            final_update["model_response"]["citations"] = partial_response["model_response"]["citations"]
            
        chat_thread_db.update_interaction(thread_id, message_id, final_update)
            
    except StopAsyncIteration:
        # Handle case where generator is empty
        pass

async def get_chat_thread_response_non_streaming(thread_id: str, get_response_input: ChatResponseInput, chat_thread_db: ChatThreadDB, knowledge_bases: dict) -> dict:
    """Get a non-streaming response for a chat thread using knowledge base search.

    Args:
        thread_id (str): Unique identifier for the chat thread.
        get_response_input (ChatResponseInput): Input parameters containing user input and optional overrides.
        chat_thread_db (ChatThreadDB): Database instance for chat threads.
        knowledge_bases (dict): Dictionary mapping knowledge base IDs to instances.

    Returns:
        dict: Formatted interaction containing complete response and metadata.
    """
    print("NON-STREAMING RESPONSE")
    user_input = get_response_input.user_input
    chat_thread_params_override = get_response_input.chat_thread_params
    metadata_filter = get_response_input.metadata_filter
    run_exa_search = get_response_input.run_exa_search
    exa_search_query = get_response_input.exa_search_query
    exa_include_domains = get_response_input.exa_include_domains
    
    thread = chat_thread_db.get_chat_thread(thread_id)
    
    if chat_thread_params_override is not None:
        # Complete override of chat thread params
        chat_thread_params = chat_thread_params_override
    else:
        chat_thread_params = thread["params"]
    
    chat_thread_interactions = thread['interactions']
    
    # Get the kbs from the chat_thread_params
    missing_kbs = [kb_id for kb_id in chat_thread_params["kb_ids"] if kb_id not in knowledge_bases]
    if missing_kbs:
        return {"message": f"Missing knowledge bases: {', '.join(missing_kbs)}"}

    kbs = {kb_id: knowledge_bases[kb_id] for kb_id in chat_thread_params["kb_ids"]}

    # Non-streaming case - get complete response with the non-streaming function
    interaction = _get_chat_response(
        user_input, 
        kbs, 
        chat_thread_params, 
        chat_thread_interactions, 
        metadata_filter,
        run_exa_search,
        exa_search_query,
        exa_include_domains
    )
    
    # Set status to "finished" for non-streaming responses
    if "model_response" in interaction:
        interaction["model_response"]["status"] = "finished"
    
    formatted_interaction = _get_filenames_and_types(interaction, knowledge_bases)

    # Add this interaction to the chat thread db
    response = chat_thread_db.add_interaction(thread_id, interaction)
    message_id = response["message_id"]
    formatted_interaction["message_id"] = message_id

    return formatted_interaction


async def get_chat_thread_response(thread_id: str, get_response_input: ChatResponseInput, chat_thread_db: ChatThreadDB, knowledge_bases: dict, stream: bool = False):
    """Get a response for a chat thread using knowledge base search.

    This function is a router that calls the appropriate implementation based on the stream parameter.

    Args:
        thread_id (str): Unique identifier for the chat thread.
        get_response_input (ChatResponseInput): Input parameters containing:
            - user_input (str): User's message text
            - chat_thread_params (Optional[ChatThreadParams]): Optional parameter overrides
            - metadata_filter (Optional[MetadataFilter]): Optional search filter
            - run_exa_search (bool): Whether to run EXA search
            - exa_search_query (str): Query for EXA search
            - exa_include_domains (list[str]): Domains to include in EXA search
        chat_thread_db (ChatThreadDB): Database instance for chat threads.
        knowledge_bases (dict): Dictionary mapping knowledge base IDs to instances.
        stream (bool, optional): Whether to stream the response. Defaults to False.

    Returns:
        If stream=False:
            dict: Formatted interaction containing:
                - user_input (dict): User message with content and timestamp
                - model_response (dict): Model response with content and timestamp
                - search_queries (list): Generated search queries
                - relevant_segments (list): Retrieved relevant segments with file names and types
                - message_id (str): Message ID for the saved interaction
                - message (str, optional): Error message if something went wrong
        
        If stream=True:
            Iterator: Yields partial response objects during generation
    """
    print("ROUTER FUNCTION", "stream =", stream)
    if stream:
        async_gen = get_chat_thread_response_streaming(thread_id, get_response_input, chat_thread_db, knowledge_bases)
        async for response in async_gen:
            print ("response", response)
            yield response
    else:
        # For non-streaming, yield the single response
        result = await get_chat_thread_response_non_streaming(thread_id, get_response_input, chat_thread_db, knowledge_bases)
        yield result

async def anext(ait):
    return await ait.__anext__()
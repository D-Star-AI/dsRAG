from pydantic import BaseModel, Field
from typing import Optional, List
import instructor
from dsrag.chat.chat_types import ExaSearchResults
import openai
import os
import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial

class Citation(BaseModel):
    doc_id: Optional[str] = Field(..., description="The document ID where the information used to generate the response was found.")
    url: Optional[str] = Field(None, description="The URL where the information used to generate the response was found.")
    title: Optional[str] = Field(None, description="The title of the website where the information used to generate the response was found.")
    page_number: Optional[int] = Field(None, description="The page where the information was found. May only be None if page numbers are not provided.")
    cited_text: str = Field(..., description="The exact text containing the information used to generate the response.")

class ResponseWithCitations(BaseModel):
    response: str = Field(..., description="The response to the user's question")
    citations: List[Citation] = Field(..., description="The citations used to generate the response")

# Create a partial version of ResponseWithCitations for streaming
PartialResponseWithCitations = instructor.Partial[ResponseWithCitations]

def format_page_content(page_number: int, content: str) -> str:
    """Format a single page's content with page number annotations"""
    return f"<page_{page_number}>\n{content}\n</page_{page_number}>"

def get_source_text(kb_id: str, doc_id: str, page_start: Optional[int], page_end: Optional[int], file_system) -> Optional[str]:
    """
    Get the source text for a given document and page range with page number annotations.
    Returns None if no page content is available.
    """
    if page_start is None or page_end is None:
        return None
    
    page_contents = file_system.load_page_content_range(kb_id, doc_id, page_start, page_end)
    if not page_contents:
        print(f"No page contents found for doc_id: {doc_id}, page_start: {page_start}, page_end: {page_end}")
        return None
    
    source_text = f"<doc_id: {doc_id}>\n"
    for i, content in enumerate(page_contents):
        page_number = page_start + i
        source_text += format_page_content(page_number, content) + "\n"
    source_text += f"</doc_id: {doc_id}>"

    return source_text

def format_sources_for_context(search_results: list[dict], kb_id: str, file_system) -> tuple[str, list[str]]:
    """
    Format all search results into a context string for the LLM.
    Handles both cases with and without page numbers.
    """
    context_parts = []
    all_doc_ids = []
    
    for result in search_results:
        doc_id = result["doc_id"]
        page_start = result.get("segment_page_start")
        page_end = result.get("segment_page_end")
        all_doc_ids.append(doc_id)
        
        full_page_source_text = None
        if page_start is not None and page_end is not None:
            # if we have page numbers, then we assume we have the page content - but this is not always the case
            full_page_source_text = get_source_text(kb_id, doc_id, page_start, page_end, file_system)
        
        if full_page_source_text:
            context_parts.append(full_page_source_text)
        else:
            result_content = result.get("content", "")
            context_parts.append(f"<doc_id: {doc_id}>\n{result_content}\n</doc_id: {doc_id}>")
    
    return "\n\n".join(context_parts), all_doc_ids

class WebsiteContent(BaseModel):
    content: str = Field(..., description="The content of the website where the information used to generate the response was found.")

class FirecrawlSearchResult(BaseModel):
    relevant_content: list[WebsiteContent] = Field(..., description="A list of the most relevant content from the website.")

async def format_firecrawl_search_results(search_results: list[dict], user_inout: str) -> list[dict]:
    """
    Make parallel llm calls to GPT-4o to filter the web results to the most relevant ones.
    """
    
    SYSTEM_PROMPT = """
    You are a helpful assistant that filters web search results to the most relevant ones.
    
    Here is the result from the website:
    
    URL: {url}
    TITLE: {title}
    DESCRIPTION: {description}
    CONTENT: {content}
    
    Here is the user's input:
    {user_input}
    
    Please extract the most relevant information from the result and return it as a list.
    You should just return the relevant content.
    DO NOT SUMMARIZE THE CONTENT. YOU MUST SIMPLY RETURN THE EXACT TEXT FROM THE WEBSITE FOR THE RELEVANT CONTENT.
    Try to return as much of the content as possible. If there is relevant information in a paragraph, return the entire paragraph plus the surrounding paragraphs.
    """
    
    async def process_single_result(result):
        prompt = SYSTEM_PROMPT.format(url=result["url"], title=result["title"], description=result["description"], content=result["content"], user_input=user_inout)
        message = {
            "role": "user",
            "content": prompt
        }
        
        client = instructor.from_openai(openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"]))
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[message],
            response_model=FirecrawlSearchResult,
            temperature=0.2,
            max_tokens=6000
        )
        
        # Add in the URL, title, and description from the original result
        response.url = result["url"]
        response.title = result["title"]
        response.description = result["description"]
        
        return response

    # Create tasks for all results
    tasks = [process_single_result(result) for result in search_results]
    
    # Run all tasks in parallel
    responses = await asyncio.gather(*tasks)
    
    return responses

async def filter_exa_search_results(search_results: list[ExaSearchResults], user_inout: str) -> tuple[str, list[str]]:
    """Format EXA search results into a context string for the LLM."""
    
    SYSTEM_PROMPT = """
    You are a helpful assistant that filters web search results to the most relevant ones.
    
    Here is the result from the website:
    
    URL: {url}
    TITLE: {title}
    CONTENT: {content}
    
    Here is the user's input:
    {user_input}
    
    Please extract the most relevant information from the result and return it as a list.
    You should just return the relevant content.
    DO NOT SUMMARIZE THE CONTENT. YOU MUST SIMPLY RETURN THE EXACT TEXT FROM THE WEBSITE FOR THE RELEVANT CONTENT.
    Try to return as much of the content as possible. If there is relevant information in a paragraph, return the entire paragraph plus the surrounding paragraphs.
    """
    
    def process_single_result(result):
                
        prompt = SYSTEM_PROMPT.format(
            url=result.url, 
            title=result.title, 
            content=result.text, 
            user_input=user_inout
        )
        message = {
            "role": "user",
            "content": prompt
        }
        
        client = instructor.from_openai(openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"]))
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[message],
            response_model=FirecrawlSearchResult,
            temperature=0.2,
            max_tokens=6000
        )
        
        # Turn the response into a dictionary, and concatenate the relevant content
        response_dict = response.model_dump()
        response_dict["relevant_content"] = "\n\n".join([content.content for content in response.relevant_content])
        
         # Add in the URL, title, and description from the original result
        response_dict["url"] = result.url
        response_dict["title"] = result.title
        
        return response_dict

    loop = asyncio.get_event_loop()
    
    print ("\n")
    print ("search_results length", len(search_results))
    print ("\n")
    
    seen_urls = set()
    unique_results = []
    for results in search_results:
        for result in results.results:
            print ("result type", type(result))
            # print the attributes of the result
            if result.url not in seen_urls:
                seen_urls.add(result.url)
                unique_results.append(result)
    
    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=10) as executor:
        # Create futures for all results
        futures = [
            loop.run_in_executor(executor, process_single_result, result)
            for result in unique_results
        ]
        
        # Wait for all futures to complete
        responses = await asyncio.gather(*futures)
    
    return responses

def format_exa_search_results(search_results: list[ExaSearchResults]) -> tuple[str, list[str]]:
    """
    Format EXA search results into a context string for the LLM.
    """
    
    """seen_urls = set()
    unique_results = []
    for results in search_results:
        for result in results.results:
            print ("result type", type(result))
            # print the attributes of the result
            if result.url not in seen_urls:
                seen_urls.add(result["url"])
                unique_results.append(result)"""
    
    # Extract and combine content from search results
    website_content = "RELEVANT WEBSITE CONTENT\n\n The following is content from websites that may be relevant to the user's question. You can use this content to help you write your response, but be careful: not all search results will be relevant, and sometimes you won't need to use any of them, so use your best judgement when deciding what to include in your response. Ignore any information here that is not relevant to the user's input.\n\n"
    website_content += "\n\n".join([
        f"<url: {result['url']}>\nTitle: {result['title']}\nContent: {result['relevant_content']} \n</url: {result['url']}>\n" # truncate to 10000 characters
        for i, result in enumerate(search_results)
    ])
        
    return website_content

def convert_elements_to_page_content(elements: list[dict], kb_id: str, doc_id: str, file_system) -> None:
    """
    Convert elements to page content and save it using the page content methods.
    This should be called when a document is first added to the knowledge base.
    Only processes documents where elements have page numbers.
    """
    # Check if this document has page numbers
    if not elements or "page_number" not in elements[0]:
        print(f"No page numbers found for document {doc_id}")
        return

    # Group elements by page
    pages = {}
    for element in elements:
        page_num = element["page_number"]
        if page_num not in pages:
            pages[page_num] = []
        pages[page_num].append(element["content"])

    # Save each page's content
    for page_num, contents in pages.items():
        page_content = "\n".join(contents)
        file_system.save_page_content(kb_id, doc_id, page_num, page_content)
from pydantic import BaseModel, Field
from typing import List
from firecrawl import FirecrawlApp
import instructor
import openai
import os
import ast
import dotenv

dotenv.load_dotenv()

firecrawl_api_key = os.getenv("FIRECRAWL_API_KEY")

class WebSearchResult(BaseModel):
    title: str = Field(..., description="The title of the website where the information used to generate the response was found.")
    url: str = Field(..., description="The URL where the information used to generate the response was found.")
    description: str = Field(..., description="The description of the website where the information used to generate the response was found.")

class ResponseWithWebSearchResults(BaseModel):
    response: str = Field(..., description="The response to the user's question")
    web_search_results: List[WebSearchResult] = Field(..., description="The web search results used to generate the response")

def filter_brave_search_results(brave_search_results: list[dict], user_query: str):
    # Use an LLM to filter the brave search results to the 5 most relevant results
    #llm = OpenAI(model="gpt-4o-mini", temperature=0)
    client = instructor.from_openai(openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"]))
    
    search_results = "\n".join([f"Title: {result['title']}\nURL: {result['url']}\nDescription: {result['description']}" for result in brave_search_results])
    prompt = f"""
    You are a helpful assistant that filters the web search results to the 5 most relevant results.
    Here are the web search results:
    {search_results}
    
    Here is the user query:
    {user_query}
    
    Return the 5 most relevant results from the web search results.
    """
    
    # Format the prompt to be in openai format
    prompt = [{"role": "user", "content": prompt}]
    
    response_model = ResponseWithWebSearchResults
    
    print ("brave_search_results", brave_search_results)
    print ("\n\n")
    print ("user_query", user_query)
    print ("\n\n")
    
    response =client.chat.completions.create(
        model="gpt-4o-mini",
        messages=prompt,
        response_model=response_model,
        temperature=0,
        max_tokens=1000
    )
    return response.web_search_results


def extract_failed_url_index(error_message: str) -> int:
    try:
        # Find the error list by looking for "Bad Request - "
        error_start = error_message.find("Bad Request - ") + len("Bad Request - ")
        if error_start == -1:
            return None
            
        # Get the full error list
        error_list_str = error_message[error_start:]
        try:
            error_list = ast.literal_eval(error_list_str)
        except Exception as e:
            return None
            
        # Find the path that contains the URL index
        for error in error_list:
            if 'path' in error and error['path'][0] == 'urls':
                return error['path'][1]  # Return the index
        return None
    except Exception as e:
        print("Unexpected error in extract_failed_url_index:", e)
        return None
    
    
def run_firecrawl_scrape_with_retry(urls: list) -> dict:
    while urls:
        try:
            return run_firecrawl_scrape(urls)
        except Exception as e:
            error_msg = str(e)
            if "This website is no longer supported" in error_msg:
                print ("\n\n")
                print ("Website is no longer supported, trying again with the next URL")
                print ("\n\n")
                failed_index = extract_failed_url_index(error_msg)
                print ("failed_index: ", failed_index)
                if failed_index is not None:
                    print(f"Removing unsupported URL: {urls[failed_index]}")
                    urls.pop(failed_index)
                    continue
            # If we can't handle the error, re-raise it
            raise e
    return {}

def run_firecrawl_scrape(urls: list[str]):
    app = FirecrawlApp(api_key=firecrawl_api_key)
    scrape_result = app.async_batch_scrape_urls(urls, {"formats": ["markdown"]})
    return scrape_result


def get_firecrawl_status(scrape_id: str):
    app = FirecrawlApp(api_key=firecrawl_api_key)
    batch_scrape_status = app.check_batch_scrape_status(scrape_id)
    return batch_scrape_status
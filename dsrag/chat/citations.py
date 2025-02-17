from pydantic import BaseModel, Field
from typing import Optional, List

class Citation(BaseModel):
    doc_id: str = Field(..., description="The document ID where the information used to generate the response was found.")
    page_numbers: Optional[List[int]] = Field(None, description="The page numbers where the information was found. May be None if page numbers are not available.")
    cited_text: Optional[str] = Field(None, description="The exact text containing the information used to generate the response. May be None if page content is not available.")

class ResponseWithCitations(BaseModel):
    response: str = Field(..., description="The response to the user's question")
    citations: List[Citation] = Field(..., description="The citations used to generate the response")

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
        return None
    
    source_text = f"<doc_id: {doc_id}>\n"
    for i, content in enumerate(page_contents):
        page_number = page_start + i
        source_text += format_page_content(page_number, content) + "\n"
    source_text += f"</doc_id: {doc_id}>"

    return source_text

def format_sources_for_context(search_results: list[dict], kb_id: str, file_system) -> str:
    """
    Format all search results into a context string for the LLM.
    Handles both cases with and without page numbers.
    """
    context_parts = []
    
    for result in search_results:
        doc_id = result["doc_id"]
        page_start = result.get("segment_page_start")
        page_end = result.get("segment_page_end")
        
        source_text = None
        if page_start is not None and page_end is not None:
            source_text = get_source_text(kb_id, doc_id, page_start, page_end, file_system)
        
        if source_text:
            context_parts.append(source_text)
        else:
            result_content = result.get("content", "")
            context_parts.append(f"<doc_id: {doc_id}>\n{result_content}\n</doc_id: {doc_id}>")
    
    return "\n\n".join(context_parts)

def convert_elements_to_page_content(elements: list[dict], kb_id: str, doc_id: str, file_system) -> None:
    """
    Convert elements to page content and save it using the page content methods.
    This should be called when a document is first added to the knowledge base.
    Only processes documents where elements have page numbers.
    """
    # Check if this document has page numbers
    if not elements or "page_number" not in elements[0]:
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
from dsrag.llm import LLM
import tiktoken

DOCUMENT_SUMMARIZATION_PROMPT = """
INSTRUCTIONS
What is the following document, and what is it about? 

Your response should be a single sentence, and it shouldn't be an excessively long sentence. DO NOT respond with anything else.

You MUST include the name of the document in your response (if available), as that is a critical piece of information. Be as specific and detailed as possible in your document name. You can even include things like the author's name or the date of publication if that information is available. DO NOT just use the filename as the document name. It needs to be a descriptive and human-readable name.

Your response should take the form of "This document is: X, and is about: Y". For example, if the document is a book about the history of the United States called A People's History of the United States, your response might be "This document is: A People's History of the United States, and is about the history of the United States, covering the period from 1776 to the present day." If the document is the 2023 Form 10-K for Apple Inc., your response might be "This document is: Apple Inc. FY2023 Form 10-K, and is about: the financial performance and operations of Apple Inc. during the fiscal year 2023."

{document_summarization_guidance}

{truncation_message}

DOCUMENT
File name: {document_title}

{document_text}
""".strip()

TRUNCATION_MESSAGE = """
Also note that the document text provided below is just the first ~4500 words of the document. Your response should still pertain to the entire document, not just the text provided below.
""".strip()

SECTION_SUMMARIZATION_PROMPT = """
INSTRUCTIONS
What is the following section about? 

Your response should be a single sentence, and it shouldn't be an excessively long sentence. DO NOT respond with anything else.

Your response should take the form of "This section is about: X". For example, if the section is a balance sheet from a financial report about Apple, your response might be "This section is about: the financial position of Apple as of the end of the fiscal year." If the section is a chapter from a book on the history of the United States, and this chapter covers the Civil War, your response might be "This section is about: the causes and consequences of the American Civil War."

{section_summarization_guidance}

SECTION
Document name: {document_title}
Section name: {section_title}

{section_text}
""".strip()

def truncate_content(content: str, max_tokens: int):
    TOKEN_ENCODER = tiktoken.encoding_for_model('gpt-3.5-turbo')
    tokens = TOKEN_ENCODER.encode(content, disallowed_special=())
    truncated_tokens = tokens[:max_tokens]
    return TOKEN_ENCODER.decode(truncated_tokens), min(len(tokens), max_tokens)

def get_document_summary(auto_context_model: LLM, document_text: str, document_title: str, document_summarization_guidance: str = ""):
    # truncate the content if it's too long
    max_content_tokens = 6000 # if this number changes, also update the truncation message above
    document_text, num_tokens = truncate_content(document_text, max_content_tokens)
    if num_tokens < max_content_tokens:
        truncation_message = ""
    else:
        truncation_message = TRUNCATION_MESSAGE
    
    # get document summary
    prompt = DOCUMENT_SUMMARIZATION_PROMPT.format(document_summarization_guidance=document_summarization_guidance, document_text=document_text, document_title=document_title, truncation_message=truncation_message)
    chat_messages = [{"role": "user", "content": prompt}]
    document_summary = auto_context_model.make_llm_call(chat_messages)
    return document_summary

def get_section_summary(auto_context_model: LLM, text: str, document_title: str, section_title: str, section_summarization_guidance: str = ""):
    prompt = SECTION_SUMMARIZATION_PROMPT.format(section_summarization_guidance=section_summarization_guidance, section_text=text, document_title=document_title, section_title=section_title)
    chat_messages = [{"role": "user", "content": prompt}]
    section_summary = auto_context_model.make_llm_call(chat_messages)
    return section_summary

def get_chunk_header(document_title: str = "", document_summary: str = "", section_title: str = "", section_summary: str = ""):
    """
    The chunk header is what gets prepended to each chunk before embedding or reranking. At the very least, it should contain the document title.
    """
    chunk_header = ""
    if document_title:
        chunk_header += f"Document context: the following excerpt is from a document titled '{document_title}'. {document_summary}"
    if section_title:
        chunk_header += f"\nSection context: this excerpt is from the section titled '{section_title}'. {section_summary}"
    return chunk_header

def get_segment_header(document_title: str = "", document_summary: str = ""):
    """
    The segment header is what gets prepended to each segment (i.e. search result). This provides context to the LLM about the segment.
    """
    segment_header = ""
    if document_title:
        segment_header += f"Document context: the following excerpt is from a document titled '{document_title}'. {document_summary}"

from dsrag.llm import LLM
import tiktoken

PROMPT = """
INSTRUCTIONS
What is the following document, and what is it about? 

Your response should be a single sentence, and it shouldn't be an excessively long sentence. DO NOT respond with anything else.

You MUST include the name of the document in your response (if available), as that is a critical piece of information. Be as specific and detailed as possible in your document name. You can even include things like the author's name or the date of publication if that information is available. DO NOT just use the filename as the document name. It needs to be a descriptive and human-readable name.

Your response should take the form of "This document is: X, and is about: Y". For example, if the document is a book about the history of the United States called A People's History of the United States, your response might be "This document is: A People's History of the United States, and is about the history of the United States, covering the period from 1776 to the present day." If the document is the 2023 Form 10-K for Apple Inc., your response might be "This document is: Apple Inc. FY2023 Form 10-K, and is about: the financial performance and operations of Apple Inc. during the fiscal year 2023."

{auto_context_guidance}

{truncation_message}

DOCUMENT
filename: {document_title}

{document}
""".strip()

TRUNCATION_MESSAGE = """
Also note that the document text provided below is just the first ~4500 words of the document. Your response should still pertain to the entire document, not just the text provided below.
""".strip()

def truncate_content(content: str, max_tokens: int):
    TOKEN_ENCODER = tiktoken.encoding_for_model('gpt-3.5-turbo')
    tokens = TOKEN_ENCODER.encode(content, disallowed_special=())
    truncated_tokens = tokens[:max_tokens]
    return TOKEN_ENCODER.decode(truncated_tokens), min(len(tokens), max_tokens)

def get_document_context(auto_context_model: LLM, text: str, document_title: str, auto_context_guidance: str = ""):
    # truncate the content if it's too long
    max_content_tokens = 6000 # if this number changes, also update the truncation message above
    text, num_tokens = truncate_content(text, max_content_tokens)
    if num_tokens < max_content_tokens:
        truncation_message = ""
    else:
        truncation_message = TRUNCATION_MESSAGE
    
    # get document context
    prompt = PROMPT.format(auto_context_guidance=auto_context_guidance, document=text, document_title=document_title, truncation_message=truncation_message)
    chat_messages = [{"role": "user", "content": prompt}]
    document_context = auto_context_model.make_llm_call(chat_messages)
    return document_context

def get_chunk_header(file_name, document_context):
    chunk_header = f"Document context: the following excerpt is from {file_name}. {document_context}"
    return chunk_header
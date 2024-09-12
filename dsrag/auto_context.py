from dsrag.llm import LLM
import tiktoken

DOCUMENT_TITLE_PROMPT = """
INSTRUCTIONS
What is the title of the following document?

Your response MUST be the title of the document, and nothing else. DO NOT respond with anything else.

{document_title_guidance}

{non_english_addendum}

{truncation_message}

DOCUMENT
{document_text}
""".strip()

DOCUMENT_SUMMARIZATION_PROMPT = """
INSTRUCTIONS
What is the following document, and what is it about? 

Your response should be a single sentence, and it shouldn't be an excessively long sentence. DO NOT respond with anything else.

Your response should take the form of "This document is about: X" (This part should be translated into the language of the document if it's not in English). For example, if the document is a book about the history of the United States called A People's History of the United States, your response might be "This document is about: the history of the United States, covering the period from 1776 to the present day." If the document is the 2023 Form 10-K for Apple Inc., your response might be "This document is about: the financial performance and operations of Apple Inc. during the fiscal year 2023." If the document is the novel Les Miserables by Victor Hugo, your response might be "Ce document concerne : le roman "Les Misérables" de Victor Hugo, qui explore les thèmes de l'injustice sociale, de la rédemption et des luttes de divers personnages dans la France du XIXe siècle."

{document_summarization_guidance}

{non_english_addendum}

{truncation_message}

DOCUMENT
Document name: {document_title}

{document_text}
""".strip()

TRUNCATION_MESSAGE = """
Also note that the document text provided below is just the first ~{num_words} words of the document. That should be plenty for this task. Your response should still pertain to the entire document, not just the text provided below.
""".strip()

SECTION_SUMMARIZATION_PROMPT = """
INSTRUCTIONS
What is the following section about? 

Your response should be a single sentence, and it shouldn't be an excessively long sentence. DO NOT respond with anything else.

Your response should take the form of "This section is about: X". (This part should be translated into the language of the document if it's not in English). For example, if the section is a balance sheet from a financial report about Apple, your response might be "This section is about: the financial position of Apple as of the end of the fiscal year." If the section is a chapter from a book on the history of the United States, and this chapter covers the Civil War, your response might be "This section is about: the causes and consequences of the American Civil War."

{section_summarization_guidance}

{non_english_addendum}

SECTION
Document name: {document_title}
Section name: {section_title}

{section_text}
""".strip()

LANGUAGE_ADDENDUM = "YOU MUST use the same language as the document for your entire response. If the document is in English, your response MUST BE entirely in English. If the document is in another language, your response MUST BE entirely in that language."

def truncate_content(content: str, max_tokens: int):
    TOKEN_ENCODER = tiktoken.encoding_for_model('gpt-3.5-turbo')
    tokens = TOKEN_ENCODER.encode(content, disallowed_special=())
    truncated_tokens = tokens[:max_tokens]
    return TOKEN_ENCODER.decode(truncated_tokens), min(len(tokens), max_tokens)

def get_document_title(auto_context_model: LLM, document_text: str, document_title_guidance: str = "", language: str = "en"):
    # truncate the content if it's too long
    max_content_tokens = 4000 # if this number changes, also update num_words in the truncation message below
    document_text, num_tokens = truncate_content(document_text, max_content_tokens)
    if num_tokens < max_content_tokens:
        truncation_message = ""
    else:
        truncation_message = TRUNCATION_MESSAGE.format(num_words=3000)

    # see if we need to add an addendum about non-English responses
    if language != "en":
        non_english_addendum = LANGUAGE_ADDENDUM
    else:
        non_english_addendum = ""

    # get document title
    prompt = DOCUMENT_TITLE_PROMPT.format(document_title_guidance=document_title_guidance, non_english_addendum=non_english_addendum, document_text=document_text, truncation_message=truncation_message)
    chat_messages = [{"role": "user", "content": prompt}]
    document_title = auto_context_model.make_llm_call(chat_messages)
    return document_title

def get_document_summary(auto_context_model: LLM, document_text: str, document_title: str, document_summarization_guidance: str = "", language: str = "en"):
    # truncate the content if it's too long
    max_content_tokens = 8000 # if this number changes, also update num_words in the truncation message below
    document_text, num_tokens = truncate_content(document_text, max_content_tokens)
    if num_tokens < max_content_tokens:
        truncation_message = ""
    else:
        truncation_message = TRUNCATION_MESSAGE.format(num_words=6000)
    
    # see if we need to add an addendum about non-English responses
    if language != "en":
        non_english_addendum = LANGUAGE_ADDENDUM
    else:
        non_english_addendum = ""

    # get document summary
    prompt = DOCUMENT_SUMMARIZATION_PROMPT.format(document_summarization_guidance=document_summarization_guidance, non_english_addendum=non_english_addendum, document_text=document_text, document_title=document_title, truncation_message=truncation_message)
    chat_messages = [{"role": "user", "content": prompt}]
    document_summary = auto_context_model.make_llm_call(chat_messages)
    return document_summary

def get_section_summary(auto_context_model: LLM, section_text: str, document_title: str, section_title: str, section_summarization_guidance: str = "", language: str = "en"):
    # see if we need to add an addendum about non-English responses
    if language != "en":
        non_english_addendum = LANGUAGE_ADDENDUM
    else:
        non_english_addendum = ""
    
    prompt = SECTION_SUMMARIZATION_PROMPT.format(section_summarization_guidance=section_summarization_guidance, non_english_addendum=non_english_addendum, section_text=section_text, document_title=document_title, section_title=section_title)
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
        chunk_header += f"\n\nSection context: this excerpt is from the section titled '{section_title}'. {section_summary}"
    return chunk_header

def get_segment_header(document_title: str = "", document_summary: str = ""):
    """
    The segment header is what gets prepended to each segment (i.e. search result). This provides context to the LLM about the segment. The segment header only contains document-level information. Section-level information is not needed here, because either the segment will be large enough to not need section-level context, or the it will be a chunk where the section-level context isn't relevant to the query (because otherwise more of the section would have been included in the segment).
    """
    segment_header = ""
    if document_title:
        segment_header += f"Document context: the following excerpt is from a document titled '{document_title}'. {document_summary}"
    return segment_header
import os
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from anthropic import Anthropic
from openai import OpenAI
import instructor


class Section(BaseModel):
    title: str = Field(description="main topic of this section of the document (very descriptive)")
    start_index: int = Field(description="line number where the section begins (inclusive)")

class StructuredDocument(BaseModel):
    """obtains meaningful sections, each centered around a single concept/topic"""
    sections: List[Section] = Field(description="a list of sections of the document")


SYSTEM_PROMPT = """
Read the document below and extract a StructuredDocument object from it where each section of the document is centered around a single concept/topic. Whenever possible, your sections (and section titles) should match up with the natural sections of the document (i.e. Introduction, Conclusion, References, etc.). Sections can vary in length, but should generally be anywhere from a few paragraphs to a few pages long.
Each line of the document is marked with its line number in square brackets (e.g. [1], [2], [3], etc). Use the line numbers to indicate section start.
The start line numbers will be treated as inclusive. For example, if the first line of a section is line 5, the start_index should be 5. Your goal is to find the starting line number of a given section, where a section is a group of lines that are thematically related.
The first section must start at the first line number of the document ({start_line} in this case). The sections MUST cover the entire document. 
Section titles should be descriptive enough such that a person who is just skimming over the section titles and not actually reading the document can get a clear idea of what each section is about.
Note: the document provided to you may just be an excerpt from a larger document, rather than a complete document. Therefore, you can't always assume, for example, that the first line of the document is the beginning of the Introduction section and the last line is the end of the Conclusion section (if those section are even present).
"""

LANGUAGE_ADDENDUM = "For your section titles, YOU MUST use the same language as the document. If the document is in English, your section titles should be in English. If the document is in another language, your section titles should be in that language."


def get_document_lines(document: str) -> List[str]:
    document_lines = document.split("\n")
    return document_lines

def get_document_with_lines(document_lines: List[str], start_line: int, max_characters: int) -> str:
    document_with_line_numbers = ""
    character_count = 0
    for i in range(start_line, len(document_lines)):
        line = document_lines[i]
        document_with_line_numbers += f"[{i}] {line}\n"
        character_count += len(line)
        if character_count > max_characters or i == len(document_lines) - 1:
            end_line = i
            break
    return document_with_line_numbers, end_line

def get_structured_document(document_with_line_numbers: str, start_line: int, end_line: int, llm_provider: str, model: str, language: str) -> StructuredDocument:
    """
    Note: This function relies on Instructor, which only supports certain model providers. That's why this function doesn't use the LLM abstract base class that is used elsewhere in the project.
    """

    formatted_system_prompt = SYSTEM_PROMPT.format(start_line=start_line)
    if language != "en":
        formatted_system_prompt += "\n" + LANGUAGE_ADDENDUM

    if llm_provider == "anthropic":
        base_url = os.environ.get("DSRAG_ANTHROPIC_BASE_URL", None)
        if base_url is not None:
            client = instructor.from_anthropic(Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"], base_url=base_url))
        else:
            client = instructor.from_anthropic(Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"]))
        return client.chat.completions.create(
            model=model,
            response_model=StructuredDocument,
            max_tokens=4000,
            temperature=0.0,
            system=formatted_system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": document_with_line_numbers,
                },
            ],
        )
    elif llm_provider == "openai":
        base_url = os.environ.get("DSRAG_OPENAI_BASE_URL", None)
        if base_url is not None:
            client = instructor.from_openai(OpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url=base_url))
        else:
            client = instructor.from_openai(OpenAI(api_key=os.environ["OPENAI_API_KEY"]))
        return client.chat.completions.create(
            model=model,
            response_model=StructuredDocument,
            max_tokens=4000,
            temperature=0.0,
            messages=[
                {
                    "role": "system",
                    "content": formatted_system_prompt,
                },
                {
                    "role": "user",
                    "content": document_with_line_numbers,
                },
            ],
        )
    else:
        raise ValueError("Invalid provider. Must be either 'anthropic' or 'openai'.")

def get_sections_text(sections: List[Section], document_lines: List[str]):
    """
    Takes in a list of Section objects and returns a list of dictionaries containing the attributes of each Section object plus the content of the section.
    """
    section_dicts = []
    for i,s in enumerate(sections):
        if i == len(sections) - 1:
            end_index = len(document_lines) - 1
        else:
            end_index = sections[i+1].start_index - 1
        contents = document_lines[s.start_index:end_index+1] # +1 because end_index is inclusive
        section_dicts.append({
            "title": s.title,
            "content": "\n".join(contents),
            "start": s.start_index,
            "end": end_index
        })
    return section_dicts


def get_sections(document: str, max_characters: int = 20000, llm_provider: str = "openai", model: str = "gpt-4o-mini", language: str = "en") -> List[Dict[str, Any]]:
    """
    Inputs
    - document: str - the text of the document
    - max_characters: int - the maximum number of characters to process in one call to the LLM
    - llm_provider: str - the LLM provider to use (either "anthropic" or "openai")
    - model: str - the name of the LLM model to use

    Returns
    - all_sections: a list of dictionaries, each containing the following keys:
        - title: str - the main topic of this section of the document (very descriptive)
        - start: int - line number where the section begins (inclusive)
        - end: int - line number where the section ends (inclusive)
        - content: str - the text of the section
    """
    max_iterations = 2*(len(document) // max_characters + 1)
    document_lines = get_document_lines(document)
    start_line = 0
    all_sections = []
    for _ in range(max_iterations):
        document_with_line_numbers, end_line = get_document_with_lines(document_lines, start_line, max_characters)
        structured_doc = get_structured_document(document_with_line_numbers, start_line, end_line, llm_provider=llm_provider, model=model, language=language)
        new_sections = structured_doc.sections
        all_sections.extend(new_sections)
        
        if end_line >= len(document_lines) - 1:
            # reached the end of the document
            break
        else:
            if len(new_sections) > 1:
                start_line = all_sections[-1].start_index # start from the next line after the last section
                all_sections.pop()
            else:
                start_line = end_line + 1

    # get the section text
    section_dicts = get_sections_text(all_sections, document_lines)

    return section_dicts, document_lines
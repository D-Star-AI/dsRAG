import os
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from anthropic import Anthropic
from openai import OpenAI
import instructor
from ..models.types import SemanticSectioningConfig, Line, Section, Element, ElementType

class DocumentSection(BaseModel):
    title: str = Field(description="main topic of this section of the document (very descriptive)")
    start_index: int = Field(description="line number where the section begins (inclusive)")

class StructuredDocument(BaseModel):
    """obtains meaningful sections, each centered around a single concept/topic"""
    sections: List[DocumentSection] = Field(description="a list of sections of the document")


SYSTEM_PROMPT = """
Read the document below and extract a StructuredDocument object from it where each section of the document is centered around a single concept/topic. Whenever possible, your sections (and section titles) should match up with the natural sections of the document (i.e. Introduction, Conclusion, References, etc.). Sections can vary in length, but should generally be anywhere from a few paragraphs to a few pages long.
Each line of the document is marked with its line number in square brackets (e.g. [1], [2], [3], etc). Use the line numbers to indicate section start.
The start line numbers will be treated as inclusive. For example, if the first line of a section is line 5, the start_index should be 5. Your goal is to find the starting line number of a given section, where a section is a group of lines that are thematically related.
The first section must start at the first line number of the document ({start_line} in this case). The sections MUST cover the entire document. 
Section titles should be descriptive enough such that a person who is just skimming over the section titles and not actually reading the document can get a clear idea of what each section is about.
Note: the document provided to you may just be an excerpt from a larger document, rather than a complete document. Therefore, you can't always assume, for example, that the first line of the document is the beginning of the Introduction section and the last line is the end of the Conclusion section (if those section are even present).
"""

LANGUAGE_ADDENDUM = "For your section titles, YOU MUST use the same language as the document. If the document is in English, your section titles should be in English. If the document is in another language, your section titles should be in that language."


def get_document_with_lines(document_lines: List[Line], start_line: int, max_characters: int) -> tuple[str, int]:
    document_with_line_numbers = ""
    character_count = 0
    for i in range(start_line, len(document_lines)):
        line = document_lines[i]["content"]
        document_with_line_numbers += f"[{i}] {line}\n"
        character_count += len(line)
        if character_count > max_characters or i == len(document_lines) - 1:
            end_line = i
            break
    return document_with_line_numbers, end_line

def get_structured_document(document_with_line_numbers: str, start_line: int, llm_provider: str, model: str, language: str) -> StructuredDocument:
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

def get_sections_text(sections: List[DocumentSection], document_lines: List[Line]) -> List[Section]:
    """
    Takes in a list of DocumentSection objects and returns a list of dictionaries containing the attributes of each Section object plus the content of the section.
    """
    section_dicts = []
    for i,s in enumerate(sections):
        if i == len(sections) - 1:
            end_index = len(document_lines) - 1
        else:
            end_index = sections[i+1].start_index - 1
        contents = [document_lines[j]["content"] for j in range(s.start_index, end_index+1)]

        section_dicts.append(Section(
            title=s.title,
            content="\n".join(contents),
            start=s.start_index,
            end=end_index
        ))
    return section_dicts


def get_sections(document_lines: List[Line], max_iterations: int, max_characters: int = 20000, llm_provider: str = "openai", model: str = "gpt-4o-mini", language: str = "en") -> List[Section]:
    """
    Inputs
    - document_lines: list[dict] - the text of the document
    - max_iterations: int - the maximum number of iterations to run (used as a safety measure to prevent the possibility of an infinite loop)
    - max_characters: int - the maximum number of characters to process in one call to the LLM
    - llm_provider: str - the LLM provider to use (either "anthropic" or "openai")
    - model: str - the name of the LLM model to use

    Returns
    - sections: a list of dictionaries, each containing the following keys:
        - title: str - the main topic of this section of the document (very descriptive)
        - start: int - line number where the section begins (inclusive)
        - end: int - line number where the section ends (inclusive)
        - content: str - the text of the section
    """
    
    start_line = 0
    all_sections = []
    for _ in range(max_iterations):
        document_with_line_numbers, end_line = get_document_with_lines(document_lines, start_line, max_characters)
        structured_doc = get_structured_document(document_with_line_numbers, start_line, llm_provider=llm_provider, model=model, language=language)
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
    sections = get_sections_text(all_sections, document_lines)

    return sections

def elements_to_lines(elements: List[Element], exclude_elements: List[str], visual_elements: List[str]) -> List[Line]:
    """
    Inputs
    - elements: list[dict] - the elements of the document
    - exclude_elements: list[str] - the types of elements to exclude
    - visual_elements: list[str] - the types of elements that are visual and therefore should not be split
    """
    document_lines = []
    for element in elements:
        if element["type"] in exclude_elements:
            continue
        elif element["type"] in visual_elements:
            # don't split visual elements
            document_lines.append({
                "content": element["content"],
                "element_type": element["type"],
                "page_number": element.get("page_number", None),
                "is_visual": True,
            })
        else:
            lines = element["content"].split("\n")
            for line in lines:
                document_lines.append({
                    "content": line,
                    "element_type": element["type"],
                    "page_number": element.get("page_number", None),
                    "is_visual": False,
                })

    return document_lines

def str_to_lines(document: str) -> List[Line]:
    document_lines = []
    lines = document.split("\n")
    for line in lines:
        document_lines.append({
            "content": line,
            "element_type": "NarrativeText",
            "page_number": None,
            "is_visual": False,
        })

    return document_lines

def pages_to_lines(pages: List[str]) -> List[Line]:
    document_lines = []
    for i, page in enumerate(pages):
        lines = page.split("\n")
        for line in lines:
            document_lines.append({
                "content": line,
                "element_type": "NarrativeText",
                "page_number": i+1, # page numbers are 1-indexed
                "is_visual": False,
            })

    return document_lines

def no_semantic_sectioning(document: str, num_lines: int) -> List[Section]:
    # return the entire document as a single section
    sections = [{
        "title": "",
        "content": document,
        "start": 0,
        "end": num_lines - 1 # 0-indexed and inclusive
    }]
    return sections

def get_sections_from_elements(elements: List[Element], element_types: List[ElementType], exclude_elements: List[str] = [], max_characters: int = 20000, semantic_sectioning_config: SemanticSectioningConfig = {}) -> tuple[List[Section], List[Line]]:
    # get the semantic sectioning config params, using defaults if not provided
    use_semantic_sectioning = semantic_sectioning_config.get("use_semantic_sectioning", True)
    llm_provider = semantic_sectioning_config.get("llm_provider", "openai")
    model = semantic_sectioning_config.get("model", "gpt-4o-mini")
    language = semantic_sectioning_config.get("language", "en")

    visual_elements = [e["name"] for e in element_types if e["is_visual"]]

    document_lines = elements_to_lines(elements=elements, exclude_elements=exclude_elements, visual_elements=visual_elements)
    document_lines_str = [line["content"] for line in document_lines]
    document_str = "\n".join(document_lines_str)
    
    if use_semantic_sectioning:
        max_iterations = 2*(len(document_str) // max_characters + 1)
        sections = get_sections(
            document_lines=document_lines, 
            max_iterations=max_iterations, 
            max_characters=max_characters, 
            llm_provider=llm_provider, 
            model=model, 
            language=language
        )
    else:
        sections = no_semantic_sectioning(document=document_str, num_lines=len(document_lines))
    
    return sections, document_lines

def get_sections_from_str(document: str, max_characters: int = 20000, semantic_sectioning_config: SemanticSectioningConfig = {}) -> tuple[List[Section], List[Line]]:
    # get the semantic sectioning config params, using defaults if not provided
    use_semantic_sectioning = semantic_sectioning_config.get("use_semantic_sectioning", True)
    llm_provider = semantic_sectioning_config.get("llm_provider", "openai")
    model = semantic_sectioning_config.get("model", "gpt-4o-mini")
    language = semantic_sectioning_config.get("language", "en")

    document_lines = str_to_lines(document)
    
    if use_semantic_sectioning:
        max_iterations = 2*(len(document) // max_characters + 1)
        sections = get_sections(
            document_lines=document_lines, 
            max_iterations=max_iterations, 
            max_characters=max_characters, 
            llm_provider=llm_provider, 
            model=model, 
            language=language
        )
    else:
        sections = no_semantic_sectioning(document=document, num_lines=len(document_lines))
    return sections, document_lines

def get_sections_from_pages(pages: List[str], max_characters: int = 20000, semantic_sectioning_config: SemanticSectioningConfig = {}) -> tuple[List[Section], List[Line]]:
    # get the semantic sectioning config params, using defaults if not provided
    use_semantic_sectioning = semantic_sectioning_config.get("use_semantic_sectioning", True)
    llm_provider = semantic_sectioning_config.get("llm_provider", "openai")
    model = semantic_sectioning_config.get("model", "gpt-4o-mini")
    language = semantic_sectioning_config.get("language", "en")

    document_lines = pages_to_lines(pages)
    document_lines_str = [line["content"] for line in document_lines]
    document_str = "\n".join(document_lines_str)
    
    if use_semantic_sectioning:
        max_iterations = 2*(len(document_str) // max_characters + 1)
        sections = get_sections(
            document_lines=document_lines, 
            max_iterations=max_iterations, 
            max_characters=max_characters, 
            llm_provider=llm_provider, 
            model=model, 
            language=language
        )
    else:
        sections = no_semantic_sectioning(document=document_str, num_lines=len(document_lines))
    
    return sections, document_lines
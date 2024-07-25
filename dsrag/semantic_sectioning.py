from pydantic import BaseModel, Field
from typing import List, Dict, Any
from anthropic import Anthropic
from openai import OpenAI
import instructor


class Section(BaseModel):
    title: str = Field(description="main topic of this section of the document (very descriptive)")
    start_index: int = Field(description="line number where the section begins (inclusive)")
    end_index: int = Field(description="line number where the section ends (inclusive)")


class StructuredDocument(BaseModel):
    """obtains meaningful sections, each centered around a single concept/topic"""
    sections: List[Section] = Field(description="a list of sections of the document")


system_prompt = """
Read the document below and extract a StructuredDocument object from it where each section of the document is centered around a single concept/topic. Whenever possible, your sections (and section titles) should match up with the natural sections of the document (i.e. Introduction, Conclusion, References, etc.). Sections can vary in length, but should generally be anywhere from a few paragraphs to a few pages long.
Each line of the document is marked with its line number in square brackets (e.g. [1], [2], [3], etc). Use the line numbers to indicate section start and end.
The start and end line numbers will be treated as inclusive. For example, if the first line of a section is line 5 and the last line is line 10, the start_index should be 5 and the end_index should be 10.
The first section must start at the first line number of the document ({start_line} in this case), and the last section must end at the last line of the document ({end_line} in this case). The sections MUST be non-overlapping and cover the entire document. In other words, they must form a partition of the document.
Section titles should be descriptive enough such that a person who is just skimming over the section titles and not actually reading the document can get a clear idea of what each section is about.
Note: the document provided to you may just be an excerpt from a larger document, rather than a complete document. Therefore, you can't always assume, for example, that the first line of the document is the beginning of the Introduction section and the last line is the end of the Conclusion section (if those section are even present).
"""


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

def get_structured_document(document_with_line_numbers: str, start_line: int, end_line: int, llm_provider: str, model: str) -> StructuredDocument:
    """
    Note: This function relies on Instructor, which only supports certain model providers. That's why this function doesn't use the LLM abstract base class that is used elsewhere in the project.
    """
    if llm_provider == "anthropic":
        client = instructor.from_anthropic(Anthropic())
        return client.chat.completions.create(
            model=model,
            response_model=StructuredDocument,
            max_tokens=4000,
            temperature=0.0,
            system=system_prompt.format(start_line=start_line, end_line=end_line),
            messages=[
                {
                    "role": "user",
                    "content": document_with_line_numbers,
                },
            ],
        )
    elif llm_provider == "openai":
        client = instructor.from_openai(OpenAI())
        return client.chat.completions.create(
            model=model,
            response_model=StructuredDocument,
            max_tokens=4000,
            temperature=0.0,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt.format(start_line=start_line, end_line=end_line),
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
    for s in sections:
        contents = document_lines[s.start_index:s.end_index+1] # end_index is inclusive
        section_dicts.append({
            "title": s.title,
            "content": "\n".join(contents),
            "start": s.start_index,
            "end": s.end_index
        })
    return section_dicts

def partition_sections(sections, a, b):
    """
    - sections: a list of Section objects, each containing the following attributes:
        - title: str - the main topic of this section of the document
        - start_index: int - line number where the section begins (inclusive)
        - end_index: int - line number where the section ends (inclusive)
    """
    if len(sections) == 0:
        return [
            Section(title="", start_index=a, end_index=b)
        ]
    
    # Filter out sections that are completely outside the range [a, b]
    sections = [s for s in sections if s.start_index <= b and s.end_index >= a]

    # Filter out any sections where the end index is less than the start index
    sections = [s for s in sections if s.start_index <= s.end_index]

    if len(sections) == 0:
        return [
            Section(title="", start_index=a, end_index=b)
        ]

    # Adjust sections that partially overlap with the range [a, b]
    for s in sections:
        if s.start_index < a:
            s.start_index = a
            s.title = ""
        if s.end_index > b:
            s.end_index = b
            s.title = ""
    
    # Sort the intervals by their start value
    sections.sort(key=lambda x: x.start_index)

    # Remove any sections that are completely contained within another section
    i = 0
    while i < len(sections) - 1:
        if sections[i].end_index >= sections[i+1].end_index:
            sections.pop(i+1)
        else:
            i += 1

    if len(sections) == 0:
        return [
            Section(title="", start_index=a, end_index=b)
        ]

    # Ensure the first section starts at a
    if sections[0].start_index > a:
        sections.insert(0, Section(title="", start_index=a, end_index=sections[0].start_index-1))

    # Ensure the last section ends at b
    if sections[-1].end_index < b:
        sections.append(Section(title="", start_index=sections[-1].end_index+1, end_index=b))

    # Ensure there are no gaps or overlaps between sections
    completed_sections = []
    for i in range(0, len(sections)):
        if i == 0:
            # Automatically add the first sectoin
            completed_sections.append(sections[i])
        else:
            if sections[i].start_index > sections[i-1].end_index + 1:
                # There is a gap between sections[i-1] and sections[i]
                completed_sections.append(Section(title="", start_index=sections[i-1].end_index+1, end_index=sections[i].start_index-1))
            elif sections[i].start_index <= sections[i-1].end_index:
                # There is an overlap between sections[i-1] and sections[i]
                completed_sections[-1].end_index = sections[i].start_index - 1
                completed_sections[-1].title = ""
            # Always add the current iteration's section
            completed_sections.append(sections[i])


    return completed_sections


def is_valid_partition(sections, a, b):
    if sections[0].start_index != a:
        return False
    if sections[-1].end_index != b:
        return False

    for i in range(1, len(sections)):
        if sections[i].start_index != sections[i-1].end_index + 1:
            return False
    
    return True


def get_sections(document: str, max_characters: int = 20000, llm_provider: str = "openai", model: str = "gpt-4o-mini") -> List[Dict[str, Any]]:
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
        structured_doc = get_structured_document(document_with_line_numbers, start_line, end_line, llm_provider=llm_provider, model=model)
        new_sections = structured_doc.sections
        all_sections.extend(new_sections)
        
        if end_line >= len(document_lines) - 1:
            # reached the end of the document
            break
        else:
            if len(new_sections) > 1:
                # remove last section since it's assumed to be incomplete (but only if we added more than one section in this iteration)
                all_sections.pop()
            start_line = all_sections[-1].end_index + 1 # start from the next line after the last section

    # fix the sections so that they form a partition of the document
    a = 0
    b = len(document_lines) - 1

    # the fact that this is in a loop is a complete hack to deal with the fact that the partitioning function is not perfect
    all_sections = partition_sections(all_sections, a, b)

    # Verify that the sections are non-overlapping and cover the entire document
    
    if not is_valid_partition(all_sections, a, b):
        raise AssertionError("Invalid partition")

    # get the section text
    section_dicts = get_sections_text(all_sections, document_lines)

    return section_dicts
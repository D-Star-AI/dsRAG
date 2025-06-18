import os
import time
import logging
from concurrent.futures import ThreadPoolExecutor
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Tuple, Optional

# Assuming these are in a discoverable path or defined elsewhere
from ..utils.imports import instructor
from ..models.types import SemanticSectioningConfig, Line, Section, Element, ElementType, ChunkingConfig

# --- Pydantic Models (from original) ---
class DocumentSection(BaseModel):
    title: str = Field(description="main topic of this section of the document (very descriptive)")
    start_index: int = Field(description="line number where the section begins (inclusive)")

class StructuredDocument(BaseModel):
    """obtains meaningful sections, each centered around a single concept/topic"""
    sections: List[DocumentSection] = Field(description="an ordered list of sections of the document")

# --- Prompts (from original) ---
SYSTEM_PROMPT = """
Read the document below and extract a StructuredDocument object from it where each section of the document is centered around a single concept/topic. Whenever possible, your sections (and section titles) should match up with the natural sections of the document (i.e. Introduction, Conclusion, References, etc.). Sections can vary in length, but should generally be anywhere from a few paragraphs to a few pages long.

Each line of the document is marked with its line number in square brackets (e.g. [1], [2], [3], etc). Use the line numbers to indicate section start.

The start line numbers will be treated as inclusive. For example, if the first line of a section is line 5, the start_index should be 5. Your goal is to find the starting line number of a given section, where a section is a group of lines that are thematically related.

The first section must start at the first line number of the document window provided ({start_line} in this case). The sections MUST cover the entire document window, and they MUST be in order.

Section titles should be descriptive enough such that a person who is just skimming over the section titles and not actually reading the document can get a clear idea of what each section is about.

Note: the document provided to you may just be an excerpt from a larger document, rather than a complete document. Therefore, you can't always assume, for example, that the first line of the document is the beginning of the Introduction section and the last line is the end of the Conclusion section (if those section are even present).
"""

LANGUAGE_ADDENDUM = "For your section titles, YOU MUST use the same language as the document. If the document is in English, your section titles should be in English. If the document is in another language, your section titles should be in that language."

# Get the dsparse logger
logger = logging.getLogger("dsrag.dsparse.semantic_sectioning_parallel") # Changed logger name

# --- Existing Functions (Modified or Reused with new signatures if needed) ---

def get_document_text_for_window(document_lines: List[Line], window_start_line: int, window_end_line: int) -> str:
    """
    Prepares a string representation of a specific slice of document_lines
    (from window_start_line to window_end_line inclusive), prefixing each
    line with its global line number for the LLM prompt.

    Args:
        document_lines: The full list of Line objects for the document.
        window_start_line: The global starting line index for this window.
        window_end_line: The global ending line index for this window.

    Returns:
        A string containing the lines for the window, with global line numbers.
    """
    document_with_line_numbers = ""
    for i in range(window_start_line, min(window_end_line + 1, len(document_lines))):
        line = document_lines[i]["content"]
        document_with_line_numbers += f"[{i}] {line}\n"

    return document_with_line_numbers

def get_structured_document_for_window(
    window_text_with_lines: str,
    first_line_number_in_window_prompt: int,
    llm_provider: str,
    model: str,
    language: str
) -> StructuredDocument:
    """
    Sends a single window's text (with global line numbers in brackets) to the LLM
    and gets back a StructuredDocument. The LLM's returned start_index values
    are expected to be global line numbers.

    Args:
        window_text_with_lines: The text of the current window, with lines
                                prefixed by their global line numbers (e.g., "[101] text...").
        first_line_number_in_window_prompt: The global line number of the first line in this
                                            window, used to format the system prompt.
        llm_provider: The LLM provider (e.g., "openai", "anthropic").
        model: The specific LLM model name.
        language: The language of the document.

    Returns:
        A StructuredDocument object containing sections identified by the LLM for this window.
    """
    formatted_system_prompt = SYSTEM_PROMPT.format(start_line=first_line_number_in_window_prompt)
    if language != "en":
        formatted_system_prompt += "\n" + LANGUAGE_ADDENDUM

    if llm_provider == "anthropic":
        from anthropic import Anthropic
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
                    "content": window_text_with_lines,
                },
            ],
        )
    elif llm_provider == "openai":
        from openai import OpenAI
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
                    "content": window_text_with_lines,
                },
            ],
        )
    elif llm_provider == "gemini":
        import google.generativeai as genai
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        client = instructor.from_gemini(
            client=genai.GenerativeModel(model_name=f"models/{model}"),
            mode=instructor.Mode.GEMINI_JSON
        )
        # For Gemini, prepend the system prompt to the user message
        combined_prompt = f"{formatted_system_prompt}\n\n<document>\n{window_text_with_lines}\n</document>"
        return client.messages.create(
            messages=[
                {
                    "role": "user",
                    "content": combined_prompt
                }
            ],
            response_model=StructuredDocument,
            generation_config={
                "temperature": 0.0,
                "max_output_tokens": 4000
            }
        )
    else:
        raise ValueError("Invalid provider. Must be one of: 'anthropic', 'openai', 'gemini'.")

def validate_and_fix_window_sections(
    sections: List[DocumentSection],
    window_start_line: int,
    window_end_line: int,
    document_length: int
) -> List[DocumentSection]:
    """
    Validates and fixes section indices returned for a single window.
    Ensures they are ordered, unique, and fall within that window's global
    line boundaries (window_start_line to window_end_line). It also ensures
    the first section in the window starts at window_start_line if possible,
    and that sections don't exceed document_length.

    Args:
        sections: List of DocumentSection objects from the LLM for a window.
        window_start_line: The global starting line index of the window.
        window_end_line: The global ending line index of the window.
        document_length: The total number of lines in the entire document.

    Returns:
        A validated and potentially corrected list of DocumentSection objects for the window.
    """
    if not sections:
        # If no sections were identified, create a default one for the entire window
        return [DocumentSection(
            title=f"Window {window_start_line}-{window_end_line}",
            start_index=window_start_line
        )]

    # Remove sections with duplicate start_indices (keep first occurrence)
    seen_indices = set()
    unique_sections = []
    for s in sections:
        if s.start_index not in seen_indices:
            seen_indices.add(s.start_index)
            unique_sections.append(s)
    sections = unique_sections

    # Sort sections by start_index to ensure proper ordering
    original_order = [s.start_index for s in sections]
    sections = sorted(sections, key=lambda x: x.start_index)
    sorted_order = [s.start_index for s in sections]

    logger.debug(f"Window {window_start_line}-{window_end_line}: Original indices: {original_order}, Sorted indices: {sorted_order}")

    # Validate and fix each section's start index
    fixed_sections = []
    last_start = window_start_line - 1  # Initialize to just before window start

    for section in sections:
        original_start = section.start_index

        # Skip sections that start beyond document length or window end
        if original_start >= document_length or original_start > window_end_line:
            logger.debug(f"Window {window_start_line}-{window_end_line}: Skipping section '{section.title}' with invalid start {original_start}")
            continue

        # Ensure start index is valid, after the previous section, and within window bounds
        # For the first section, ensure it starts at window_start_line
        if len(fixed_sections) == 0:
            # First section should start at window_start_line
            start = window_start_line
        else:
            # Subsequent sections should be after prior section and within window
            start = max(last_start + 1, min(section.start_index, window_end_line))

        if start != original_start:
            logger.debug(f"Window {window_start_line}-{window_end_line}: Section '{section.title}' start index adjusted from {original_start} to {start}")

        fixed_sections.append(DocumentSection(
            title=section.title,
            start_index=start
        ))
        last_start = start

    # Ensure we have at least one section
    if not fixed_sections:
        fixed_sections.append(DocumentSection(
            title=f"Window {window_start_line}-{window_end_line}",
            start_index=window_start_line
        ))

    return fixed_sections

def get_sections_text(
    final_sections: List[DocumentSection],
    document_lines: List[Line]
) -> List[Section]:
    """
    Takes the final, globally merged list of DocumentSection objects and the full
    document_lines to populate the 'content' and calculate 'end' indices for each Section.

    Args:
        final_sections: The complete, ordered list of DocumentSection objects for the entire document.
        document_lines: The full list of Line objects for the document.

    Returns:
        A list of Section objects, with content and end indices populated.
    """
    section_dicts = []
    doc_length = len(document_lines)

    for i, s in enumerate(final_sections):
        if i == len(final_sections) - 1:
            end_index = doc_length - 1  # Last section ends at document end
        else:
            end_index = min(final_sections[i+1].start_index - 1, doc_length - 1)  # Section ends right before next section starts

        # Double check bounds
        start_index = min(s.start_index, doc_length - 1)
        end_index = min(end_index, doc_length - 1)

        if start_index > end_index:
            logger.warning(f"Section '{s.title}' has invalid bounds: {start_index} > {end_index}")
            continue

        try:
            contents = [document_lines[j]["content"] for j in range(start_index, end_index+1)]
        except Exception as e:
            logger.error(f"Error in get_sections_text: {e}")
            logger.error(f"Section: {s}")
            logger.error(f"Start: {start_index}, End: {end_index}")
            logger.error(f"Document length: {doc_length}")
            raise e

        section_dicts.append(Section(
            title=s.title,
            content="\n".join(contents),
            start=start_index,
            end=end_index
        ))
    return section_dicts

def split_long_line(line: str, max_line_length: int = 200) -> List[str]:
    """
    Split a long line into multiple shorter lines while trying to preserve word boundaries.
    """
    if len(line) <= max_line_length:
        return [line]

    words = line.split()
    lines = []
    current_line = []
    current_length = 0

    for word in words:
        # +1 for the space that would be added
        if current_length + len(word) + 1 <= max_line_length or not current_line:
            current_line.append(word)
            current_length += len(word) + 1
        else:
            lines.append(" ".join(current_line))
            current_line = [word]
            current_length = len(word)

    if current_line:
        lines.append(" ".join(current_line))

    return lines

def elements_to_lines(
    elements: List[Element],
    exclude_elements: List[str],
    visual_elements: List[str],
    max_line_length: int = 200
) -> List[Line]:
    """
    Converts a list of Element objects into a list of Line objects.

    Args:
        elements: List of Element objects to convert.
        exclude_elements: Types of elements to exclude.
        visual_elements: Types of elements that are visual and shouldn't be split.
        max_line_length: Maximum length for a line before splitting.

    Returns:
        List of Line objects.
    """
    document_lines = []
    for element in elements:
        try:
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
                    if len(line) <= max_line_length:
                        document_lines.append({
                            "content": line,
                            "element_type": element["type"],
                            "page_number": element.get("page_number", None),
                            "is_visual": False,
                        })
                    else:
                        # Only split if line is too long
                        split_lines = split_long_line(line, max_line_length)
                        for split_line in split_lines:
                            document_lines.append({
                                "content": split_line,
                                "element_type": element["type"],
                                "page_number": element.get("page_number", None),
                                "is_visual": False,
                            })
        except Exception as e:
            logger.error(f"Error in elements_to_lines: {e}")
            logger.error(f"Element: {element}")
            raise e

    return document_lines

def str_to_lines(document: str, max_line_length: int = 200) -> List[Line]:
    """
    Converts a document string into a list of Line objects.

    Args:
        document: String to convert.
        max_line_length: Maximum length for a line before splitting.

    Returns:
        List of Line objects.
    """
    document_lines = []
    lines = document.split("\n")
    for line in lines:
        if len(line) <= max_line_length:
            document_lines.append({
                "content": line,
                "element_type": "NarrativeText",
                "page_number": None,
                "is_visual": False,
            })
        else:
            # Only split if line is too long
            split_lines = split_long_line(line, max_line_length)
            for split_line in split_lines:
                document_lines.append({
                    "content": split_line,
                    "element_type": "NarrativeText",
                    "page_number": None,
                    "is_visual": False,
                })

    return document_lines

def pages_to_lines(pages: List[str], max_line_length: int = 200) -> List[Line]:
    """
    Converts a list of page strings into a list of Line objects.

    Args:
        pages: List of page strings to convert.
        max_line_length: Maximum length for a line before splitting.

    Returns:
        List of Line objects.
    """
    document_lines = []
    for i, page in enumerate(pages):
        lines = page.split("\n")
        for line in lines:
            if len(line) <= max_line_length:
                document_lines.append({
                    "content": line,
                    "element_type": "NarrativeText",
                    "page_number": i+1,  # page numbers are 1-indexed
                    "is_visual": False,
                })
            else:
                # Only split if line is too long
                split_lines = split_long_line(line, max_line_length)
                for split_line in split_lines:
                    document_lines.append({
                        "content": split_line,
                        "element_type": "NarrativeText",
                        "page_number": i+1,  # page numbers are 1-indexed
                        "is_visual": False,
                    })

    return document_lines

def no_semantic_sectioning(document_content: str, num_lines: int) -> List[Section]:
    """
    Fallback if semantic sectioning is disabled or fails. Returns the entire
    document content as a single section.

    Args:
        document_content: The full document content as a string.
        num_lines: The number of lines in the document.

    Returns:
        A list containing a single Section covering the whole document.
    """
    # return the entire document as a single section
    return [Section(
        title="",
        content=document_content,
        start=0,
        end=num_lines - 1 # 0-indexed and inclusive
    )]


# --- New Functions for Parallel Processing ---

def create_document_windows(
    document_lines: List[Line],
    max_characters_per_window: int,
) -> List[Tuple[int, int]]:
    """
    Divides the document_lines into a list of non-overlapping windows.
    Each window is represented by a tuple of (global_start_line_index, global_end_line_index),
    inclusive. It aims to keep each window's total character count
    (approximately) under max_characters_per_window.

    Args:
        document_lines: The full list of Line objects for the document.
        max_characters_per_window: The target maximum number of characters for each window.

    Returns:
        A list of tuples, where each tuple is (start_line_idx, end_line_idx)
        defining a window.
    """
    windows = []
    doc_length = len(document_lines)

    if doc_length == 0:
        return windows

    window_start = 0
    character_count = 0

    for i in range(doc_length):
        line = document_lines[i]["content"]
        character_count += len(line)

        # Check if we've reached the max characters or end of document
        # We use 0.9 * max_characters to leave room for line numbers which add to token count
        if character_count >= 0.9 * max_characters_per_window or i == doc_length - 1:
            windows.append((window_start, i))
            window_start = i + 1
            character_count = 0

    # If we have a partial window at the end
    if window_start < doc_length and window_start not in [w[0] for w in windows]:
        windows.append((window_start, doc_length - 1))

    logger.debug(f"Created {len(windows)} document windows")
    return windows

def process_window_with_retries(
    window_text_with_lines: str,
    first_line_number_in_window_prompt: int,
    llm_provider: str,
    model: str,
    language: str,
    max_retries: int = 3,
    initial_delay: float = 5.0,
    backoff_factor: float = 2.0,
    kb_id: str = "",
    doc_id: str = ""
) -> Optional[StructuredDocument]:
    """
    Processes a single document window by calling the LLM, including retry logic
    for transient errors. This function is intended to be run in parallel for
    multiple windows.

    Args:
        window_text_with_lines: The text of the current window for the LLM.
        first_line_number_in_window_prompt: The global line number of the first line
                                            in this window for the system prompt.
        llm_provider: LLM provider name.
        model: LLM model name.
        language: Document language.
        max_retries: Maximum number of retries for the LLM call.
        initial_delay: Initial delay in seconds before the first retry.
        backoff_factor: Factor by which the delay increases for subsequent retries.
        kb_id: Knowledge base identifier (for logging).
        doc_id: Document identifier (for logging).

    Returns:
        A StructuredDocument object if successful, or None if all retries fail.
    """
    # Create base logging context with identifiers
    base_extra = {
        "window_start": first_line_number_in_window_prompt,
        "llm_provider": llm_provider,
        "model": model
    }
    if kb_id:
        base_extra["kb_id"] = kb_id
    if doc_id:
        base_extra["doc_id"] = doc_id

    # Retry logic for LLM call
    current_delay = initial_delay

    for attempt in range(max_retries):
        try:
            logger.debug(f"Processing window starting at line {first_line_number_in_window_prompt} (attempt {attempt+1}/{max_retries})",
                        extra=base_extra)

            # Call the LLM to get structured document
            structured_doc = get_structured_document_for_window(
                window_text_with_lines,
                first_line_number_in_window_prompt,
                llm_provider,
                model,
                language
            )

            # If successful, return the result
            logger.debug(f"Successfully processed window at line {first_line_number_in_window_prompt}",
                        extra=base_extra)
            return structured_doc

        except Exception as e:
            logger.warning(f"Attempt {attempt + 1}/{max_retries} failed for window at line {first_line_number_in_window_prompt}: {e}",
                          extra=base_extra)

            if attempt < max_retries - 1:
                logger.info(f"Retrying window at line {first_line_number_in_window_prompt} in {current_delay:.2f} seconds...",
                           extra=base_extra)
                time.sleep(current_delay)
                current_delay *= backoff_factor
            else:
                # If all retries fail, log the final error
                logger.error(f"All {max_retries} attempts failed for window at line {first_line_number_in_window_prompt}.",
                            extra=base_extra)
                return None

    # This should never be reached due to the return None above, but adding as a safety
    return None

def merge_sections_across_windows(
    all_window_sections: List[List[DocumentSection]]
) -> List[DocumentSection]:
    """
    Merges the lists of sections obtained from processing each window in parallel.
    
    Merging strategy:
    - The last section of window N is merged with the first section of window N+1 ONLY IF:
      1. Both windows contain more than one section, OR
      2. The current window contains more than one section AND the next window is the last window
    - If either window contains only a single section (e.g., a consolidated section),
      no merging occurs and sections remain distinct.

    The title for a merged section will be:
    f"{title_from_window_N_last_section} / {title_from_window_N+1_first_section}"

    Args:
        all_window_sections: A list of lists, where each inner list contains
                             DocumentSection objects for a corresponding window,
                             ordered by window.

    Returns:
        A single list of DocumentSection objects representing the merged sections
        for the entire document.
    """
    if not all_window_sections:
        return []

    # If we only have sections from one window, no merging needed
    if len(all_window_sections) == 1:
        return all_window_sections[0]

    merged_sections = []
    processed_indices = set()  # Track which window indices we've processed
    
    # Process each window
    for i in range(len(all_window_sections)):
        if i in processed_indices:
            continue
            
        current_window_sections = all_window_sections[i]
        
        if not current_window_sections:
            # Skip empty windows (shouldn't happen with our validation)
            continue
            
        if i == len(all_window_sections) - 1:
            # Last window - add all sections
            merged_sections.extend(current_window_sections)
            processed_indices.add(i)
        else:
            # Not the last window - check if we should merge with next window
            next_window_sections = all_window_sections[i + 1] if i + 1 < len(all_window_sections) else None
            
            if not next_window_sections:
                # Next window is empty or doesn't exist, add all current sections
                merged_sections.extend(current_window_sections)
                processed_indices.add(i)
                continue
            
            # Check merging conditions
            current_has_multiple = len(current_window_sections) > 1
            next_has_multiple = len(next_window_sections) > 1
            is_next_last_window = (i + 1) == (len(all_window_sections) - 1)
            
            should_merge = (current_has_multiple and next_has_multiple) or \
                         (current_has_multiple and is_next_last_window)
            
            if should_merge:
                # Add all sections except the last from current window
                merged_sections.extend(current_window_sections[:-1])
                
                # Create merged section from last of current and first of next
                last_section = current_window_sections[-1]
                first_section = next_window_sections[0]
                
                merged_title = f"{last_section.title} / {first_section.title}"
                merged_section = DocumentSection(
                    title=merged_title,
                    start_index=last_section.start_index  # Keep the earlier start_index
                )
                merged_sections.append(merged_section)
                
                # Add remaining sections from next window (skip the first since it was merged)
                if len(next_window_sections) > 1:
                    merged_sections.extend(next_window_sections[1:])
                
                # Mark both windows as processed
                processed_indices.add(i)
                processed_indices.add(i + 1)
            else:
                # No merging - add all sections from current window
                merged_sections.extend(current_window_sections)
                processed_indices.add(i)

    logger.debug(f"Merged {sum(len(sections) for sections in all_window_sections)} sections from {len(all_window_sections)} windows into {len(merged_sections)} sections")

    return merged_sections

def validate_and_fix_global_sections(
    sections: List[DocumentSection],
    document_length: int,
    first_document_line_index: int = 0
) -> List[DocumentSection]:
    """
    Performs a final validation and fixing pass on the globally merged list of sections.
    Ensures sections are strictly ordered, start indices are unique, the first section
    starts at the document's actual first line (typically 0), and sections cover
    the entire document appropriately up to document_length.

    Args:
        sections: The list of globally merged DocumentSection objects.
        document_length: The total number of lines in the entire document.
        first_document_line_index: The starting line index of the document (usually 0).

    Returns:
        A finalized, validated list of DocumentSection objects.
    """
    if not sections:
        # If no sections, create a default one for the entire document
        return [DocumentSection(
            title="Document",
            start_index=first_document_line_index
        )]

    # Remove sections with duplicate start_indices (keep first occurrence)
    seen_indices = set()
    unique_sections = []
    for s in sections:
        if s.start_index not in seen_indices:
            seen_indices.add(s.start_index)
            unique_sections.append(s)
    sections = unique_sections

    # Sort sections by start_index to ensure proper ordering
    original_order = [s.start_index for s in sections]
    sections = sorted(sections, key=lambda x: x.start_index)
    sorted_order = [s.start_index for s in sections]

    if original_order != sorted_order:
        logger.warning(f"Final sections were out of order. Original indices: {original_order}, Sorted indices: {sorted_order}")

    # Validate and fix each section's start index
    fixed_sections = []
    last_start = first_document_line_index - 1  # Initialize to just before document start

    for section in sections:
        original_start = section.start_index

        # Skip sections that start beyond document length
        if original_start >= document_length:
            logger.warning(f"Skipping section '{section.title}' as it starts beyond document length")
            continue

        # Special handling for the first section - it must start at first_document_line_index
        if len(fixed_sections) == 0:
            start = first_document_line_index
        else:
            # Other sections must come after the previous section
            start = max(last_start + 1, min(section.start_index, document_length - 1))

        if start != original_start:
            logger.info(f"Section '{section.title}' start index adjusted from {original_start} to {start}")

        fixed_sections.append(DocumentSection(
            title=section.title,
            start_index=start
        ))
        last_start = start

    # Ensure we have at least one section that starts at the document beginning
    if not fixed_sections or fixed_sections[0].start_index > first_document_line_index:
        fixed_sections.insert(0, DocumentSection(
            title="Document Beginning",
            start_index=first_document_line_index
        ))

    logger.debug(f"Final validation complete: {len(fixed_sections)} sections in document")
    return fixed_sections


def get_sections(
    document_lines: List[Line],
    max_characters_per_window: int,
    llm_provider: str,
    model: str,
    language: str,
    kb_id: str = "",
    doc_id: str = "",
    llm_max_concurrent_requests: int = 5,
    min_avg_chars_per_section: int = 500
) -> List[Section]:
    """
    Orchestrates the parallel semantic sectioning of a document.
    1. Divides the document into windows.
    2. Processes each window in parallel to get sections (with retries).
    3. Validates sections for each window.
    4. Merges sections from adjacent windows.
    5. Performs a final global validation of all merged sections.
    6. Populates section content and end lines.

    Args:
        document_lines: The full list of Line objects for the document.
        max_characters_per_window: Target maximum characters for LLM processing windows.
        llm_provider: LLM provider name.
        model: LLM model name.
        language: Document language.
        kb_id: Knowledge base identifier (for logging).
        doc_id: Document identifier (for logging).
        llm_max_concurrent_requests: Maximum number of concurrent LLM API calls.
        min_avg_chars_per_section: Minimum average characters per section within a window.
            If a window has multiple sections with average length below this threshold,
            they will be consolidated into a single section. Default is 500.

    Returns:
        A list of Section objects for the entire document.
    """
    # Create base logging context with identifiers
    base_extra = {}
    if kb_id:
        base_extra["kb_id"] = kb_id
    if doc_id:
        base_extra["doc_id"] = doc_id

    # Log start of parallel sectioning operation
    start_time = time.perf_counter()
    logger.debug("Starting parallel semantic sectioning", extra={
        **base_extra,
        "document_lines_count": len(document_lines),
        "llm_provider": llm_provider,
        "model": model,
        "llm_max_concurrent_requests": llm_max_concurrent_requests
    })

    # Step 1: Divide document into windows
    doc_windows = create_document_windows(document_lines, max_characters_per_window)

    if not doc_windows:
        logger.warning("No document windows created, document might be empty", extra=base_extra)
        return []

    # Step 2: Process each window in parallel
    window_sections = []

    with ThreadPoolExecutor(max_workers=llm_max_concurrent_requests) as executor:
        window_futures = []

        # Submit all window processing tasks
        for window_idx, (window_start, window_end) in enumerate(doc_windows):
            window_text = get_document_text_for_window(document_lines, window_start, window_end)

            logger.debug(f"Submitting window {window_idx+1}/{len(doc_windows)} for processing",
                        extra={**base_extra, "window_start": window_start, "window_end": window_end})

            future = executor.submit(
                process_window_with_retries,
                window_text,
                window_start,
                llm_provider,
                model,
                language,
                kb_id=kb_id,
                doc_id=doc_id
            )
            window_futures.append((window_idx, window_start, window_end, future))

        # Process results as they complete
        all_window_sections = [None] * len(doc_windows)  # Pre-allocate to maintain window order

        for window_idx, window_start, window_end, future in window_futures:
            try:
                result = future.result()
                if result:
                    # Get window text to calculate character count
                    window_text = get_document_text_for_window(document_lines, window_start, window_end)
                    
                    # Character-based safeguard against excessive sections
                    if len(result.sections) > 1:
                        avg_chars = len(window_text) / len(result.sections)
                        if avg_chars < min_avg_chars_per_section:
                            logger.warning(
                                f"Window {window_idx+1}/{len(doc_windows)} has {len(result.sections)} sections "
                                f"with only {avg_chars:.0f} avg chars per section (below {min_avg_chars_per_section}). "
                                f"Consolidating into single section.",
                                extra={**base_extra, "window_start": window_start, "window_end": window_end}
                            )
                            # Replace with single consolidated section
                            result.sections = [DocumentSection(
                                title="Consolidated Section",
                                start_index=window_start
                            )]
                    # Step 3: Validate sections from this window
                    validated_sections = validate_and_fix_window_sections(
                        result.sections, window_start, window_end, len(document_lines)
                    )

                    logger.debug(f"Window {window_idx+1}/{len(doc_windows)} processed successfully with {len(validated_sections)} sections",
                                extra={**base_extra, "window_start": window_start, "window_end": window_end})

                    # Store in the correct position to maintain window order
                    all_window_sections[window_idx] = validated_sections
                else:
                    logger.warning(f"Window {window_idx+1}/{len(doc_windows)} processing failed",
                                  extra={**base_extra, "window_start": window_start, "window_end": window_end})
                    # Provide a fallback section for this window
                    all_window_sections[window_idx] = [DocumentSection(
                        title=f"Window {window_start}-{window_end}",
                        start_index=window_start
                    )]
            except Exception as e:
                logger.error(f"Error processing window {window_idx+1}/{len(doc_windows)}: {e}",
                           extra={**base_extra, "window_start": window_start, "window_end": window_end})
                # Provide a fallback section for this window
                all_window_sections[window_idx] = [DocumentSection(
                    title=f"Window {window_start}-{window_end} (Error)",
                    start_index=window_start
                )]

    # Remove any None values in case of unexpected issues
    all_window_sections = [sections for sections in all_window_sections if sections is not None]

    # Step 4: Merge sections from all windows
    merged_sections = merge_sections_across_windows(all_window_sections)

    # Step 5: Perform final global validation
    final_sections = validate_and_fix_global_sections(merged_sections, len(document_lines))

    # Step 6: Populate section content and end lines
    result_sections = get_sections_text(final_sections, document_lines)

    # Calculate and log overall duration
    total_duration = time.perf_counter() - start_time
    logger.debug("Parallel semantic sectioning complete", extra={
        **base_extra,
        "total_duration_s": round(total_duration, 4),
        "windows_count": len(doc_windows),
        "sections_count": len(result_sections)
    })

    return result_sections


# --- Main Entry Point Functions (Parallel Versions) ---

def get_sections_from_elements(
    elements: List[Element],
    element_types: List[ElementType],
    exclude_elements: List[str] = [],
    max_characters_per_window: int = 20000,
    semantic_sectioning_config: Dict[str, Any] = None,
    chunking_config: Dict[str, Any] = None,
    kb_id: str = "",
    doc_id: str = ""
) -> Tuple[List[Section], List[Line]]:
    """
    Generates sections from a list of document elements using parallel processing.
    Converts elements to lines, then calls parallel_get_sections.

    Args:
        elements: List of Element objects representing the document content.
        element_types: List of ElementType definitions.
        exclude_elements: Types of elements to exclude from processing.
        max_characters_per_window: Maximum characters per processing window.
        semantic_sectioning_config: Configuration for semantic sectioning.
        chunking_config: Configuration for document chunking.
        kb_id: Knowledge base identifier (for logging).
        doc_id: Document identifier (for logging).

    Returns:
        A tuple of (sections, document_lines).
    """
    if semantic_sectioning_config is None:
        semantic_sectioning_config = {}
    if chunking_config is None:
        chunking_config = {}

    # Get the semantic sectioning config params, using defaults if not provided
    use_semantic_sectioning = semantic_sectioning_config.get("use_semantic_sectioning", True)
    llm_provider = semantic_sectioning_config.get("llm_provider", "openai")
    model = semantic_sectioning_config.get("model", "gpt-4o-mini")
    language = semantic_sectioning_config.get("language", "en")
    llm_max_concurrent_requests = semantic_sectioning_config.get("llm_max_concurrent_requests", 5)
    min_avg_chars_per_section = semantic_sectioning_config.get("min_avg_chars_per_section", 500)
    min_length_for_chunking = chunking_config.get("min_length_for_chunking", 0)

    visual_elements = [e["name"] for e in element_types if e["is_visual"]]

    # Convert elements to lines
    document_lines = elements_to_lines(elements=elements, exclude_elements=exclude_elements, visual_elements=visual_elements)
    document_lines_str = [line["content"] for line in document_lines]
    document_str = "\n".join(document_lines_str)

    # Check if we should use semantic sectioning
    if use_semantic_sectioning and len(document_str) > min_length_for_chunking:
        sections = get_sections(
            document_lines=document_lines,
            max_characters_per_window=max_characters_per_window,
            llm_provider=llm_provider,
            model=model,
            language=language,
            kb_id=kb_id,
            doc_id=doc_id,
            llm_max_concurrent_requests=llm_max_concurrent_requests,
            min_avg_chars_per_section=min_avg_chars_per_section
        )
    else:
        # Fallback to no semantic sectioning
        sections = no_semantic_sectioning(document_content=document_str, num_lines=len(document_lines))

    return sections, document_lines

def get_sections_from_str(
    document: str,
    max_characters_per_window: int = 20000,
    semantic_sectioning_config: Dict[str, Any] = None,
    chunking_config: Dict[str, Any] = None,
    kb_id: str = "",
    doc_id: str = ""
) -> Tuple[List[Section], List[Line]]:
    """
    Generates sections from a document string using parallel processing.
    Converts the string to lines, then calls parallel_get_sections.

    Args:
        document: Document content as a string.
        max_characters_per_window: Maximum characters per processing window.
        semantic_sectioning_config: Configuration for semantic sectioning.
        chunking_config: Configuration for document chunking.
        kb_id: Knowledge base identifier (for logging).
        doc_id: Document identifier (for logging).

    Returns:
        A tuple of (sections, document_lines).
    """
    if semantic_sectioning_config is None:
        semantic_sectioning_config = {}
    if chunking_config is None:
        chunking_config = {}

    # Get the semantic sectioning config params, using defaults if not provided
    use_semantic_sectioning = semantic_sectioning_config.get("use_semantic_sectioning", True)
    llm_provider = semantic_sectioning_config.get("llm_provider", "openai")
    model = semantic_sectioning_config.get("model", "gpt-4o-mini")
    language = semantic_sectioning_config.get("language", "en")
    llm_max_concurrent_requests = semantic_sectioning_config.get("llm_max_concurrent_requests", 5)
    min_avg_chars_per_section = semantic_sectioning_config.get("min_avg_chars_per_section", 500)
    min_length_for_chunking = chunking_config.get("min_length_for_chunking", 0)

    # Convert string to lines
    document_lines = str_to_lines(document)

    # Check if we should use semantic sectioning
    if use_semantic_sectioning and len(document) > min_length_for_chunking:
        sections = get_sections(
            document_lines=document_lines,
            max_characters_per_window=max_characters_per_window,
            llm_provider=llm_provider,
            model=model,
            language=language,
            kb_id=kb_id,
            doc_id=doc_id,
            llm_max_concurrent_requests=llm_max_concurrent_requests,
            min_avg_chars_per_section=min_avg_chars_per_section
        )
    else:
        # Fallback to no semantic sectioning
        sections = no_semantic_sectioning(document_content=document, num_lines=len(document_lines))

    return sections, document_lines

def get_sections_from_pages(
    pages: List[str],
    max_characters_per_window: int = 20000,
    semantic_sectioning_config: Dict[str, Any] = None,
    chunking_config: Dict[str, Any] = None,
    kb_id: str = "",
    doc_id: str = ""
) -> Tuple[List[Section], List[Line]]:
    """
    Generates sections from a list of page strings using parallel processing.
    Converts pages to lines, then calls parallel_get_sections.

    Args:
        pages: List of page strings.
        max_characters_per_window: Maximum characters per processing window.
        semantic_sectioning_config: Configuration for semantic sectioning.
        chunking_config: Configuration for document chunking.
        kb_id: Knowledge base identifier (for logging).
        doc_id: Document identifier (for logging).

    Returns:
        A tuple of (sections, document_lines).
    """
    if semantic_sectioning_config is None:
        semantic_sectioning_config = {}
    if chunking_config is None:
        chunking_config = {}

    # Get the semantic sectioning config params, using defaults if not provided
    use_semantic_sectioning = semantic_sectioning_config.get("use_semantic_sectioning", True)
    llm_provider = semantic_sectioning_config.get("llm_provider", "openai")
    model = semantic_sectioning_config.get("model", "gpt-4o-mini")
    language = semantic_sectioning_config.get("language", "en")
    llm_max_concurrent_requests = semantic_sectioning_config.get("llm_max_concurrent_requests", 5)
    min_avg_chars_per_section = semantic_sectioning_config.get("min_avg_chars_per_section", 500)
    min_length_for_chunking = chunking_config.get("min_length_for_chunking", 0)

    # Convert pages to lines
    document_lines = pages_to_lines(pages)
    document_lines_str = [line["content"] for line in document_lines]
    document_str = "\n".join(document_lines_str)

    # Check if we should use semantic sectioning
    if use_semantic_sectioning and len(document_str) > min_length_for_chunking:
        sections = get_sections(
            document_lines=document_lines,
            max_characters_per_window=max_characters_per_window,
            llm_provider=llm_provider,
            model=model,
            language=language,
            kb_id=kb_id,
            doc_id=doc_id,
            llm_max_concurrent_requests=llm_max_concurrent_requests,
            min_avg_chars_per_section=min_avg_chars_per_section
        )
    else:
        # Fallback to no semantic sectioning
        sections = no_semantic_sectioning(document_content=document_str, num_lines=len(document_lines))

    return sections, document_lines
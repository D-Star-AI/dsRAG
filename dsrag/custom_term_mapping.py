import re
from typing import List, Dict
from pydantic import BaseModel, Field
from dsrag.utils.llm import get_response

FIND_TARGET_TERMS_PROMPT = """
Your task is to take a list of target terms and return a list of all instances of those terms in the provided text. The instances you return must be direct matches to the text, but can be fuzzy matches to the terms. In other words, you should allow for some variation in spelling and capitalization relative to the terms. For example, if the synonym is "capital of France", and the test has the phrase "Capitol of Frances", you should return the phrase "Capitol of Frances".

<target_terms>
{target_terms}
</target_terms>

<text>
{text}
</text>
""".strip()

class Terms(BaseModel):
    terms: List[str] = Field(description="A list of target terms (or variations thereof) that were found in the text")

def find_target_terms_batch(text: str, target_terms: List[str]) -> List[str]:
    """Find fuzzy matches of target terms in a batch of text using LLM."""
    target_terms_str = "\n".join(target_terms)
    terms = get_response(
        prompt=FIND_TARGET_TERMS_PROMPT.format(target_terms=target_terms_str, text=text),
        response_model=Terms,
    ).terms
    return terms

def find_all_term_variations(chunks: List[str], target_terms: List[str]) -> List[str]:
    """Find all variations of target terms across all chunks, processing in batches."""
    all_variations = set()
    batch_size = 15  # Process 15 chunks at a time
    
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i:i + batch_size]
        batch_text = "\n\n".join(batch_chunks)
        variations = find_target_terms_batch(batch_text, target_terms)
        all_variations.update(variations)
    
    return list(all_variations)

def annotate_chunk(chunk: str, key: str, terms: List[str]) -> str:
    """Annotate a single chunk with terms and their mapping."""
    # Process instances in reverse order to avoid messing up string indices
    for term in terms:
        instances = re.finditer(re.escape(term), chunk)
        instances = list(instances)
        for instance in reversed(instances):
            start = instance.start()
            end = instance.end()
            chunk = chunk[:end] + f" ({key})" + chunk[end:]
    return chunk.strip()

def annotate_chunks(chunks: List[str], custom_term_mapping: Dict[str, List[str]]) -> List[str]:
    """
    Annotate chunks with custom term mappings.
    
    Args:
        chunks: list[str] - list of all chunks in the document
        custom_term_mapping: dict - a dictionary of custom term mapping for the document
            - key: str - the term to map to
            - value: list[str] - the list of terms to map to the key
    
    Returns:
        list[str] - annotated chunks
    """
    # First, find all variations for each key's terms
    term_variations = {}
    for key, target_terms in custom_term_mapping.items():
        variations = find_all_term_variations(chunks, target_terms)
        term_variations[key] = list(set(variations + target_terms))  # Include original terms
        # remove an exact match of the key from the list of terms
        term_variations[key] = [term for term in term_variations[key] if term != key]
    
    # Then annotate each chunk with all terms
    annotated_chunks = []
    for chunk in chunks:
        annotated_chunk = chunk
        for key, terms in term_variations.items():
            annotated_chunk = annotate_chunk(annotated_chunk, key, terms)
        annotated_chunks.append(annotated_chunk)
    
    return annotated_chunks
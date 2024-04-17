from spRAG import KnowledgeBase
import numpy as np
import pickle
import time

def get_rse_params(segment_length: str) -> dict:
    if segment_length == 'very_short':
        return {
            'irrelevant_chunk_penalty': 0.4,
            'max_length': 4,
            'overall_max_length': 14,
            'overall_max_length_extension': 3,
            'minimum_value': 0.2,
            'max_queries': 3,
        }
    elif segment_length == 'short':
        return {
            'irrelevant_chunk_penalty': 0.3,
            'max_length': 7,
            'overall_max_length': 18,
            'overall_max_length_extension': 4,
            'minimum_value': 0.5,
            'max_queries': 3,
        }
    elif segment_length == 'medium':
        return {
            'irrelevant_chunk_penalty': 0.2,
            'max_length': 10,
            'overall_max_length': 20,
            'overall_max_length_extension': 5,
            'minimum_value': 0.7,
            'max_queries': 2,
        }
    elif segment_length == 'long':
        return {
            'irrelevant_chunk_penalty': 0.12,
            'max_length': 15,
            'overall_max_length': 25,
            'overall_max_length_extension': 6,
            'minimum_value': 0.8,
            'max_queries': 2,
        }
    elif segment_length == 'very_long':
        return {
            'irrelevant_chunk_penalty': 0.08,
            'max_length': 25,
            'overall_max_length': 40,
            'overall_max_length_extension': 8,
            'minimum_value': 0.9,
            'max_queries': 2,
        }
    else:
        raise ValueError(f'Invalid segment length: {segment_length}')

def search_kb(kb_id: str, query: str, top_k: int) -> list:
    """
    Search a knowledge base with a query and return the top k results, in order
    - search_results is a list of dictionaries, where each dictionary has the following keys:
        - ["metadata"]["doc_id"]
        - ["metadata"]["chunk_index"]
        - ["similarity"]
        - ["content"]
    """
    kb = KnowledgeBase(kb_id)
    search_results = kb.search(query, top_k)
    return search_results

def get_segment_text_from_database(doc_id: str, chunk_start: int, chunk_end: int) -> str:
    # figure out which kb_id this doc_id belongs to by loading the mapping from disk and then load the KB
    with open('~/spRAG/doc_id_to_kb_id.pkl', 'rb') as f:
        doc_id_to_kb_id = pickle.load(f)
    kb_id = doc_id_to_kb_id[doc_id]
    kb = KnowledgeBase(kb_id) # load the kb

    # get the text for the segment
    segment = f"[{kb.get_chunk_header(doc_id, chunk_start)}]\n" # initialize the segment with the chunk header
    for chunk_index in range(chunk_start, chunk_end): # NOTE: end index is non-inclusive
        chunk_text = kb.get_chunk(doc_id, chunk_index)
        segment += chunk_text

    return segment.strip()

def get_best_segments(all_relevance_values: list[list], document_splits: list[int], max_length: int, overall_max_length: int, minimum_value: float) -> list[tuple]:
    """
    This function takes the chunk relevance values and then runs an optimization algorithm to find the best segments.

    - all_relevance_values: a list of lists of relevance values for each chunk of a meta-document, with each outer list representing a query
    - document_splits: a list of indices that represent the start of each document - best segments will not overlap with these

    Returns
    - best_segments: a list of tuples (start, end) that represent the indices of the best segments (the end index is non-inclusive) in the meta-document
    """
    best_segments = []
    total_length = 0
    rv_index = 0
    bad_rv_indices = []
    while total_length < overall_max_length:
        # cycle through the queries
        if rv_index >= len(all_relevance_values):
            rv_index = 0
        # if none of the queries have any more valid segments, we're done
        if len(bad_rv_indices) >= len(all_relevance_values):
            break        
        # check if we've already determined that there are no more valid segments for this query - if so, skip it
        if rv_index in bad_rv_indices:
            rv_index += 1
            continue
        
        # find the best remaining segment for this query
        relevance_values = all_relevance_values[rv_index] # get the relevance values for this query
        best_segment = None
        best_value = -1000
        for start in range(len(relevance_values)):
            # skip over negative value starting points
            if relevance_values[start] < 0:
                continue
            for end in range(start+1, min(start+max_length+1, len(relevance_values)+1)):
                # skip over negative value ending points
                if relevance_values[end-1] < 0:
                    continue
                # check if this segment overlaps with any of the best segments
                if any(start < seg_end and end > seg_start for seg_start, seg_end in best_segments):
                    continue
                # check if this segment overlaps with any of the document splits
                if any(start < split and end > split for split in document_splits):
                    continue
                # check if this segment would push us over the overall max length
                if total_length + end - start > overall_max_length:
                    continue
                segment_value = sum(relevance_values[start:end]) # define segment value as the sum of the relevance values of its chunks
                if segment_value > best_value:
                    best_value = segment_value
                    best_segment = (start, end)
        
        # if we didn't find a valid segment, mark this query as done
        if best_segment is None or best_value < minimum_value:
            bad_rv_indices.append(rv_index)
            rv_index += 1
            continue

        # otherwise, add the segment to the list of best segments
        best_segments.append(best_segment)
        total_length += best_segment[1] - best_segment[0]
        rv_index += 1
    
    return best_segments

# define the value of a given rank
def convert_rank_to_value(rank: int, irrelevant_chunk_penalty: float, decay_rate: int = 20):
    """
    The irrelevant_chunk_penalty term has the effect of controlling how large of segments are created:
    - 0.05 gives very long segments of 20-50 chunks
    - 0.1 gives long segments of 10-20 chunks
    - 0.2 gives medium segments of 4-10 chunks
    - 0.3 gives short segments of 2-6 chunks
    - 0.4 gives very short segments of 1-3 chunks
    """
    return np.exp(-rank / decay_rate) - irrelevant_chunk_penalty

def get_all_ranked_results(search_queries: list[tuple]):
    """
    - search_queries: list of tuples of the form (kb_id, query)
    """
    all_ranked_results = []
    for kb_id, query in search_queries:
        ranked_results = search_kb(kb_id, query, 100)
        all_ranked_results.append(ranked_results)
    return all_ranked_results

def get_meta_document(all_ranked_results: list[list], top_k_for_document_selection: int = 7):
    # get the top_k results for each query - and the document IDs for the top results across all queries
    top_document_ids = []
    for ranked_results in all_ranked_results:
        top_document_ids.extend([result["metadata"]["doc_id"] for result in ranked_results[:top_k_for_document_selection]]) # get document IDs for top results for each query
    unique_document_ids = list(set(top_document_ids)) # get the unique document IDs for the top results across all queries

    # get the max chunk index for each document and use this to get the document splits and document start points for the meta-document (i.e. the concatenation of all the documents)
    document_splits = [] # indices that represent the (non-inclusive) end of each document in the meta-document
    document_start_points = {} # index of the first chunk of each document in the meta-document, keyed on document_id
    for document_id in unique_document_ids:
        max_chunk_index = -1
        for ranked_results in all_ranked_results:
            for result in ranked_results:
                if result["metadata"]["doc_id"] == document_id:
                    max_chunk_index = max(max_chunk_index, result["metadata"]["chunk_index"])
        document_start_points[document_id] = document_splits[-1] if document_splits else 0
        document_splits.append(int(max_chunk_index + document_splits[-1] + 1 if document_splits else max_chunk_index + 1)) # basically the start point of the next document

    return document_splits, document_start_points, unique_document_ids

def get_relevance_values(all_ranked_results: list[list], meta_document_length: int, document_start_points: dict[str, int], unique_document_ids: list[str], irrelevant_chunk_penalty: float, decay_rate: int = 20):
    # get the relevance values for each chunk in the meta-document, separately for each query
    all_relevance_values = []
    for ranked_results in all_ranked_results:
        # loop through the top results for each query and add their rank to the relevance ranks list
        relevance_ranks = 1000 * np.ones(meta_document_length) # initialize all chunks to rank 1000 - this is the rank we give to chunks that are not in the top k results
        for rank, result in enumerate(ranked_results):
            document_id = result["metadata"]["doc_id"]
            if document_id not in unique_document_ids:
                continue
            chunk_index = int(result["metadata"]["chunk_index"])
            meta_document_index = int(document_start_points[document_id] + chunk_index) # find the correct index for this chunk in the meta-document
            relevance_ranks[meta_document_index] = rank

        # convert the relevance ranks to relevance values using the convert_rank_to_value function, which uses an exponential decay function to define the value of a given rank
        relevance_values = [convert_rank_to_value(rank, irrelevant_chunk_penalty, decay_rate) for rank in relevance_ranks]
        all_relevance_values.append(relevance_values)
    
    return all_relevance_values

def relevant_segment_extraction(search_queries: list[tuple], max_length: int, overall_max_length: int, minimum_value: float, irrelevant_chunk_penalty: float, overall_max_length_extension: int = 0, decay_rate: int = 20, top_k_for_document_selection: int = 7, latency_profiling: bool = False) -> list[dict]:
    """
    Inputs:
    - search_queries: list of tuples of the form (kb_id, query)
    - max_length: maximum length of a segment, measured in number of chunks
    - overall_max_length: maximum length of all segments combined, measured in number of chunks
    - minimum_value: minimum value of a segment, measured in relevance value
    - irrelevant_chunk_penalty: float between 0 and 1
    - overall_max_length_extension: the maximum length of all segments combined will be increased by this amount for each additional query beyond the first

    Returns relevant_segment_info, a list of segment_info dictionaries that each contain:
    - doc_id: the document ID of the document that the segment is from
    - chunk_start: the start index of the segment in the document
    - chunk_end: the (non-inclusive) end index of the segment in the document
    - text: the full text of the segment
    """

    overall_max_length += (len(search_queries) - 1) * overall_max_length_extension # increase the overall max length for each additional query

    start_time = time.time()
    all_ranked_results = get_all_ranked_results(search_queries=search_queries)
    if latency_profiling:
        print(f"get_all_ranked_results took {time.time() - start_time} seconds to run for {len(search_queries)} queries")

    document_splits, document_start_points, unique_document_ids = get_meta_document(all_ranked_results=all_ranked_results, top_k_for_document_selection=top_k_for_document_selection)

    # verify that we have a valid meta-document - otherwise return an empty list of segments
    if len(document_splits) == 0:
        return []
    
    # get the length of the meta-document so we don't have to pass in the whole list of splits
    meta_document_length = document_splits[-1]

    all_relevance_values = get_relevance_values(all_ranked_results=all_ranked_results, meta_document_length=meta_document_length, document_start_points=document_start_points, unique_document_ids=unique_document_ids, irrelevant_chunk_penalty=irrelevant_chunk_penalty, decay_rate=decay_rate)
    best_segments = get_best_segments(all_relevance_values=all_relevance_values, document_splits=document_splits, max_length=max_length, overall_max_length=overall_max_length, minimum_value=minimum_value)
    
    # convert the best segments into a list of dictionaries that contain the document id and the start and end of the chunk
    relevant_segment_info = []
    for start, end in best_segments:
        # find the document that this segment starts in
        for i, split in enumerate(document_splits):
            if start < split: # splits represent the end of each document
                doc_start = document_splits[i-1] if i > 0 else 0
                relevant_segment_info.append({"doc_id": unique_document_ids[i], "chunk_start": start - doc_start, "chunk_end": end - doc_start}) # NOTE: end index is non-inclusive
                break
    
    # get the relevant segments by concatenating their chunks - this is where we need to retrieve the actual text from the database
    for segment_info in relevant_segment_info:
        segment_info["text"] = (get_segment_text_from_database(segment_info["doc_id"], segment_info["chunk_start"], segment_info["chunk_end"])) # NOTE: this is where the chunk header is added to the segment text

    return relevant_segment_info


if __name__ == "__main__":
    # integration test
    kb_id = "snowflake_10k"
    search_queries = [(kb_id, "What is Snowflake's core product?")]
    max_length = 10
    overall_max_length = 15
    minimum_value = 0.5
    irrelevant_chunk_penalty = 0.2
    relevant_segment_info = relevant_segment_extraction(search_queries=search_queries, max_length=max_length, overall_max_length=overall_max_length, minimum_value=minimum_value, irrelevant_chunk_penalty=irrelevant_chunk_penalty)
    for segment_info in relevant_segment_info:
        print(segment_info)
        print()
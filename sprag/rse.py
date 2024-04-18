import numpy as np

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
import numpy as np

def get_best_segments(all_relevance_values: list[list], document_splits: list[int], max_length: int, overall_max_length: int, minimum_value: float):
    """
    This function takes the chunk relevance values and then runs an optimization algorithm to find the best segments.

    - all_relevance_values: a list of lists of relevance values for each chunk of a meta-document, with each outer list representing a query
    - document_splits: a list of indices that represent the start of each document - best segments will not overlap with these

    Returns
    - best_segments: a list of tuples (start, end) that represent the indices of the best segments (the end index is non-inclusive) in the meta-document
    - scores: a list of the scores for each of the best segments
    """
    best_segments = []
    scores = []
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
        scores.append(best_value)
        total_length += best_segment[1] - best_segment[0]
        rv_index += 1
    
    return best_segments, scores

def get_meta_document(all_ranked_results: list[list], top_k_for_document_selection: int):
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

# define the value of a given rank
def get_chunk_value(chunk_info: dict, irrelevant_chunk_penalty: float, decay_rate: int):
    """
    The irrelevant_chunk_penalty term has the effect of controlling how large of segments are created:
    - 0.05 gives very long segments of 20-50 chunks
    - 0.1 gives long segments of 10-20 chunks
    - 0.2 gives medium segments of 4-10 chunks
    - 0.3 gives short segments of 2-6 chunks
    - 0.4 gives very short segments of 1-3 chunks
    """

    rank = chunk_info.get('rank', 1000) # if rank is not provided, default to 1000
    absolute_relevance_value = chunk_info.get('absolute_relevance_value', 0.0) # if absolute_relevance_value is not provided, default to 0.0
    
    v = np.exp(-rank / decay_rate)*absolute_relevance_value - irrelevant_chunk_penalty
    return v

def get_relevance_values(all_ranked_results: list[list], meta_document_length: int, document_start_points: dict[str, int], unique_document_ids: list[str], irrelevant_chunk_penalty: float, decay_rate: int = 20, chunk_length_adjustment = True):
    # get the relevance values for each chunk in the meta-document, separately for each query
    all_relevance_values = []
    for ranked_results in all_ranked_results:
        
        # loop through the top results for each query and add their rank to the relevance ranks list
        all_chunk_info = [{} for _ in range(meta_document_length)]
        for rank, result in enumerate(ranked_results):
            document_id = result["metadata"]["doc_id"]
            if document_id not in unique_document_ids:
                continue
            
            chunk_index = int(result["metadata"]["chunk_index"])
            meta_document_index = int(document_start_points[document_id] + chunk_index) # find the correct index for this chunk in the meta-document
            absolute_relevance_value = result["similarity"]
            chunk_length = len(result["metadata"]["chunk_text"]) # get the length of the chunk in characters
            all_chunk_info[meta_document_index] = {'rank': rank, 'absolute_relevance_value': absolute_relevance_value, 'chunk_length': chunk_length}

        # convert the relevance ranks and other info to chunk values
        relevance_values = [get_chunk_value(chunk_info, irrelevant_chunk_penalty, decay_rate) for chunk_info in all_chunk_info]

        if chunk_length_adjustment:
            # adjust the relevance values for the length of the chunks
            chunk_lengths = [chunk_info.get('chunk_length', 0.0) for chunk_info in all_chunk_info]
            relevance_values = adjust_relevance_values_for_chunk_length(relevance_values, chunk_lengths)

        all_relevance_values.append(relevance_values)
    
    return all_relevance_values

def adjust_relevance_values_for_chunk_length(relevance_values: list[float], chunk_lengths: list[int], reference_length: int = 700):
    """
    Scale the chunk values by chunk length relative to the reference length
    - reference_length is the length of a standard chunk, measured in number of characters (default is 700 characters, because this is the average length of a chunk when you set the max to 800, which is the default.)
    """
    assert len(relevance_values) == len(chunk_lengths), "The length of relevance_values and chunk_lengths must be the same"
    adjusted_relevance_values = []
    for relevance_value, chunk_length in zip(relevance_values, chunk_lengths):
        bounded_chunk_length = max(chunk_length, reference_length) # only adjust relevance values for chunks that are longer than the reference length
        adjusted_relevance_values.append(relevance_value * (bounded_chunk_length / reference_length))
    return adjusted_relevance_values

RSE_PARAMS_PRESETS = {
    "balanced": {
        'max_length': 15,
        'overall_max_length': 30,
        'minimum_value': 0.5,
        'irrelevant_chunk_penalty': 0.18,
        'overall_max_length_extension': 5,
        'decay_rate': 30,
        'top_k_for_document_selection': 10,
        'chunk_length_adjustment': True,
    },
    "precision": {
        'max_length': 15,
        'overall_max_length': 30,
        'minimum_value': 0.7,
        'irrelevant_chunk_penalty': 0.2,
        'overall_max_length_extension': 5,
        'decay_rate': 30,
        'top_k_for_document_selection': 10,
        'chunk_length_adjustment': True,
    },
    "find_all": {
        'max_length': 40,
        'overall_max_length': 200,
        'minimum_value': 0.4,
        'irrelevant_chunk_penalty': 0.18,
        'overall_max_length_extension': 0,
        'decay_rate': 200,
        'top_k_for_document_selection': 200,
        'chunk_length_adjustment': True,
    },
}
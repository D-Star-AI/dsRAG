import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from sprag.auto_query import get_search_queries
import pandas as pd

if __name__ == "__main__":
    AUTO_QUERY_GUIDANCE = """
    The knowledge base contains SEC filings for publicly traded companies, like 10-Ks, 10-Qs, and 8-Ks. Keep this in mind when generating search queries. The things you search for should be things that are likely to be found in these documents.

    When deciding what to search for, first consider the pieces of information that will be needed to answer the question. Then, consider what to search for to find those pieces of information. For example, if the question asks what the change in revenue was from 2019 to 2020, you would want to search for the 2019 and 2020 revenue numbers in two separate search queries, since those are the two separate pieces of information needed. You should also think about where you are most likely to find the information you're looking for. If you're looking for assets and liabilities, you may want to search for the balance sheet, for example.
    """.strip()

    dataset_file_path = "/Users/zach/Code/spRAG/tests/data/financebench_sample_150.csv"

    # Read in the data
    df = pd.read_csv(dataset_file_path)

    # get the questions and answers from the question column - turn into lists
    questions = df.question.tolist()
    answers = df.answer.tolist()

    for i in range(3):
        question = questions[i]
        answer = answers[i]
        search_queries = get_search_queries(question, max_queries=6, auto_query_guidance=AUTO_QUERY_GUIDANCE)
        assert len(search_queries) <= 6, "Should return at most 6 search queries"
        assert all(isinstance(query, str) for query in search_queries), "All search queries should be strings"
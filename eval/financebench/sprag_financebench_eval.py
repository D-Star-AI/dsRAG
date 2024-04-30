import pandas as pd
import os
import sys

# add ../../ to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# import the necessary modules
from sprag.knowledge_base import KnowledgeBase
from sprag.auto_query import get_search_queries
from sprag.llm import OpenAIChatAPI

AUTO_QUERY_GUIDANCE = """
The knowledge base contains SEC filings for publicly traded companies, like 10-Ks, 10-Qs, and 8-Ks. Keep this in mind when generating search queries. The things you search for should be things that are likely to be found in these documents.

When deciding what to search for, first consider the pieces of information that will be needed to answer the question. Then, consider what to search for to find those pieces of information. For example, if the question asks what the change in revenue was from 2019 to 2020, you would want to search for the 2019 and 2020 revenue numbers in two separate search queries, since those are the two separate pieces of information needed. You should also think about where you are most likely to find the information you're looking for. If you're looking for assets and liabilities, you may want to search for the balance sheet, for example.
""".strip()

RESPONSE_SYSTEM_MESSAGE = """
You are a response generation system. Please generate a response to the user input based on the provided context. Your response should be as concise as possible while still fully answering the user's question.

CONTEXT
{context}
""".strip()

def get_response(question: str, context: str):
    client = OpenAIChatAPI(model="gpt-4-turbo", temperature=0.0)
    chat_messages = [{"role": "system", "content": RESPONSE_SYSTEM_MESSAGE.format(context=context)}, {"role": "user", "content": question}]
    return client.make_llm_call(chat_messages)

# Get the absolute path of the script file
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the absolute path to the dataset file
dataset_file_path = os.path.join(script_dir, "../../tests/data/financebench_sample_150.csv")

# Read in the data
df = pd.read_csv(dataset_file_path)

# get the questions and answers from the question column - turn into lists
questions = df.question.tolist()
answers = df.answer.tolist()

# load the knowledge base
kb = KnowledgeBase("finance_bench")

# set parameters for relevant segment extraction - slightly longer context than default
rse_params = {
    'max_length': 12,
    'overall_max_length': 30,
    'overall_max_length_extension': 6,
}

# adjust range if you only want to run a subset of the questions (there are 150 total)
for i in range(150):
    print (f"Question {i+1}")
    question = questions[i]
    answer = answers[i]
    search_queries = get_search_queries(question, max_queries=6, auto_query_guidance=AUTO_QUERY_GUIDANCE)
    relevant_segments = kb.query(search_queries)
    context = "\n\n".join([segment['text'] for segment in relevant_segments])
    response = get_response(question, context)
    print (f"\nQuestion: {question}")
    print (f"\nSearch queries: {search_queries}")
    print (f"\nModel response: {response}")
    print (f"\nGround truth answer: {answer}")
    print ("\n---\n")
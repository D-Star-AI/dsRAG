import os
import instructor
from anthropic import Anthropic
from pydantic import BaseModel
from typing import List

SYSTEM_MESSAGE = """
You are a query generation system. Please generate one or more search queries (up to a maximum of {max_queries}) based on the provided user input. DO NOT generate the answer, just queries.

Each of the queries you generate will be used to search a knowledge base for information that can be used to respond to the user input. Make sure each query is specific enough to return relevant information. If multiple pieces of information would be useful, you should generate multiple queries, one for each specific piece of information needed.

{auto_query_guidance}
""".strip()


def get_search_queries(user_input: str, auto_query_guidance: str = "", max_queries: int = 5):
    base_url = os.environ.get("DSRAG_ANTHROPIC_BASE_URL", None)
    if base_url is not None:
        client = instructor.from_anthropic(Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"], base_url=base_url))
    else:
        client = instructor.from_anthropic(Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"]))

    class Queries(BaseModel):
        queries: List[str]

    resp = client.messages.create(
        model="claude-3-sonnet-20240229",
        max_tokens=400,
        temperature=0.2,
        system=SYSTEM_MESSAGE.format(max_queries=max_queries, auto_query_guidance=auto_query_guidance),
        messages=[
            {
                "role": "user",
                "content": user_input
            }
        ],
        response_model=Queries,
    )

    return resp.queries[:max_queries]
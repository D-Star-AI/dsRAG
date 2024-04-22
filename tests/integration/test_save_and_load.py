import sys
import os

# add ../../ to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from sprag.knowledge_base import KnowledgeBase
from sprag.llm import OpenAIChatAPI
from sprag.embedding import VoyageAIEmbedding


# initialize a KnowledgeBase object
auto_context_model = OpenAIChatAPI(model="gpt-4-turbo")
embedding_model = VoyageAIEmbedding(model="voyage-code-2")
kb = KnowledgeBase(kb_id="test_kb", auto_context_model=auto_context_model, embedding_model=embedding_model)

# load the KnowledgeBase object
kb1 = KnowledgeBase(kb_id="test_kb")

# verify that the KnowledgeBase object has the right parameters
assert kb1.auto_context_model.model == "gpt-4-turbo"
assert kb1.embedding_model.model == "voyage-code-2"

# delete the KnowledgeBase object
kb1.delete()
import sys
import os

# add ../../ to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from sprag.knowledge_base import KnowledgeBase

# initialize a KnowledgeBase object

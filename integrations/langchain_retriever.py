from typing import List, Dict, Union

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun

from dsrag.knowledge_base import KnowledgeBase

class DsRAGLangchainRetriever(BaseRetriever):
    """
    A retriever for the DsRAG project that uses the KnowledgeBase class to retrieve documents.
    """

    kb_id: str = ""
    rse_params: Union[Dict, str] = "balanced"

    def __init__(self, kb_id: str, rse_params: Union[Dict, str] = "balanced"):
        super().__init__()
        self.kb_id = kb_id
        self.rse_params = rse_params
    

    """
    Initializes a DsRAGLangchainRetriever instance.

    Args:
        kb_id: A KnowledgeBase id to use for retrieving documents.
        rse_params: The RSE parameters to use for querying
    """
        

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        
        kb = KnowledgeBase(kb_id=self.kb_id, exists_ok=True)
        
        segment_info = kb.query([query], rse_params=self.rse_params)
        # Create a Document object for each segment
        documents = []
        for segment in segment_info:
            document = Document(
                page_content=segment["text"],
                metadata={"doc_id": segment["doc_id"], "chunk_start": segment["chunk_start"], "chunk_end": segment["chunk_end"]},
            )
            documents.append(document)
        
        return documents

from typing import Any
from dsrag.database.chunk.db import ChunkDB
from dsrag.database.vector.types import TranscriptChunk


class PostgresChunkDB(ChunkDB):
    def __init__(self, db: Session, kb_id: str):
        self.kb_id = kb_id
        self.db = db

    def add_document(self, doc_id: str, chunks: dict[str, dict[str, Any]]) -> None:
        for chunk_index, chunk in chunks.items():
            document_title = chunk.get("document_title", "")
            document_summary = chunk.get("document_summary", "")
            section_title = chunk.get("section_title", "")
            section_summary = chunk.get("section_summary", "")
            chunk_text = chunk.get("chunk_text", "")
            document = TranscriptChunk(
                doc_id=doc_id,
                document_title=document_title,
                document_summary=document_summary,
                section_title=section_title,
                section_summary=section_summary,
                chunk_text=chunk_text,
                chunk_index=chunk_index,
            )
            self.db.add(document)
        self.db.commit()

    def remove_document(self, doc_id: str) -> None:
        self.db.query(TranscriptChunk).filter(TranscriptChunk.doc_id == doc_id).delete()
        self.db.commit()

    def get_document(self, doc_id: str, include_content: bool = False) -> dict:
        columns = [
            TranscriptChunk.doc_id,
            TranscriptChunk.document_title,
            TranscriptChunk.document_summary,
        ]
        if include_content:
            columns += [
                TranscriptChunk.section_title,
                TranscriptChunk.section_summary,
                TranscriptChunk.chunk_text,
                TranscriptChunk.chunk_index,
            ]

        results = (
            self.db.query(TranscriptChunk)
            .filter(TranscriptChunk.doc_id == doc_id)
            .all()
        )

        if not results:
            return {}

        formatted_results = {}
        if include_content:
            full_document_string = "\n".join([result.chunk_text for result in results])
            formatted_results["content"] = full_document_string

        formatted_results["doc_id"] = doc_id
        formatted_results["document_title"] = results[0].document_title

        return formatted_results

    def get_chunk_text(self, doc_id: str, chunk_index: int) -> str:
        chunk_text = (
            self.db.query(TranscriptChunk.chunk_text)
            .filter(
                TranscriptChunk.doc_id == doc_id,
                TranscriptChunk.chunk_index == chunk_index,
            )
            .one()
        )
        if chunk_text:
            return chunk_text[0]
        return ""

    def get_document_title(self, doc_id: str, chunk_index: int) -> str:
        result = (
            self.db.query(TranscriptChunk.document_title)
            .filter(
                TranscriptChunk.doc_id == doc_id,
                TranscriptChunk.chunk_index == chunk_index,
            )
            .one()
        )
        if result:
            return result[0]
        return ""

    def get_document_summary(self, doc_id: str, chunk_index: int) -> str:
        document_summary = (
            self.db.query(TranscriptChunk.document_summary)
            .filter(
                TranscriptChunk.doc_id == doc_id,
                TranscriptChunk.chunk_index == chunk_index,
            )
            .one()
        )
        if document_summary:
            return document_summary[0]
        return ""

    def get_section_title(self, doc_id: str, chunk_index: int) -> str:
        section_title = (
            self.db.query(TranscriptChunk.section_title)
            .filter(
                TranscriptChunk.doc_id == doc_id,
                TranscriptChunk.chunk_index == chunk_index,
            )
            .one()
        )
        if section_title:
            return section_title[0]
        return ""

    def get_section_summary(self, doc_id: str, chunk_index: int) -> str:
        section_summary = (
            self.db.query(TranscriptChunk.section_summary)
            .filter(
                TranscriptChunk.doc_id == doc_id,
                TranscriptChunk.chunk_index == chunk_index,
            )
            .one()
        )
        if section_summary:
            return section_summary[0]
        return ""

    def get_all_doc_ids(self, supp_id: str | None = None) -> list:  # noqa: ARG002
        results = self.db.query(TranscriptChunk.doc_id).distinct().all()
        return [result.doc_id for result in results]

    def delete(self) -> None: ...

    def to_dict(self) -> dict[str, str]:
        return {
            **super().to_dict(),
        }

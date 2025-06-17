from datetime import datetime
from typing import Optional
from typing_extensions import TypedDict


class FormattedDocument(TypedDict):
    id: Optional[str] = None
    title: Optional[str] = None
    content: Optional[str] = None
    summary: Optional[str] = None
    created_on: Optional[datetime] = None
    supp_id: Optional[str] = None
    metadata: Optional[dict] = {}
    chunk_count: Optional[int] = 0

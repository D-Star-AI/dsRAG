from datetime import datetime
from typing import Optional
from typing_extensions import TypedDict


class FormattedDocument(TypedDict):
    id: str
    title: str
    content: Optional[str]
    summary: Optional[str]
    created_on: datetime | None

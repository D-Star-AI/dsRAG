from datetime import datetime
from typing_extensions import TypedDict


class FormattedDocument(TypedDict):
    id: str
    title: str
    content: str | None
    summary: str | None
    created_on: datetime | None

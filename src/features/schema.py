"""Feature schema helpers."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DataSchema:
    target_column: str
    timestamp_column: str | None
    group_column: str | None
    id_columns: list[str]

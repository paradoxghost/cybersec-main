"""Project path helpers."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def project_path(*parts: str) -> Path:
    return ROOT.joinpath(*parts)

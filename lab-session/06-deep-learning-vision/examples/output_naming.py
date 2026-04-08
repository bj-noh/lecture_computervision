from __future__ import annotations

from pathlib import Path


def script_prefix(script_path: str | Path) -> str:
    parts: list[str] = []
    for part in Path(script_path).stem.split("_"):
        if part.isdigit():
            parts.append(part)
        else:
            break
    return "_".join(parts) or Path(script_path).stem


def with_script_prefix(script_path: str | Path, filename: str) -> str:
    prefix = script_prefix(script_path)
    return filename if filename.startswith(f"{prefix}_") else f"{prefix}_{filename}"

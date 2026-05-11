from __future__ import annotations

import json
from pathlib import Path


def load_link_ids(path: str | Path, selection_key: str = "selected_link_ids") -> list[str]:
    selection_path = Path(path)
    if selection_path.suffix.lower() == ".json":
        with selection_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, list):
            return [str(item).strip() for item in data if str(item).strip()]
        if isinstance(data, dict):
            if selection_key in data:
                return [str(item).strip() for item in data.get(selection_key, []) if str(item).strip()]
            if selection_key != "selected_link_ids":
                return []
            return [str(item).strip() for item in data.get("selected_link_ids", []) if str(item).strip()]
        raise ValueError("selection json must be a list or dict")

    with selection_path.open("r", encoding="utf-8") as handle:
        tokens: list[str] = []
        for line in handle:
            tokens.extend(line.replace(",", " ").split())
    return [token.strip() for token in tokens if token.strip()]

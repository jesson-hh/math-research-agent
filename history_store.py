"""File-based history storage for ideas, pools, notes, and networks."""

import json
import os
import uuid
from datetime import datetime
from pathlib import Path

HISTORY_DIR = Path("research_log/history")
VALID_TYPES = ("ideas", "pools", "notes", "networks")


def _ensure_dirs():
    for t in VALID_TYPES:
        (HISTORY_DIR / t).mkdir(parents=True, exist_ok=True)


_ensure_dirs()


def _path(item_type: str, item_id: str) -> Path:
    return HISTORY_DIR / item_type / f"{item_id}.json"


def list_items(item_type: str) -> list:
    """List all items of a type (id, name, created only — no data)."""
    if item_type not in VALID_TYPES:
        return []
    folder = HISTORY_DIR / item_type
    if not folder.exists():
        return []
    items = []
    for f in folder.glob("*.json"):
        try:
            with open(f, "r", encoding="utf-8") as fh:
                meta = json.load(fh)
                items.append({
                    "id": meta["id"],
                    "name": meta["name"],
                    "created": meta["created"],
                })
        except (json.JSONDecodeError, KeyError):
            continue
    items.sort(key=lambda x: x["created"], reverse=True)
    return items


def get_item(item_type: str, item_id: str) -> dict | None:
    """Get a single item with full data."""
    p = _path(item_type, item_id)
    if not p.exists():
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, KeyError):
        return None


def save_item(item_type: str, name: str, data: dict) -> dict:
    """Save a new item. Returns the saved metadata."""
    if item_type not in VALID_TYPES:
        raise ValueError(f"Invalid type: {item_type}")
    item_id = uuid.uuid4().hex[:12]
    now = datetime.now().isoformat(timespec="seconds")
    item = {
        "id": item_id,
        "name": name,
        "type": item_type,
        "created": now,
        "data": data,
    }
    _ensure_dirs()
    with open(_path(item_type, item_id), "w", encoding="utf-8") as f:
        json.dump(item, f, ensure_ascii=False, indent=2)
    return {"id": item_id, "name": name, "created": now}


def update_item(item_type: str, item_id: str, name: str) -> bool:
    """Rename an item. Returns True if successful."""
    p = _path(item_type, item_id)
    if not p.exists():
        return False
    try:
        with open(p, "r", encoding="utf-8") as f:
            item = json.load(f)
        item["name"] = name
        with open(p, "w", encoding="utf-8") as f:
            json.dump(item, f, ensure_ascii=False, indent=2)
        return True
    except (json.JSONDecodeError, KeyError):
        return False


def delete_item(item_type: str, item_id: str) -> bool:
    """Delete an item. Returns True if successful."""
    p = _path(item_type, item_id)
    if p.exists():
        p.unlink()
        return True
    return False

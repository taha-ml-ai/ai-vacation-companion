from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

def load_json(name: str) -> List[Dict[str, Any]]:
    path = DATA_DIR / name
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

import hashlib
import re
from typing import Optional


def robot_number_from_label(label: str) -> Optional[int]:
    compact = (label or "").strip()
    if not compact:
        return None
    numbers = re.findall(r"\d+", compact)
    if not numbers:
        return None
    try:
        return int(numbers[-1])
    except ValueError:
        return None


def stable_numeric_id(label: str) -> int:
    robot_id = robot_number_from_label(label)
    if robot_id is not None and (label or "").strip().lower().startswith("robot"):
        return robot_id
    return int(hashlib.sha1((label or "object").encode("utf-8")).hexdigest()[:8], 16) & 0x7FFFFFFF


def identity_global_id(label: Optional[str], category: Optional[str] = None, fallback: Optional[int] = None) -> Optional[int]:
    if not label:
        return fallback
    cat = (category or "").lower()
    lower = label.lower()
    if cat == "person" or lower.startswith("person"):
        return fallback
    if cat == "robot" or lower.startswith("robot"):
        robot_id = robot_number_from_label(label)
        return robot_id if robot_id is not None else stable_numeric_id(label)
    return stable_numeric_id(label)

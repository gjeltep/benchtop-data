"""Common parsing utilities for extracting structured data from LLM responses."""

import re
from typing import List, Optional


def extract_field(text: str, field_name: str, default: str = "", stop_patterns: Optional[List[str]] = None) -> str:
    """Extract a text field value from structured LLM output."""
    if stop_patterns is None:
        stop_patterns = ["\n-", "\n*"]

    patterns = [f"{field_name}:", f"{field_name} :", f"- {field_name}:"]

    for pattern in patterns:
        idx = text.lower().find(pattern.lower())
        if idx != -1:
            start = idx + len(pattern)
            end = len(text)
            for stop_pattern in stop_patterns:
                next_idx = text.find(stop_pattern, start)
                if next_idx != -1 and next_idx < end:
                    end = next_idx

            value = text[start:end].strip().lstrip("-").strip()
            return value if value else default

    return default


def extract_boolean(text: str, field_name: str, default: bool = False) -> bool:
    """Extract a boolean field value."""
    patterns = [f"{field_name}:", f"{field_name} :", f"- {field_name}:"]

    for pattern in patterns:
        idx = text.lower().find(pattern.lower())
        if idx != -1:
            snippet = text[idx:idx+50].lower()
            if "true" in snippet:
                return True
            elif "false" in snippet:
                return False

    return default


def extract_float(text: str, field_name: str, default: float = 0.0) -> float:
    """Extract a float field value."""
    patterns = [f"{field_name}:", f"{field_name} :", f"- {field_name}:"]

    for pattern in patterns:
        idx = text.lower().find(pattern.lower())
        if idx != -1:
            snippet = text[idx:idx+50]
            match = re.search(r'(\d+\.?\d*)', snippet)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    pass

    return default


def extract_list(text: str, field_name: str, stop_markers: Optional[List[str]] = None) -> List[str]:
    """Extract a list field value."""
    if stop_markers is None:
        stop_markers = ["is_complete", "is_accurate", "confidence", "should_refine", "reasoning"]

    items = []
    patterns = [f"{field_name}:", f"{field_name} :", f"- {field_name}:"]

    for pattern in patterns:
        idx = text.lower().find(pattern.lower())
        if idx != -1:
            start = idx + len(pattern)
            lines = text[start:].split("\n")

            for line in lines[:10]:  # Limit to first 10 lines
                line = line.strip()
                if not line:
                    continue

                # Stop at next field marker
                if any(marker in line.lower() for marker in stop_markers):
                    break

                # Extract list items
                if line.startswith(("-", "*")) or (line and line[0].isdigit()):
                    item = line.lstrip("-*").strip()
                    if "." in item[:3]:
                        item = item.split(".", 1)[-1].strip()
                    if item:
                        items.append(item)
                elif line and not any(marker in line.lower() for marker in [":", "="]):
                    items.append(line)

            break

    return items


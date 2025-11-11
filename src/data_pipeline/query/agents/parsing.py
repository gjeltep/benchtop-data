"""Common parsing utilities for extracting structured data from LLM responses.

This module uses functional programming patterns for parsing LLM output,
making the code more declarative and easier to reason about.
"""

import re
from typing import List, Optional, Callable, Tuple
from .config import AgenticConfig
from ...logging import get_logger

logger = get_logger(__name__)


class ParsingError(Exception):
    """Raised when LLM output cannot be parsed."""
    pass


def _find_pattern(text: str, patterns: List[str], case_sensitive: bool = False) -> Optional[Tuple[int, str]]:
    """
    Find the first matching pattern in text.

    Returns:
        Tuple of (index, matched_pattern) or None
    """
    search_text = text if case_sensitive else text.lower()
    for pattern in patterns:
        idx = search_text.find(pattern.lower() if not case_sensitive else pattern)
        if idx != -1:
            return (idx, pattern)
    return None


def _extract_until_stop(text: str, start: int, stop_patterns: List[str]) -> str:
    """Extract text from start position until a stop pattern is found."""
    end = len(text)
    for stop_pattern in stop_patterns:
        next_idx = text.find(stop_pattern, start)
        if next_idx != -1 and next_idx < end:
            end = next_idx
    return text[start:end].strip().lstrip("-").strip()


def _group_into_blocks(lines: List[str], is_block_start: Callable[[str], bool]) -> List[List[str]]:
    """Group lines into blocks based on a predicate."""
    blocks = []
    current_block = []

    for line in lines:
        if is_block_start(line):
            if current_block:
                blocks.append(current_block)
            current_block = [line]
        else:
            current_block.append(line)

    if current_block:
        blocks.append(current_block)

    return blocks


def _is_question_block(line: str) -> bool:
    """Check if a line starts a question block."""
    line_lower = line.lower().strip()
    return any(marker in line_lower for marker in ["question:", "sub-question:", "1.", "2.", "3.", "4.", "5."])


def extract_field(
    text: str,
    field_name: str,
    default: str = "",
    stop_patterns: Optional[List[str]] = None,
    config: Optional[AgenticConfig] = None
) -> str:
    """
    Extract a text field value from structured LLM output.

    Uses functional patterns: pattern matching -> extraction -> validation.

    Args:
        text: Text to parse
        field_name: Name of field to extract
        default: Default value if extraction fails
        stop_patterns: Patterns that indicate end of field
        config: Optional configuration (uses defaults if not provided)

    Returns:
        Extracted field value or default
    """
    stop_patterns = stop_patterns or ["\n-", "\n*"]
    config = config or AgenticConfig()
    patterns = [f"{field_name}:", f"{field_name} :", f"- {field_name}:"]

    try:
        result = _find_pattern(text, patterns)
        if result:
            idx, _ = result
            start = idx + len(result[1])
            value = _extract_until_stop(text, start, stop_patterns)
            return value if value else default
    except Exception as e:
        logger.warning(f"Error extracting field '{field_name}': {e}")

    return default


def extract_boolean(
    text: str,
    field_name: str,
    default: bool = False,
    config: Optional[AgenticConfig] = None
) -> bool:
    """
    Extract a boolean field value using functional pattern matching.

    Args:
        text: Text to parse
        field_name: Name of field to extract
        default: Default value if extraction fails
        config: Optional configuration (uses defaults if not provided)

    Returns:
        Extracted boolean value or default
    """
    config = config or AgenticConfig()
    patterns = [f"{field_name}:", f"{field_name} :", f"- {field_name}:"]

    try:
        result = _find_pattern(text, patterns)
        if result:
            idx, _ = result
            snippet = text[idx:idx+config.max_field_snippet_length].lower()
            if "true" in snippet:
                return True
            elif "false" in snippet:
                return False
    except Exception as e:
        logger.warning(f"Error extracting boolean field '{field_name}': {e}")

    return default


def extract_float(
    text: str,
    field_name: str,
    default: float = 0.0,
    config: Optional[AgenticConfig] = None
) -> float:
    """
    Extract a float field value using functional pattern matching.

    Args:
        text: Text to parse
        field_name: Name of field to extract
        default: Default value if extraction fails
        config: Optional configuration (uses defaults if not provided)

    Returns:
        Extracted float value or default
    """
    config = config or AgenticConfig()
    patterns = [f"{field_name}:", f"{field_name} :", f"- {field_name}:"]

    try:
        result = _find_pattern(text, patterns)
        if result:
            idx, _ = result
            snippet = text[idx:idx+config.max_field_snippet_length]
            match = re.search(r'(\d+\.?\d*)', snippet)
            if match:
                try:
                    value = float(match.group(1))
                    # Validate range (0.0-1.0 for confidence scores, etc.)
                    if 0.0 <= value <= 1.0 or field_name != "confidence_score":
                        return value
                except ValueError:
                    pass
    except Exception as e:
        logger.warning(f"Error extracting float field '{field_name}': {e}")

    return default


def _parse_list_item(line: str) -> Optional[str]:
    """Parse a single list item from a line."""
    line = line.strip()
    if not line:
        return None

    # Extract list items
    if line.startswith(("-", "*")) or (line and line[0].isdigit()):
        item = line.lstrip("-*").strip()
        if "." in item[:3]:
            item = item.split(".", 1)[-1].strip()
        return item if item else None
    elif line and not any(marker in line.lower() for marker in [":", "="]):
        return line

    return None


def extract_list(
    text: str,
    field_name: str,
    stop_markers: Optional[List[str]] = None,
    config: Optional[AgenticConfig] = None
) -> List[str]:
    """
    Extract a list field value using functional patterns.

    Uses map/filter pattern: find field -> extract lines -> parse items -> filter None.

    Args:
        text: Text to parse
        field_name: Name of field to extract
        stop_markers: Markers that indicate end of list
        config: Optional configuration (uses defaults if not provided)

    Returns:
        List of extracted items (empty list if extraction fails)
    """
    stop_markers = stop_markers or ["is_complete", "is_accurate", "confidence", "should_refine", "reasoning"]
    config = config or AgenticConfig()
    patterns = [f"{field_name}:", f"{field_name} :", f"- {field_name}:"]

    try:
        result = _find_pattern(text, patterns)
        if result:
            idx, _ = result
            start = idx + len(result[1])
            lines = text[start:].split("\n")

            # Functional approach: map -> filter -> map -> filter
            processed_lines = map(str.strip, lines[:config.max_list_extraction_lines])
            filtered_lines = filter(
                lambda l: l and not any(marker in l.lower() for marker in stop_markers),
                processed_lines
            )
            parsed_items = map(_parse_list_item, filtered_lines)
            items = list(filter(lambda x: x is not None, parsed_items))

            return items
    except Exception as e:
        logger.warning(f"Error extracting list field '{field_name}': {e}")

    return []


def extract_sub_questions(text: str) -> List[dict]:
    """
    Extract sub-questions using functional approach.

    Groups lines into blocks, filters question blocks, then parses each block.

    Args:
        text: Text containing sub-questions

    Returns:
        List of sub-question dictionaries
    """
    lines = text.split("\n")
    blocks = _group_into_blocks(lines, _is_question_block)

    def parse_block(block: List[str]) -> Optional[dict]:
        """Parse a question block into a sub-question dict."""
        if not block:
            return None

        question = None
        reasoning = ""
        requires_sql = False
        requires_semantic = False

        for line in block:
            line_lower = line.lower().strip()

            if any(marker in line_lower for marker in ["question:", "sub-question:"]):
                question = line.split(":", 1)[1].strip() if ":" in line else None
            elif any(marker in line_lower for marker in ["1.", "2.", "3.", "4.", "5."]):
                parts = line.split(".", 1)
                if len(parts) > 1:
                    question = parts[1].strip()
            elif "reasoning:" in line_lower:
                reasoning = line.split(":", 1)[1].strip() if ":" in line else ""
            elif "requires_sql:" in line_lower or "sql:" in line_lower:
                requires_sql = "true" in line_lower
            elif "requires_semantic:" in line_lower or "semantic:" in line_lower:
                requires_semantic = "true" in line_lower

        return {
            "question": question,
            "reasoning": reasoning or "Extracted from decomposition",
            "requires_sql": requires_sql,
            "requires_semantic": requires_semantic,
        } if question else None

    # Functional: map -> filter None
    sub_questions = list(filter(None, map(parse_block, blocks)))
    return sub_questions


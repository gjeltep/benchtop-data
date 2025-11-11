"""Configuration for agentic query processing patterns."""

from dataclasses import dataclass

## TODO: Toplevel UX with CLI/ UI cannot change these values, would be good to expose as parameters to the workflow

@dataclass
class AgenticConfig:
    """Configuration for agentic patterns in query processing."""

    # Reflection configuration
    reflection_threshold: float = 0.7
    max_refinement_iterations: int = 2

    # Response processing limits
    max_response_length: int = 2000
    refinement_context_length: int = 150

    # Chat history configuration
    chat_history_context_size: int = 6

    # Hallucination detection
    hallucination_detection_threshold: int = 3 # heuristic, best practice investigation needed

    # Parsing limits
    max_list_extraction_lines: int = 10
    max_field_snippet_length: int = 50

    # ReAct Agent configuration
    react_max_iterations: int = 10
    react_timeout: int = 180
    react_temperature: float = 0.1


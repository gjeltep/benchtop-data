"""Constants for workflow context keys."""

class ContextKeys:
    """
    Centralized context keys for workflow store.

    Eliminates magic strings and provides type safety.
    """
    QUERY = "query"
    ORIGINAL_QUERY = "original_query"
    REFINEMENT_ITERATION = "refinement_iteration"
    REFLECTION_FEEDBACK = "reflection_feedback"
    PREVIOUS_RESPONSE = "previous_response"
    ENGINES_USED = "engines_used"
    FINAL_RESPONSE_OBJ = "final_response_obj"
    REFINED_QUERY = "refined_query"
    DECOMPOSITION = "decomposition"
    SUB_QUESTIONS = "sub_questions"
    HAS_SUB_QUESTIONS = "has_sub_questions"
    REFLECTION = "reflection"
    SELECTED_ENGINE_INDEX = "selected_engine_index"
    SELECTED_ENGINE_INDICES = "selected_engine_indices"


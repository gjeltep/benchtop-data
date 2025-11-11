from typing import List
from llama_index.core.tools import QueryEngineTool
from llama_index.core.agent.workflow import ReActAgent
from .workflow import ReActAgentWorkflow
from ..workflow_protocol import QueryWorkflow
from ...llm.factory import LLMFactory
from ...agents.config import AgenticConfig
from ....logging import get_logger

logger = get_logger(__name__)


def create_react_agent_workflow(
    tools: List[QueryEngineTool],
    llm_config: dict,
    agentic_config: AgenticConfig,
) -> ReActAgentWorkflow:
    """
    Factory function to create ReActAgent workflow.

    Args:
        tools: Query engine tools for the agent
        llm_config: LLM configuration dictionary
        agentic_config: Agentic configuration

    Returns:
        ReActAgentWorkflow instance
    """
    react_llm = LLMFactory.create(
        reasoning_handler=None,  # This seems to break the ReActAgent if set to non-nil
        **llm_config,
        temperature=agentic_config.react_temperature,
    )

    agent = ReActAgent(
        tools=tools,
        llm=react_llm,
        verbose=True,
        max_iterations=agentic_config.react_max_iterations,
        timeout=agentic_config.react_timeout,
    )

    logger.info("ReActAgent workflow created")
    return ReActAgentWorkflow(agent)


from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    intent: str
    retrieved_docs: list
    tools_result: list
    confidence: float
    requires_human: bool
    final_response: str
    evaluation_score: float
    evaluation_feedback: str
    clarification_count: int
    awaiting_clarification: bool
    user_id: str
    guardrail_failed: bool
    guardrail_reason: str
    clarification_topic: str
    escalated: bool
    cache_hit: bool

from langgraph.graph import StateGraph, START, END
from agent.state import AgentState
from agent.nodes import (
    input_guardrails_node,
    semantic_cache_node,
    intent_classifier_node,
    greeting_node,
    clarification_node,
    faq_rag_node,
    tool_use_node,
    escalation_node,
    evaluator_node,
    hitl_node,
    output_guardrails_node,
)
from logger import get_logger
from langgraph.checkpoint.memory import MemorySaver

logger = get_logger(__name__)

MAX_CLARIFICATIONS = 2

checkpointer = MemorySaver()


def route_after_input_guardrails(state: AgentState) -> str:
    if state.get("guardrail_failed", False):
        logger.warning(
            "Short-circuiting graph  guardrail failed: %s",
            state.get("guardrail_reason"),
        )
        return "blocked"
    return "continue"


def route_by_intent(state: AgentState) -> str:
    intent = state.get("intent", "faq")
    if intent == "faq":
        return "faq_rag"
    elif intent == "tool":
        return "tool_use"
    elif intent == "greeting":
        return "greeting"
    else:
        return "escalation"


def route_clarification(state: AgentState) -> str:
    """Check if clarification is needed before routing to main nodes.
    Auto-escalates if clarification cap is reached.
    """
    if state.get("awaiting_clarification", False):
        if state.get("clarification_count", 0) >= MAX_CLARIFICATIONS:
            logger.warning("Clarification cap reached, escalating to human")
            return "escalation"
        return "clarification"
    return "route_intent"


def route_after_evaluation(state: AgentState) -> str:
    """After evaluation, decide whether to pass through or trigger HITL."""
    score = state.get("evaluation_score", 1.0)
    if score < 0.7:
        logger.warning("Evaluation score low: %.2f, triggering HITL", score)
        return "hitl"
    return "hitl"  # always passes through hitl — interrupt logic is inside


def build_graph():
    graph_builder = StateGraph(AgentState)

    # Register all nodes
    graph_builder.add_node("input_guardrails", input_guardrails_node)
    graph_builder.add_node("semantic_cache", semantic_cache_node)
    graph_builder.add_node("intent_classifier", intent_classifier_node)
    graph_builder.add_node("clarification", clarification_node)
    graph_builder.add_node("faq_rag", faq_rag_node)
    graph_builder.add_node("tool_use", tool_use_node)
    graph_builder.add_node("escalation", escalation_node)
    graph_builder.add_node("evaluator", evaluator_node)
    graph_builder.add_node("hitl", hitl_node)
    graph_builder.add_node("output_guardrails", output_guardrails_node)
    graph_builder.add_node("greeting", greeting_node)

    # Linear flow into classifier
    graph_builder.set_entry_point("input_guardrails")

    graph_builder.add_conditional_edges(
        "input_guardrails",
        route_after_input_guardrails,
        {
            "blocked": END,
            "continue": "semantic_cache",
        },
    )
    graph_builder.add_edge("semantic_cache", "intent_classifier")

    # Clarification check after classification
    graph_builder.add_conditional_edges(
        "intent_classifier",
        route_clarification,
        {
            "clarification": "clarification",
            "route_intent": "intent_router",
            "escalation": "escalation",
        },
    )

    # Need a pass-through router node for intent routing

    graph_builder.add_node("intent_router", lambda state: {})
    graph_builder.add_conditional_edges(
        "intent_router",
        route_by_intent,
        {
            "faq_rag": "faq_rag",
            "tool_use": "tool_use",
            "escalation": "escalation",
            "greeting": "greeting",
        },
    )
    graph_builder.add_edge("greeting", "evaluator")
    # Clarification loops back to intent_classifier
    graph_builder.add_edge("clarification", "intent_classifier")

    # All three branches converge at evaluator
    graph_builder.add_edge("faq_rag", "evaluator")
    graph_builder.add_edge("tool_use", "evaluator")
    graph_builder.add_edge("escalation", "evaluator")

    # Evaluator to HITL
    graph_builder.add_conditional_edges(
        "evaluator",
        route_after_evaluation,
        {
            "hitl": "hitl",
        },
    )

    # HITL to output guardrails to end
    graph_builder.add_edge("hitl", "output_guardrails")
    graph_builder.add_edge("output_guardrails", END)

    graph = graph_builder.compile(checkpointer=checkpointer)
    logger.info("Graph compiled")

    png_data = graph.get_graph().draw_mermaid_png()
    with open("agent_graph.png", "wb") as f:
        f.write(png_data)
    logger.info("Graph image saved")

    return graph


agent_graph = build_graph()

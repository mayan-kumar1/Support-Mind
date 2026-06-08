import asyncio
from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import HumanMessage
from agent.graph import agent_graph
from logger import get_logger

logger = get_logger(__name__)


async def run(user_input: str, user_id: str = "default_user"):
    logger.info("Starting agent with input: %s", user_input)

    config = {"configurable": {"thread_id": user_id}}

    initial_state = {
        "messages": [HumanMessage(content=user_input)],
        "intent": "",
        "retrieved_docs": [],
        "tool_result": {},
        "confidence": 0.0,
        "requires_human": False,
        "final_response": "",
        "evaluation_score": 1.0,
        "evaluation_feedback": "",
        "clarification_count": 0,
        "awaiting_clarification": False,
        "clarification_topic": "",
        "user_id": user_id,
        "guardrail_failed": False,
        "guardrail_reason": "",
        "escalated": False,
        "cache_hit": False,
    }

    result = await agent_graph.ainvoke(initial_state, config=config)  # type: ignore

    logger.info(
        "Agent completed  intent: %s  response: %s",
        result.get("intent"),
        result.get("final_response"),
    )
    print(result.get("final_response"))
    return result


if __name__ == "__main__":
    asyncio.run(run("What is your return policy?", user_id="user_001"))

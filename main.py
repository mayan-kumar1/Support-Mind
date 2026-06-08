from langchain_core.messages import HumanMessage
from agent.graph import agent_graph
from logger import get_logger
from dotenv import load_dotenv

load_dotenv(override=True)
logger = get_logger(__name__)


def run(user_input: str, user_id="default_user"):
    logger.info("Starting agent with input: %s", user_input)

    config = {"configurable": {"thread_id": user_id}}

    # Create initial state
    initial_state = {
        "messages": [HumanMessage(content=user_input)],
        "intent": "",
        "retrieved_docs": [],
        "tools_result": [],
        "confidence": 0.0,
        "requires_human": False,
        "final_response": "",
    }

    result = agent_graph.invoke(initial_state, config=config)  # type: ignore
    logger.info(
        "Agent completed | intent: %s | response: %s",
        result.get("intent"),
        result.get("final_response"),
    )
    return result


if __name__ == "__main__":

    run("What is your return policy?", user_id="user_001")

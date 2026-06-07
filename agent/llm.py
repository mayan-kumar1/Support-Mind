import os
from langchain_groq import ChatGroq
from logger import get_logger

logger = get_logger(__name__)

_llm_instance = None
_judge_instance = None


def get_llm(
    model: str = "llama-3.3-70b-versatile", temperature: float = 0.0
) -> ChatGroq:
    global _llm_instance
    if _llm_instance is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not set in environment")
        logger.info("Initialising LLM  model: %s  temperature: %s", model, temperature)
        _llm_instance = ChatGroq(model=model, temperature=temperature, api_key=api_key)  # type: ignore
    return _llm_instance


def get_judge_llm(
    model: str = "openai/gpt-oss-120b", temperature: float = 0.0
) -> ChatGroq:
    global _judge_instance
    if _judge_instance is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not set in environment")
        logger.info(
            "Initialising judge LLM  model: %s  temperature: %s", model, temperature
        )
        _judge_instance = ChatGroq(
            model=model, temperature=temperature, api_key=api_key  # type: ignore
        )
    return _judge_instance

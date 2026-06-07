import os
import dspy
from dspy_modules.intent_classifier import IntentClassifierModule
from logger import get_logger

logger = get_logger(__name__)

COMPILED_PATH = "dspy_modules/compiled_classifier.json"
_classifier = None


def get_classifier() -> IntentClassifierModule:
    global _classifier
    if _classifier is None:
        api_key = os.getenv("GROQ_API_KEY")
        lm = dspy.LM(
            model="groq/llama-3.3-70b-versatile",
            api_key=api_key,
            temperature=0.0,
        )
        dspy.configure(lm=lm)

        _classifier = IntentClassifierModule()

        if os.path.exists(COMPILED_PATH):
            _classifier.load(COMPILED_PATH)
            logger.info("Loaded compiled DSPy classifier from %s", COMPILED_PATH)
        else:
            logger.warning(
                "No compiled classifier found  using baseline  path: %s", COMPILED_PATH
            )

    return _classifier

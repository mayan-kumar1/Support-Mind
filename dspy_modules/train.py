import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv

load_dotenv()

import dspy
from dspy.teleprompt import BootstrapFewShot
from dspy_modules.intent_classifier import (
    IntentClassifierModule,
    TRAINING_EXAMPLES,
    intent_metric,
)
from logger import get_logger

logger = get_logger(__name__)

COMPILED_PATH = "dspy_modules/compiled_classifier.json"

# Use subset to reduce API calls during compilation
TRAIN_SET = TRAINING_EXAMPLES[:15]
EVAL_SET = TRAINING_EXAMPLES[:10]


def safe_predict(classifier, example, retries=3):
    """Predict with retry logic on rate limit errors."""
    for attempt in range(retries):
        try:
            return classifier(message=example.message, history=example.history)
        except Exception as e:
            if "rate_limit" in str(e).lower():
                wait = (attempt + 1) * 10
                logger.warning(
                    "Rate limit hit  waiting %ds  attempt: %d of %d",
                    wait,
                    attempt + 1,
                    retries,
                )
                time.sleep(wait)
            else:
                logger.error("Prediction error  error: %s", str(e))
                return None
    logger.error("All retries exhausted for example: %s", example.message)
    return None


def evaluate(classifier, examples):
    """Evaluate classifier accuracy with rate limit protection."""
    correct = 0
    total = 0

    for i, example in enumerate(examples):
        logger.info(
            "Evaluating example %d of %d  message: %s",
            i + 1,
            len(examples),
            example.message,
        )

        pred = safe_predict(classifier, example)

        if pred is not None:
            total += 1
            if pred.intent == example.intent:
                correct += 1
            else:
                logger.debug(
                    "Mismatch  expected: %s  got: %s  message: %s",
                    example.intent,
                    pred.intent,
                    example.message,
                )

        # Respect rate limit between calls
        time.sleep(3)

    accuracy = correct / total if total > 0 else 0.0
    logger.info(
        "Evaluation complete  correct: %d  total: %d  accuracy: %.2f%%",
        correct,
        total,
        accuracy * 100,
    )
    return accuracy


def train():
    # Configure DSPy with Groq small model
    api_key = os.getenv("GROQ_API_KEY")
    lm = dspy.LM(
        model="groq/llama-3.1-8b-instant",
        api_key=api_key,
        temperature=0.0,
    )
    dspy.configure(lm=lm)

    logger.info("Starting DSPy training")
    logger.info("Train set size: %d  Eval set size: %d", len(TRAIN_SET), len(EVAL_SET))

    # Baseline — before optimisation
    logger.info("Evaluating baseline accuracy...")
    baseline = IntentClassifierModule()
    baseline_accuracy = evaluate(baseline, EVAL_SET)
    logger.info("Baseline accuracy: %.2f%%", baseline_accuracy * 100)

    # Compile with BootstrapFewShot — reduced demos to respect rate limits
    teleprompter = BootstrapFewShot(
        metric=intent_metric,
        max_bootstrapped_demos=2,
        max_labeled_demos=4,
    )

    logger.info("Compiling with BootstrapFewShot  this will take a few minutes...")
    compiled = teleprompter.compile(
        IntentClassifierModule(),
        trainset=TRAIN_SET,
    )
    logger.info("Compilation complete")

    # Wait before evaluation to avoid rate limits
    logger.info("Waiting 15s before evaluation to respect rate limits...")
    time.sleep(15)

    # Evaluate compiled classifier
    logger.info("Evaluating compiled accuracy...")
    compiled_accuracy = evaluate(compiled, EVAL_SET)
    logger.info("Compiled accuracy: %.2f%%", compiled_accuracy * 100)
    logger.info("Improvement: %+.2f%%", (compiled_accuracy - baseline_accuracy) * 100)

    # Save compiled classifier
    compiled.save(COMPILED_PATH)
    logger.info("Compiled classifier saved to %s", COMPILED_PATH)

    return baseline_accuracy, compiled_accuracy


if __name__ == "__main__":
    baseline, compiled = train()
    print(f"\n{'='*40}")
    print(f"DSPy Training Results")
    print(f"{'='*40}")
    print(f"Model:              llama-3.1-8b-instant")
    print(f"Train examples:     {len(TRAIN_SET)}")
    print(f"Eval examples:      {len(EVAL_SET)}")
    print(f"Baseline accuracy:  {baseline * 100:.1f}%")
    print(f"Compiled accuracy:  {compiled * 100:.1f}%")
    print(f"Improvement:        {(compiled - baseline) * 100:+.1f}%")
    print(f"{'='*40}")

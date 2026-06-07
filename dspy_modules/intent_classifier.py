import dspy
from dspy import InputField, OutputField, Signature
from typing import Literal
from logger import get_logger

logger = get_logger(__name__)


class IntentSignature(Signature):
    """Classify the intent of a customer support message for an e-commerce store.

    Intents:
    - faq: General questions about policies, products, shipping, payments, account
    - tool: Requests needing order lookup — status, tracking, returns, refunds
    - escalation: Angry users, explicit human requests, fraud, legal threats
    - greeting: Greetings, small talk, capability questions
    - clarification: Message is too vague to classify confidently
    """

    message: str = InputField(desc="The customer message to classify")
    history: str = InputField(desc="Prior conversation turns for context")
    intent: Literal["faq", "tool", "escalation", "greeting", "clarification"] = (
        OutputField(desc="The classified intent")
    )
    confidence: float = OutputField(desc="Confidence score between 0.0 and 1.0")
    awaiting_clarification: bool = OutputField(
        desc="True if more information is needed to act on this message"
    )
    clarification_topic: str = OutputField(
        desc="What specific information is missing. Empty string if not needed."
    )


# ── Training examples ──────────────────────────────────────────────────────────
TRAINING_EXAMPLES = [
    # FAQ
    dspy.Example(
        message="What is your return policy?",
        history="",
        intent="faq",
        confidence=1.0,
        awaiting_clarification=False,
        clarification_topic="",
    ).with_inputs("message", "history"),
    dspy.Example(
        message="How long does shipping take?",
        history="",
        intent="faq",
        confidence=1.0,
        awaiting_clarification=False,
        clarification_topic="",
    ).with_inputs("message", "history"),
    dspy.Example(
        message="Do you offer free shipping?",
        history="",
        intent="faq",
        confidence=1.0,
        awaiting_clarification=False,
        clarification_topic="",
    ).with_inputs("message", "history"),
    dspy.Example(
        message="What payment methods do you accept?",
        history="",
        intent="faq",
        confidence=1.0,
        awaiting_clarification=False,
        clarification_topic="",
    ).with_inputs("message", "history"),
    dspy.Example(
        message="Is my card information safe?",
        history="",
        intent="faq",
        confidence=1.0,
        awaiting_clarification=False,
        clarification_topic="",
    ).with_inputs("message", "history"),
    dspy.Example(
        message="Do products come with a warranty?",
        history="",
        intent="faq",
        confidence=1.0,
        awaiting_clarification=False,
        clarification_topic="",
    ).with_inputs("message", "history"),
    dspy.Example(
        message="How do I reset my password?",
        history="",
        intent="faq",
        confidence=1.0,
        awaiting_clarification=False,
        clarification_topic="",
    ).with_inputs("message", "history"),
    # Tool
    dspy.Example(
        message="Where is my order ORD001?",
        history="",
        intent="tool",
        confidence=1.0,
        awaiting_clarification=False,
        clarification_topic="",
    ).with_inputs("message", "history"),
    dspy.Example(
        message="I want to return order ORD003",
        history="",
        intent="tool",
        confidence=1.0,
        awaiting_clarification=False,
        clarification_topic="",
    ).with_inputs("message", "history"),
    dspy.Example(
        message="Can I get a refund for ORD002?",
        history="",
        intent="tool",
        confidence=1.0,
        awaiting_clarification=False,
        clarification_topic="",
    ).with_inputs("message", "history"),
    dspy.Example(
        message="What is the tracking number for my order ORD004?",
        history="",
        intent="tool",
        confidence=1.0,
        awaiting_clarification=False,
        clarification_topic="",
    ).with_inputs("message", "history"),
    dspy.Example(
        message="I want to cancel my order ORD001",
        history="",
        intent="tool",
        confidence=1.0,
        awaiting_clarification=False,
        clarification_topic="",
    ).with_inputs("message", "history"),
    # Tool with clarification needed
    dspy.Example(
        message="Where is my Samsung TV?",
        history="",
        intent="tool",
        confidence=0.9,
        awaiting_clarification=True,
        clarification_topic="order ID",
    ).with_inputs("message", "history"),
    dspy.Example(
        message="I want to return my laptop",
        history="",
        intent="tool",
        confidence=0.9,
        awaiting_clarification=True,
        clarification_topic="order ID",
    ).with_inputs("message", "history"),
    dspy.Example(
        message="Check my order status",
        history="",
        intent="tool",
        confidence=0.9,
        awaiting_clarification=True,
        clarification_topic="order ID",
    ).with_inputs("message", "history"),
    # Escalation
    dspy.Example(
        message="I want to speak to a human right now",
        history="",
        intent="escalation",
        confidence=1.0,
        awaiting_clarification=False,
        clarification_topic="",
    ).with_inputs("message", "history"),
    dspy.Example(
        message="This is absolutely ridiculous I have been waiting 3 weeks",
        history="",
        intent="escalation",
        confidence=1.0,
        awaiting_clarification=False,
        clarification_topic="",
    ).with_inputs("message", "history"),
    dspy.Example(
        message="I think someone hacked my account",
        history="",
        intent="escalation",
        confidence=1.0,
        awaiting_clarification=False,
        clarification_topic="",
    ).with_inputs("message", "history"),
    dspy.Example(
        message="I will take legal action if this is not resolved",
        history="",
        intent="escalation",
        confidence=1.0,
        awaiting_clarification=False,
        clarification_topic="",
    ).with_inputs("message", "history"),
    dspy.Example(
        message="Connect me to your manager",
        history="",
        intent="escalation",
        confidence=1.0,
        awaiting_clarification=False,
        clarification_topic="",
    ).with_inputs("message", "history"),
    # Greeting
    dspy.Example(
        message="Hi there",
        history="",
        intent="greeting",
        confidence=1.0,
        awaiting_clarification=False,
        clarification_topic="",
    ).with_inputs("message", "history"),
    dspy.Example(
        message="Hola",
        history="",
        intent="greeting",
        confidence=1.0,
        awaiting_clarification=False,
        clarification_topic="",
    ).with_inputs("message", "history"),
    dspy.Example(
        message="Good morning",
        history="",
        intent="greeting",
        confidence=1.0,
        awaiting_clarification=False,
        clarification_topic="",
    ).with_inputs("message", "history"),
    dspy.Example(
        message="What can you help me with?",
        history="",
        intent="greeting",
        confidence=1.0,
        awaiting_clarification=False,
        clarification_topic="",
    ).with_inputs("message", "history"),
    dspy.Example(
        message="What are your capabilities?",
        history="",
        intent="greeting",
        confidence=1.0,
        awaiting_clarification=False,
        clarification_topic="",
    ).with_inputs("message", "history"),
    # Clarification / vague
    dspy.Example(
        message="I have a problem",
        history="",
        intent="faq",
        confidence=0.5,
        awaiting_clarification=True,
        clarification_topic="what problem they are experiencing",
    ).with_inputs("message", "history"),
    dspy.Example(
        message="It still hasn't arrived",
        history="",
        intent="tool",
        confidence=0.8,
        awaiting_clarification=True,
        clarification_topic="order ID and what item they are referring to",
    ).with_inputs("message", "history"),
    dspy.Example(
        message="I want to cancel",
        history="",
        intent="tool",
        confidence=0.8,
        awaiting_clarification=True,
        clarification_topic="order ID and what they want to cancel",
    ).with_inputs("message", "history"),
    dspy.Example(
        message="Help me",
        history="",
        intent="faq",
        confidence=0.4,
        awaiting_clarification=True,
        clarification_topic="what they need help with",
    ).with_inputs("message", "history"),
    dspy.Example(
        message="Something went wrong with my purchase",
        history="",
        intent="tool",
        confidence=0.7,
        awaiting_clarification=True,
        clarification_topic="order ID and what went wrong",
    ).with_inputs("message", "history"),
]


def intent_metric(example: dspy.Example, prediction, trace=None) -> bool:
    """Score a prediction — intent must match exactly."""
    return prediction.intent == example.intent


class IntentClassifierModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.classify = dspy.Predict(IntentSignature)

    def forward(self, message: str, history: str = "") -> dspy.Prediction:
        return self.classify(message=message, history=history)

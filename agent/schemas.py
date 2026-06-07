from pydantic import BaseModel, Field
from typing import Literal


class IntentClassification(BaseModel):
    intent: Literal["faq", "tool", "escalation", "greeting"] = Field(
        description="""The classified intent of the user message.

        - faq: General questions about policies, products, shipping times, payment methods,
        account management, warranties, or any question that can be answered from a knowledge base.

        - tool: Requests that require looking up or acting on a specific order. This includes
        order status, tracking, returns, refunds, cancellations, or exchanges.

        - escalation: The user is angry, frustrated, or distressed. The user explicitly asks
        for a human agent. The message contains legal threats, fraud claims, or account
        security concerns.

        - greeting: The user is greeting the agent, making small talk, or asking meta questions
        about the agent itself such as 'what can you do', 'how can you help me', 'what are
        your capabilities', 'who are you', 'what is this'. These require no knowledge base
        lookup. Set awaiting_clarification to False always for this intent."""
    )
    confidence: float = Field(
        description="Confidence score between 0.0 and 1.0.", ge=0.0, le=1.0
    )
    awaiting_clarification: bool = Field(
        description="""True if the message does not have enough information to act on.
        Examples:
        - Intent is tool but no order ID or specific order reference is present
        - User mentions a problem but is too vague to determine what action to take
        - User wants to cancel or return but has not specified what
        - Any case where a follow-up question is genuinely needed to help the user
        Set False if the intent is clear and actionable as-is."""
    )
    clarification_topic: str = Field(
        description="What specific information is missing. Empty string if awaiting_clarification is False.",
        default="",
    )
    reasoning: str = Field(
        description="One sentence explaining why this intent was chosen."
    )


class OffTopicCheck(BaseModel):
    is_off_topic: bool = Field(
        description="True if the message is completely unrelated to e-commerce customer support."
    )
    reasoning: str = Field(description="One sentence explaining the decision.")


class EvaluationResult(BaseModel):
    score: float = Field(
        description="Quality score between 0.0 and 1.0.", ge=0.0, le=1.0
    )
    feedback: str = Field(description="One sentence explaining the score.")
    passes: bool = Field(
        description="True if the response is good enough to send to the user."
    )

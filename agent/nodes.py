from agent.guardrails.input_checks import run_input_checks
from agent.state import AgentState
from logger import get_logger
from rag.pipeline import retrieve
import re
from tools.mock_apis import get_order_status, initiate_return
from langgraph.types import interrupt
from agent.llm import get_llm, get_judge_llm
from agent.schemas import EvaluationResult
from agent.guardrails.output_checks import run_output_checks
from cache.semantic_cache import semantic_cache
from dspy_modules.load import get_classifier

logger = get_logger(__name__)


FAQ_SYSTEM_PROMPT = """You are a helpful and friendly e-commerce customer support agent.

Your job is to answer the user's question using ONLY the context provided below.
The context contains relevant FAQ entries retrieved from our knowledge base.

Rules:
- Answer in a natural, conversational tone
- Be concise — no unnecessary filler
- Do not make up information not present in the context
- If the context does not contain enough information to answer, say so honestly
- Never reveal that you are using a knowledge base or retrieved documents
- Never say "based on the context" or "according to the FAQ"
- Speak as if you know this information directly
"""

TOOL_RESPONSE_SYSTEM_PROMPT = """You are a helpful e-commerce customer support agent.

You have just retrieved real order information from our system.
Your job is to present this information to the customer in a natural, friendly, and concise way.

Rules:
- Be conversational, not robotic
- Do not expose raw data fields or technical terms
- If the action was successful, confirm it clearly
- If there was an error, explain it simply and suggest next steps
- Never say "based on the data" or "according to our system"
- Speak as if you know this information directly
"""

ORDER_ID_PATTERN = re.compile(r"\b(ORD[-]?\d+|TRK\d+)\b", re.IGNORECASE)

CLARIFICATION_SYSTEM_PROMPT = """You are a helpful and friendly e-commerce customer support agent.

The user wants help but has not provided enough information to act on their request.

Your job is to ask for the missing information in a natural, friendly, and concise way.
- Keep it to one sentence
- Be specific about what you need
- Do not repeat what the user said back to them
- Do not apologise excessively
- Sound human, not robotic
- Never mention order IDs, databases, or systems directly

For awaiting_clarification:
- Scan the ENTIRE conversation history carefully, not just the latest message.
- Order IDs can appear in any format: ORD001, ORD-001, tracking numbers like TRK998877.
- If ANY order reference appears anywhere in the history, set awaiting_clarification to False.
- Only set True if the history has been checked and no reference exists.
"""

GREETING_SYSTEM_PROMPT = """You are a friendly e-commerce customer support agent called SupportMind.

The customer is either greeting you or asking about what you can help with.

You can help with:
- Order status and tracking
- Returns and refunds
- Product questions and policies
- Payment and account queries
- Shipping information
- Complaints and escalations to human agents

Rules:
- If it is a greeting, respond warmly and invite them to share their issue
- If they ask what you can do, briefly list your capabilities in a friendly natural way
- Keep it to 2 to 3 sentences maximum
- Sound human, not like a feature list
- Match the energy of the message
"""

EVALUATOR_SYSTEM_PROMPT = """You are an independent quality evaluator for an e-commerce customer support agent.

Your job is to evaluate whether the agent's response is good enough to send to the customer.

Scoring criteria:
- 0.9 to 1.0: Perfect. Response is accurate, helpful, complete, and natural
- 0.7 to 0.9: Good. Response is mostly correct with minor issues
- 0.5 to 0.7: Acceptable but weak. Response is vague, incomplete, or slightly off
- 0.3 to 0.5: Poor. Response misses the point or contains questionable information
- 0.0 to 0.3: Unacceptable. Response is wrong, harmful, or completely unhelpful

Key things to check:
- Does the response actually answer what the customer asked?
- Is the response grounded in the conversation context?
- Is there any sign of hallucination or made-up information?
- Is the tone appropriate — friendly, professional, not robotic?
- For tool responses — does the response accurately reflect the order data?
- For FAQ responses — is the answer consistent with standard e-commerce policies?
- For escalations — is the handoff empathetic and reassuring?
- For greetings — is the response warm and natural?

Be strict but fair. A response that is technically correct but robotic or unhelpful
should score no higher than 0.6.
"""

ESCALATION_SYSTEM_PROMPT = """You are a helpful and empathetic e-commerce customer support agent.

The customer needs to be connected to a human agent. Your job is to:
1. Acknowledge their concern warmly and empathetically
2. Let them know a human agent will be with them shortly
3. Briefly summarise their issue so they don't have to repeat themselves to the human agent

Rules:
- Be warm, calm, and reassuring
- Never be defensive or dismissive
- Keep it concise — 2 to 3 sentences maximum
- Do not promise specific wait times
- Do not try to resolve the issue yourself
"""

HITL_ESCALATION_RESPONSE = "I'm connecting you with one of our human agents who will be able to assist you further. Please hold on."


def extract_order_id_from_history(messages: list) -> str | None:
    for msg in reversed(messages):
        match = ORDER_ID_PATTERN.search(msg.content)
        if match:
            return match.group(0).upper()
    return None


def determine_tool_action(messages: list) -> str:
    full_history = " ".join([msg.content.lower() for msg in messages])
    return_keywords = ["return", "refund", "send back", "give back", "exchange"]
    for keyword in return_keywords:
        if keyword in full_history:
            return "return"
    return "status"


# ── Nodes ──────────────────────────────────────────────────────────────────────


async def input_guardrails_node(state: AgentState) -> dict:
    logger.info("Running input guardrails node")
    last_message = state["messages"][-1].content
    failed, reason = run_input_checks(last_message)
    if failed:
        logger.warning("Input guardrail failed  reason: %s", reason)
        return {
            "guardrail_failed": True,
            "guardrail_reason": reason,
            "final_response": reason,
        }
    logger.info("Input guardrails passed")
    return {
        "guardrail_failed": False,
        "guardrail_reason": "",
    }


async def semantic_cache_node(state: AgentState) -> dict:
    logger.info("Running semantic_cache_node")
    last_message = state["messages"][-1].content
    cached_response = semantic_cache.get(last_message)
    if cached_response:
        logger.info("Serving response from semantic cache")
        return {
            "final_response": cached_response,
            "cache_hit": True,
            "confidence": 1.0,
            "intent": "faq",
        }
    return {"cache_hit": False}


async def intent_classifier_node(state: AgentState) -> dict:
    logger.info("Running intent classifier node")
    classifier = get_classifier()
    history = "\n".join(
        [f"{msg.__class__.__name__}: {msg.content}" for msg in state["messages"]]
    )
    last_message = state["messages"][-1].content
    try:
        result = classifier(message=last_message, history=history)
        logger.info(
            "Intent classified  intent: %s  confidence: %.2f  awaiting_clarification: %s  topic: %s",
            result.intent,
            result.confidence,
            result.awaiting_clarification,
            result.clarification_topic,
        )
        return {
            "intent": result.intent,
            "confidence": result.confidence,
            "awaiting_clarification": result.awaiting_clarification,
            "clarification_topic": result.clarification_topic,
        }
    except Exception as e:
        logger.error("Intent classification failed  error: %s", str(e))
        return {
            "intent": "escalation",
            "confidence": 0.0,
            "awaiting_clarification": False,
            "clarification_topic": "",
        }


async def clarification_node(state: AgentState) -> dict:
    count = state.get("clarification_count", 0) + 1
    logger.info("Running clarification node  round: %d", count)

    llm = get_llm()
    last_message = state["messages"][-1].content
    topic = state.get("clarification_topic", "the information needed")

    prompt = f"""The user said: "{last_message}"

What is missing: {topic}

Ask the user for this specific missing information naturally and concisely in one sentence."""

    try:
        response = await llm.ainvoke(
            [
                {"role": "system", "content": CLARIFICATION_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
        )
        clarification_question = response.content
    except Exception as e:
        logger.error("Clarification LLM failed  error: %s", str(e))
        clarification_question = (
            "Could you please provide more details so I can help you?"
        )

    logger.info(
        "Clarification question generated  round: %d  question: %s",
        count,
        clarification_question,
    )

    interrupt(clarification_question)

    return {
        "clarification_count": count,
        "awaiting_clarification": False,
        "final_response": clarification_question,
    }


async def faq_rag_node(state: AgentState) -> dict:
    logger.info("Running faq_rag node")

    llm = get_llm()
    last_message = state["messages"][-1].content
    docs = retrieve(last_message, top_k=3)

    if not docs:
        logger.warning("No docs retrieved for query: %s", last_message)
        return {
            "retrieved_docs": [],
            "confidence": 0.0,
            "final_response": "I am sorry, I could not find information related to your question. Let me connect you with a human agent.",
            "requires_human": True,
        }

    context = "\n\n".join([f"Q: {doc['question']}\nA: {doc['answer']}" for doc in docs])
    top_score = docs[0]["score"]
    logger.info("Top retrieval score: %.4f", top_score)

    prompt = f"""Context:
{context}

Customer question: {last_message}

Answer the customer's question using only the context above."""

    try:
        response = await llm.ainvoke(
            [
                {"role": "system", "content": FAQ_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
        )
        final_response = response.content
        logger.info(
            "FAQ response generated  top_score: %.4f  response_length: %d",
            top_score,
            len(final_response),
        )
        if top_score >= 0.6:
            semantic_cache.set(last_message, final_response)  # type: ignore
            logger.info("Response stored in semantic cache")

        confidence = top_score if top_score >= 0.5 else top_score * 0.5
        return {
            "retrieved_docs": docs,
            "confidence": confidence,
            "final_response": final_response,
            "requires_human": confidence < 0.35,
        }
    except Exception as e:
        logger.error("FAQ RAG node failed  error: %s", str(e))
        return {
            "retrieved_docs": docs,
            "confidence": 0.0,
            "final_response": "I am having trouble answering that right now. Let me connect you with a human agent.",
            "requires_human": True,
        }


async def greeting_node(state: AgentState) -> dict:
    logger.info("Running greeting node")
    llm = get_llm()
    last_message = state["messages"][-1].content
    try:
        response = await llm.ainvoke(
            [
                {"role": "system", "content": GREETING_SYSTEM_PROMPT},
                {"role": "user", "content": last_message},
            ]
        )
        logger.info("Greeting response generated")
        return {
            "confidence": 1.0,
            "requires_human": False,
            "final_response": response.content,
        }
    except Exception as e:
        logger.error("Greeting node failed  error: %s", str(e))
        return {
            "confidence": 1.0,
            "requires_human": False,
            "final_response": "Hello! Welcome to SupportMind. How can I help you today?",
        }


async def tool_use_node(state: AgentState) -> dict:
    logger.info("Running tool_use node")
    llm = get_llm()
    messages = state["messages"]
    order_id = extract_order_id_from_history(messages)

    if not order_id:
        logger.warning("Tool use node reached without order ID")
        return {
            "tool_result": {"success": False, "error": "No order ID found"},
            "confidence": 0.2,
            "requires_human": False,
            "awaiting_clarification": True,
            "clarification_topic": "order ID",
            "final_response": "I need your order ID to look that up. You can find it in your confirmation email.",
        }

    action = determine_tool_action(messages)
    logger.info("Tool action determined  order_id: %s  action: %s", order_id, action)

    result = (
        initiate_return(order_id) if action == "return" else get_order_status(order_id)
    )
    logger.info("Tool result  success: %s", result.get("success"))

    prompt = f"""Order ID: {order_id}
Action taken: {"Return initiated" if action == "return" else "Order status checked"}
Result: {result}

Present this information to the customer naturally."""

    try:
        response = await llm.ainvoke(
            [
                {"role": "system", "content": TOOL_RESPONSE_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
        )
        confidence = 0.95 if result.get("success") else 0.4
        return {
            "tool_result": result,
            "confidence": confidence,
            "requires_human": not result.get("success"),
            "final_response": response.content,
        }
    except Exception as e:
        logger.error("Tool use node LLM formatting failed  error: %s", str(e))
        return {
            "tool_result": result,
            "confidence": 0.3,
            "requires_human": True,
            "final_response": "I found your order but had trouble formatting the response. Let me connect you with a human agent.",
        }


async def escalation_node(state: AgentState) -> dict:
    logger.info("Running escalation node")
    llm = get_llm()
    history = "\n".join(
        [f"{msg.__class__.__name__}: {msg.content}" for msg in state["messages"]]
    )
    prompt = f"""Conversation history:
{history}

The customer needs to speak with a human agent.
Acknowledge their concern and let them know a human will assist them shortly."""

    try:
        response = await llm.ainvoke(
            [
                {"role": "system", "content": ESCALATION_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
        )
        logger.info("Escalation response generated")
        return {
            "requires_human": True,
            "confidence": 1.0,
            "escalated": True,
            "final_response": response.content,
        }
    except Exception as e:
        logger.error("Escalation node failed  error: %s", str(e))
        return {
            "requires_human": True,
            "confidence": 1.0,
            "escalated": True,
            "final_response": "I understand your concern and I'm sorry for the inconvenience. A human agent will be with you shortly to assist you further.",
        }


async def evaluator_node(state: AgentState) -> dict:
    logger.info("Evaluating response quality")
    judge = get_judge_llm()
    evaluator = judge.with_structured_output(EvaluationResult)

    last_message = state["messages"][-1].content
    final_response = state.get("final_response", "")
    intent = state.get("intent", "unknown")

    if not final_response:
        logger.warning("No final response to evaluate")
        return {
            "evaluation_score": 0.0,
            "evaluation_feedback": "No response was generated.",
            "confidence": 0.0,
        }

    prompt = f"""Intent classified: {intent}

Customer message: {last_message}

Agent response: {final_response}

Evaluate the quality of the agent response."""

    try:
        result: EvaluationResult = await evaluator.ainvoke(
            [
                {"role": "system", "content": EVALUATOR_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
        )  # type: ignore
        logger.info(
            "Evaluation complete  score: %.2f  passes: %s  feedback: %s",
            result.score,
            result.passes,
            result.feedback,
        )
        current_confidence = state.get("confidence", 1.0)
        final_confidence = min(current_confidence, result.score)
        return {
            "evaluation_score": result.score,
            "evaluation_feedback": result.feedback,
            "confidence": final_confidence,
        }
    except Exception as e:
        logger.error("Evaluator node failed  error: %s", str(e))
        return {
            "evaluation_score": 0.8,
            "evaluation_feedback": "Evaluation failed — defaulting to pass.",
            "confidence": state.get("confidence", 0.8),
        }


async def hitl_node(state: AgentState) -> dict:
    logger.info("Running HITL check node")

    confidence = state.get("confidence", 1.0)
    requires_human = state.get("requires_human", False)
    evaluation_score = state.get("evaluation_score", 1.0)
    intent = state.get("intent", "")

    should_interrupt = (
        confidence < 0.7
        or requires_human
        or evaluation_score < 0.7
        or intent == "escalation"
    )

    if not should_interrupt:
        logger.info(
            "HITL check passed  confidence: %.2f  evaluation_score: %.2f",
            confidence,
            evaluation_score,
        )
        return {}

    review_package = {
        "customer_message": state["messages"][-1].content,
        "proposed_response": state.get("final_response", ""),
        "intent": intent,
        "confidence": confidence,
        "evaluation_score": evaluation_score,
        "evaluation_feedback": state.get("evaluation_feedback", ""),
        "requires_human": requires_human,
    }

    logger.warning(
        "HITL interrupt triggered  confidence: %.2f  evaluation_score: %.2f  requires_human: %s  intent: %s",
        confidence,
        evaluation_score,
        requires_human,
        intent,
    )

    human_decision = interrupt(review_package)
    action = human_decision.get("action", "approve")
    edited_response = human_decision.get("edited_response", "")
    logger.info("Human decision received  action: %s", action)

    if action == "escalate":
        return {
            "final_response": HITL_ESCALATION_RESPONSE,
            "requires_human": True,
            "escalated": True,
        }
    elif action == "edit" and edited_response:
        logger.info("Human edited response  new_response: %s", edited_response)
        return {
            "final_response": edited_response,
            "requires_human": False,
            "escalated": False,
        }
    else:
        logger.info("Human approved response")
        return {
            "requires_human": False,
            "escalated": False,
        }


async def output_guardrails_node(state: AgentState) -> dict:
    logger.info("Running output guardrails node")
    last_message = state["messages"][-1].content
    final_response = state.get("final_response", "")

    if not final_response:
        logger.warning("No final response found in state")
        return {
            "final_response": "I apologise, something went wrong. Please try again."
        }

    safe_response = run_output_checks(last_message, final_response)

    if safe_response != final_response:
        logger.warning("Output guardrails modified the response")

    logger.info("Output guardrails complete  response_length: %d", len(safe_response))
    return {"final_response": safe_response}

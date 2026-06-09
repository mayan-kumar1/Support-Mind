import os
from logger import get_logger

logger = get_logger(__name__)

USE_LLM_GUARD = os.getenv("USE_LLM_GUARD", "true").lower() == "true"

if USE_LLM_GUARD:
    from llm_guard.input_scanners import PromptInjection, Anonymize
    from llm_guard.input_scanners.prompt_injection import MatchType
    from llm_guard.vault import Vault

    vault = Vault()
    injection_scanner = PromptInjection(match_type=MatchType.FULL)
    pii_scanner = Anonymize(vault)
    logger.info("llm-guard input scanners enabled")
else:
    injection_scanner = None
    pii_scanner = None
    logger.warning("llm-guard disabled  using LLM-based checks only")

OFF_TOPIC_SYSTEM_PROMPT = """You are a guardrail for an e-commerce customer support agent.

Decide if the user message is completely off-topic for a customer support context.

Off-topic means: weather, sports, politics, cooking, homework, general knowledge,
entertainment, or anything with zero relation to shopping, orders, products, or customer service.

NOT off-topic — even if vague:
- Questions about rules, policies, or procedures (could be store policies)
- Complaints or expressions of frustration
- Requests for help without specifics
- Greetings or small talk (let these through — agent will handle naturally)
- Anything that could plausibly be related to a purchase or customer service issue

When in doubt, set is_off_topic to False. It is better to let a borderline message
through than to block a legitimate customer."""


def check_prompt_injection(text: str) -> tuple[bool, str]:
    if not USE_LLM_GUARD or injection_scanner is None:
        logger.debug("Injection scanner disabled — skipping")
        return False, ""
    sanitized, is_valid, risk_score = injection_scanner.scan(text)
    if not is_valid:
        logger.warning("Prompt injection detected  risk_score: %.2f", risk_score)
        return True, "Your message contains content that cannot be processed."
    return False, ""


def check_pii(text: str) -> tuple[bool, str]:
    if not USE_LLM_GUARD or pii_scanner is None:
        logger.debug("PII scanner disabled — skipping")
        return False, ""
    sanitized, is_valid, risk_score = pii_scanner.scan(text)
    if not is_valid:
        logger.warning("PII detected in input  risk_score: %.2f", risk_score)
        return (
            True,
            "Please do not share sensitive personal information such as card numbers or passwords.",
        )
    return False, ""


def check_off_topic(text: str) -> tuple[bool, str]:
    from agent.llm import get_llm
    from agent.schemas import OffTopicCheck

    llm = get_llm()
    checker = llm.with_structured_output(OffTopicCheck)

    try:
        result: OffTopicCheck = checker.invoke(
            [
                {"role": "system", "content": OFF_TOPIC_SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ]
        )  # type: ignore
        if result.is_off_topic:
            logger.warning("Off-topic detected  reasoning: %s", result.reasoning)
            return (
                True,
                "I can only help with orders, shipping, returns, and product related queries.",
            )
        logger.info("Off-topic check passed  reasoning: %s", result.reasoning)
        return False, ""
    except Exception as e:
        logger.error("Off-topic check failed  error: %s  defaulting to allow", str(e))
        return False, ""


def run_input_checks(text: str) -> tuple[bool, str]:
    """Run all input checks. Returns (failed, reason)."""
    failed, reason = check_prompt_injection(text)
    if failed:
        return True, reason

    failed, reason = check_pii(text)
    if failed:
        return True, reason

    failed, reason = check_off_topic(text)
    if failed:
        return True, reason

    return False, ""

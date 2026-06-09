import os
from logger import get_logger

logger = get_logger(__name__)

USE_LLM_GUARD = os.getenv("USE_LLM_GUARD", "true").lower() == "true"

if USE_LLM_GUARD:
    from llm_guard.output_scanners import NoRefusal
    from llm_guard.output_scanners.relevance import Relevance
    from llm_guard.vault import Vault

    vault = Vault()
    relevance_scanner = Relevance()
    no_refusal_scanner = NoRefusal()
    logger.info("llm-guard output scanners enabled")
else:
    relevance_scanner = None
    no_refusal_scanner = None
    logger.warning("llm-guard output scanners disabled")


def check_pii_in_output(prompt: str, response: str) -> tuple[bool, str]:
    if not USE_LLM_GUARD:
        logger.debug("Output PII scanner disabled — skipping")
        return False, response
    try:
        from llm_guard.output_scanners import Deanonymize
        from llm_guard.vault import Vault

        deanonymize_scanner = Deanonymize(Vault())
        sanitized, is_valid, risk_score = deanonymize_scanner.scan(prompt, response)
        if not is_valid:
            logger.warning("PII detected in output  risk_score: %.2f", risk_score)
            return True, sanitized
        return False, response
    except Exception as e:
        logger.error("PII output check failed  error: %s", str(e))
        return False, response


def check_relevance(prompt: str, response: str) -> tuple[bool, str]:
    if not USE_LLM_GUARD or relevance_scanner is None:
        logger.debug("Relevance scanner disabled — skipping")
        return False, response
    try:
        sanitized, is_valid, risk_score = relevance_scanner.scan(prompt, response)
        if not is_valid:
            logger.warning("Low relevance detected  risk_score: %.2f", risk_score)
            return (
                True,
                "I apologise, I was unable to generate a relevant response. Please try rephrasing your question.",
            )
        return False, response
    except Exception as e:
        logger.error("Relevance check failed  error: %s", str(e))
        return False, response


def check_no_refusal(prompt: str, response: str) -> tuple[bool, str]:
    if not USE_LLM_GUARD or no_refusal_scanner is None:
        logger.debug("No-refusal scanner disabled — skipping")
        return False, response
    try:
        sanitized, is_valid, risk_score = no_refusal_scanner.scan(prompt, response)
        if not is_valid:
            logger.warning("Unexpected refusal detected  risk_score: %.2f", risk_score)
            return (
                True,
                "I apologise, I was unable to process your request. Please contact our support team directly.",
            )
        return False, response
    except Exception as e:
        logger.error("No refusal check failed  error: %s", str(e))
        return False, response


def check_sanity(response: str) -> tuple[bool, str]:
    if not response or not response.strip():
        logger.warning("Empty response detected")
        return True, "I apologise, something went wrong. Please try again."
    if len(response.strip()) < 10:
        logger.warning("Suspiciously short response  length: %d", len(response.strip()))
        return True, "I apologise, something went wrong. Please try again."
    return False, response


def run_output_checks(prompt: str, response: str) -> str:
    """Run all output checks. Returns the final safe response."""
    flagged, response = check_sanity(response)
    if flagged:
        return response

    flagged, response = check_pii_in_output(prompt, response)
    if flagged:
        logger.info("Response sanitized after PII detection")

    flagged, response = check_relevance(prompt, response)
    if flagged:
        return response

    flagged, response = check_no_refusal(prompt, response)
    if flagged:
        return response

    return response

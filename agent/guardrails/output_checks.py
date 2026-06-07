from llm_guard.output_scanners import Deanonymize, NoRefusal, LanguageSame
from llm_guard.output_scanners.relevance import Relevance
from llm_guard.vault import Vault
from logger import get_logger

logger = get_logger(__name__)

vault = Vault()
deanonymize_scanner = Deanonymize(vault)
no_refusal_scanner = NoRefusal()
relevance_scanner = Relevance()


def check_pii_in_output(prompt: str, response: str) -> tuple[bool, str]:
    """Check if response contains PII that should be redacted."""
    try:
        sanitized, is_valid, risk_score = deanonymize_scanner.scan(prompt, response)
        if not is_valid:
            logger.warning("PII detected in output  risk_score: %.2f", risk_score)
            return True, sanitized  # sanitized has PII replaced
        return False, response
    except Exception as e:
        logger.error("PII output check failed  error: %s", str(e))
        return False, response


def check_relevance(prompt: str, response: str) -> tuple[bool, str]:
    """Check if response is relevant to the input — catches hallucinations."""
    try:
        sanitized, is_valid, risk_score = relevance_scanner.scan(prompt, response)
        if not is_valid:
            logger.warning(
                "Low relevance detected in output  risk_score: %.2f", risk_score
            )
            return (
                True,
                "I apologise, I was unable to generate a relevant response. Please try rephrasing your question or contact our support team directly.",
            )
        return False, response
    except Exception as e:
        logger.error("Relevance check failed  error: %s", str(e))
        return False, response


def check_no_refusal(prompt: str, response: str) -> tuple[bool, str]:
    """Check the response is not an unexplained refusal."""
    try:
        sanitized, is_valid, risk_score = no_refusal_scanner.scan(prompt, response)
        if not is_valid:
            logger.warning(
                "Unexpected refusal detected in output  risk_score: %.2f", risk_score
            )
            return (
                True,
                "I apologise, I was unable to process your request. Please contact our support team directly for assistance.",
            )
        return False, response
    except Exception as e:
        logger.error("No refusal check failed  error: %s", str(e))
        return False, response


def check_sanity(response: str) -> tuple[bool, str]:
    """Basic sanity checks — empty, too short, or gibberish."""
    if not response or not response.strip():
        logger.warning("Empty response detected")
        return (
            True,
            "I apologise, something went wrong. Please try again or contact our support team.",
        )

    if len(response.strip()) < 10:
        logger.warning(
            "Suspiciously short response detected  length: %d", len(response.strip())
        )
        return (
            True,
            "I apologise, something went wrong. Please try again or contact our support team.",
        )

    return False, response


def run_output_checks(prompt: str, response: str) -> str:
    """Run all output checks. Returns the final safe response."""

    # Sanity check first — no point running others on empty response
    flagged, response = check_sanity(response)
    if flagged:
        return response

    # PII redaction — modifies response in place if PII found
    flagged, response = check_pii_in_output(prompt, response)
    if flagged:
        logger.info("Response sanitized after PII detection")
        # Don't return fallback — use sanitized version with PII replaced

    # Relevance check
    flagged, response = check_relevance(prompt, response)
    if flagged:
        return response

    # Refusal check
    flagged, response = check_no_refusal(prompt, response)
    if flagged:
        return response

    return response

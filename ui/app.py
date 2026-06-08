import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv

load_dotenv()

import streamlit as st
import httpx
from logger import get_logger

logger = get_logger(__name__)

# Read from Streamlit secrets in production, env var in development
try:
    API_BASE_URL = st.secrets["API_BASE_URL"]
except Exception:
    API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

HITL_ESCALATION_RESPONSE = "I'm connecting you with one of our human agents who will be able to assist you further. Please hold on."

st.set_page_config(page_title="SupportMind", page_icon="🛍️", layout="wide")

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Session")
    user_id = st.text_input("User ID", value="user_001")
    st.divider()
    st.markdown("**Test queries:**")
    st.markdown("- What is your return policy?")
    st.markdown("- I would like to know status of my order")
    st.markdown("- I want to return order ORD003")
    st.markdown("- I want to speak to a human")
    st.markdown("- Hi / Hola / Hey there")
    st.markdown("- Tell me everything (triggers HITL)")
    st.divider()
    if st.button("Clear conversation"):
        for key in [
            "messages",
            "interrupted",
            "hitl_pending",
            "hitl_package",
            "hitl_config",
            "escalated",
        ]:
            st.session_state[key] = (
                []
                if key == "messages"
                else None if "package" in key or "config" in key else False
            )
        st.rerun()

# ── Session state init ─────────────────────────────────────────────────────────
defaults = {
    "messages": [],
    "interrupted": False,
    "hitl_pending": False,
    "hitl_package": None,
    "escalated": False,
}
for key, default in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ── API helpers ────────────────────────────────────────────────────────────────
def call_chat(message: str, user_id: str) -> dict:
    with httpx.Client(timeout=120.0) as client:
        resp = client.post(
            f"{API_BASE_URL}/chat", json={"message": message, "user_id": user_id}
        )
        resp.raise_for_status()
        return resp.json()


def call_resume(user_id: str, resume_value) -> dict:
    with httpx.Client(timeout=120.0) as client:
        resp = client.post(
            f"{API_BASE_URL}/resume",
            json={"user_id": user_id, "resume_value": resume_value},
        )
        resp.raise_for_status()
        return resp.json()


def handle_api_response(data: dict, response: str) -> str:
    """Process API response and update session state flags."""

    if data.get("escalated"):
        st.session_state.escalated = True

    if data.get("interrupted"):
        interrupt_type = data.get("interrupt_type", "")

        if interrupt_type == "clarification":
            st.session_state.interrupted = True
            return data.get("response", "")

        elif interrupt_type == "hitl":
            st.session_state.hitl_pending = True
            st.session_state.hitl_package = data.get("hitl_package", {})
            return "⏳ This response is pending human review..."

    return data.get("response", "Something went wrong.")


# ── Chat input — top level so Streamlit pins it to bottom ─────────────────────
user_input = st.chat_input(
    "How can I help you today?", disabled=st.session_state.escalated
)

# ── Title and banner ───────────────────────────────────────────────────────────
st.title("🛍️ SupportMind")
st.caption("AI Customer Support Agent")

if st.session_state.escalated:
    st.error(
        "🔴 This conversation has been escalated to a human agent. The AI assistant is no longer active."
    )

# ── Layout ─────────────────────────────────────────────────────────────────────
chat_col, review_col = st.columns([2, 1])

# ── Chat column ────────────────────────────────────────────────────────────────
with chat_col:

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if user_input:
        logger.info("User message  user_id: %s  message: %s", user_id, user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.chat_message("user"):
            st.markdown(user_input)

        response = ""

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    if st.session_state.interrupted:
                        logger.info("Resuming clarification  user_id: %s", user_id)
                        data = call_resume(user_id, user_input)
                        st.session_state.interrupted = False
                    else:
                        data = call_chat(user_input, user_id)

                    response = handle_api_response(data, response)

                    intent = data.get("intent", "")
                    confidence = data.get("confidence", 0.0)
                    evaluation_score = data.get("evaluation_score", 0.0)
                    guardrail_failed = data.get("guardrail_failed", False)

                    st.markdown(response)

                    if guardrail_failed:
                        st.caption("🚫 Blocked by guardrails")
                    elif intent:
                        st.caption(
                            f"Intent: `{intent}` · Confidence: `{confidence:.2f}` · Eval: `{evaluation_score:.2f}`"
                        )

                except httpx.TimeoutException:
                    response = "Request timed out. The agent is taking too long. Please try again."
                    st.markdown(response)
                    logger.error("API timeout  user_id: %s", user_id)

                except httpx.HTTPStatusError as e:
                    response = f"API error: {e.response.status_code}. Please try again."
                    st.markdown(response)
                    logger.error("API HTTP error  status: %s", e.response.status_code)

                except Exception as e:
                    response = "Something went wrong. Please try again."
                    st.markdown(response)
                    logger.error("UI error  error: %s", str(e))

        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()

# ── Human review panel ─────────────────────────────────────────────────────────
with review_col:
    st.subheader("👤 Human Review Panel")

    if not st.session_state.hitl_pending:
        st.info("No responses pending review.")
    else:
        package = st.session_state.hitl_package

        st.warning("⚠️ Response flagged for review")

        with st.expander("Review details", expanded=True):
            st.markdown(
                f"**Customer message:**\n\n{package.get('customer_message', '')}"
            )
            st.divider()
            st.markdown(f"**Intent:** `{package.get('intent', '')}`")
            st.markdown(f"**Confidence:** `{package.get('confidence', 0.0):.2f}`")
            st.markdown(
                f"**Evaluation score:** `{package.get('evaluation_score', 0.0):.2f}`"
            )
            st.markdown(
                f"**Evaluator feedback:** {package.get('evaluation_feedback', '')}"
            )

        st.markdown("**Proposed response:**")
        st.info(package.get("proposed_response", ""))

        st.markdown("**Your decision:**")
        edited = st.text_area(
            "Edit response (optional)",
            value=package.get("proposed_response", ""),
            height=120,
            key="edited_response",
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("✅ Approve", use_container_width=True):
                with st.spinner("Resuming..."):
                    data = call_resume(user_id, {"action": "approve"})
                    final = data.get("response", "")
                    st.session_state.messages.append(
                        {"role": "assistant", "content": final}
                    )
                    st.session_state.hitl_pending = False
                    st.session_state.hitl_package = None
                    if data.get("escalated"):
                        st.session_state.escalated = True
                    logger.info("HITL approved")
                    st.rerun()

        with col2:
            if st.button("✏️ Edit & Send", use_container_width=True):
                with st.spinner("Sending..."):
                    data = call_resume(
                        user_id, {"action": "edit", "edited_response": edited}
                    )
                    final = data.get("response", edited)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": final}
                    )
                    st.session_state.hitl_pending = False
                    st.session_state.hitl_package = None
                    logger.info("HITL edited and sent")
                    st.rerun()

        with col3:
            if st.button("🚨 Escalate", use_container_width=True):
                with st.spinner("Escalating..."):
                    data = call_resume(user_id, {"action": "escalate"})
                    final = data.get("response", HITL_ESCALATION_RESPONSE)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": final}
                    )
                    st.session_state.hitl_pending = False
                    st.session_state.hitl_package = None
                    st.session_state.escalated = True
                    logger.info("HITL escalated")
                    st.rerun()

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from logger import get_logger

logger = get_logger(__name__)

# ── Lazy graph loader ──────────────────────────────────────────────────────────
_agent_graph = None


def get_agent_graph():
    global _agent_graph
    if _agent_graph is None:
        from agent.graph import agent_graph

        _agent_graph = agent_graph
    return _agent_graph


# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="SupportMind API",
    description="AI Customer Support Agent API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response schemas ─────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str
    user_id: str = "default_user"


class ResumeRequest(BaseModel):
    user_id: str
    resume_value: str | dict


class ChatResponse(BaseModel):
    response: str
    intent: str = ""
    confidence: float = 0.0
    evaluation_score: float = 0.0
    guardrail_failed: bool = False
    escalated: bool = False
    interrupted: bool = False
    interrupt_type: str = ""
    interrupt_value: str | dict = ""
    hitl_package: dict = {}


# ── Health check ───────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "service": "SupportMind API"}


# ── Chat endpoint ──────────────────────────────────────────────────────────────
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    logger.info(
        "Chat request  user_id: %s  message: %s", request.user_id, request.message
    )

    config = {"configurable": {"thread_id": request.user_id}}

    initial_state = {
        "messages": [HumanMessage(content=request.message)],
        "intent": "",
        "retrieved_docs": [],
        "tool_result": {},
        "confidence": 0.0,
        "requires_human": False,
        "final_response": "",
        "evaluation_score": 1.0,
        "evaluation_feedback": "",
        "clarification_count": 0,
        "awaiting_clarification": False,
        "clarification_topic": "",
        "user_id": request.user_id,
        "guardrail_failed": False,
        "guardrail_reason": "",
        "escalated": False,
        "cache_hit": False,
    }

    try:
        result = await get_agent_graph().ainvoke(initial_state, config=config)  # type: ignore
        return await _build_response(result, config)

    except Exception as e:
        logger.error("Chat endpoint error  error: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))


# ── Resume endpoint ────────────────────────────────────────────────────────────
@app.post("/resume", response_model=ChatResponse)
async def resume(request: ResumeRequest):
    logger.info("Resume request  user_id: %s", request.user_id)

    config = {"configurable": {"thread_id": request.user_id}}

    try:
        if isinstance(request.resume_value, str):
            result = await get_agent_graph().ainvoke(
                Command(
                    resume=request.resume_value,
                    update={"messages": [HumanMessage(content=request.resume_value)]},
                ),
                config=config,  # type: ignore
            )
        else:
            result = await get_agent_graph().ainvoke(
                Command(resume=request.resume_value),
                config=config,  # type: ignore
            )

        return await _build_response(result, config)

    except Exception as e:
        logger.error("Resume endpoint error  error: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))


# ── Helper ─────────────────────────────────────────────────────────────────────
async def _build_response(result: dict, config: dict) -> ChatResponse:
    graph_state = await get_agent_graph().aget_state(config)  # type: ignore

    if graph_state.next and graph_state.tasks:
        tasks = graph_state.tasks
        interrupts = tasks[0].interrupts if tasks else []

        if interrupts:
            interrupt_value = interrupts[0].value

            if isinstance(interrupt_value, str):
                logger.info("Clarification interrupt  question: %s", interrupt_value)
                return ChatResponse(
                    response=interrupt_value,
                    interrupted=True,
                    interrupt_type="clarification",
                    interrupt_value=interrupt_value,
                )

            elif isinstance(interrupt_value, dict):
                logger.info("HITL interrupt detected")
                return ChatResponse(
                    response="⏳ This response is pending human review...",
                    interrupted=True,
                    interrupt_type="hitl",
                    interrupt_value=interrupt_value,
                    hitl_package=interrupt_value,
                    intent=result.get("intent", "") if isinstance(result, dict) else "",
                    confidence=(
                        result.get("confidence", 0.0)
                        if isinstance(result, dict)
                        else 0.0
                    ),
                    evaluation_score=(
                        result.get("evaluation_score", 0.0)
                        if isinstance(result, dict)
                        else 0.0
                    ),
                )

    escalated = graph_state.values.get("escalated", False)

    return ChatResponse(
        response=result.get("final_response", "Something went wrong."),
        intent=result.get("intent", ""),
        confidence=result.get("confidence", 0.0),
        evaluation_score=result.get("evaluation_score", 0.0),
        guardrail_failed=result.get("guardrail_failed", False),
        escalated=escalated,
        interrupted=False,
    )

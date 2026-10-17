"""HTTP API.

Run locally with:
    uv run uvicorn raposa.api:app --reload
"""

from typing import Annotated, Literal

import anthropic
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from raposa.agent import RaposaAgent
from raposa.personas import DEFAULT_PERSONA, PERSONAS
from raposa.runtime import build_agent

app = FastAPI(title="Raposa Agent", version="0.1.0")


class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    persona: str = DEFAULT_PERSONA
    history: list[Turn] = Field(default_factory=list, max_length=20)


class ToolCallOut(BaseModel):
    name: str
    input: dict
    is_error: bool


class ChatResponse(BaseModel):
    answer: str
    stop_reason: str | None
    tool_calls: list[ToolCallOut]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/personas")
def personas() -> list[dict[str, str]]:
    return [
        {"key": p.key, "label": p.label, "description": p.description} for p in PERSONAS.values()
    ]


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest, agent: Annotated[RaposaAgent, Depends(build_agent)]) -> ChatResponse:
    if request.persona not in PERSONAS:
        raise HTTPException(422, f"Unknown persona. Options: {sorted(PERSONAS)}")
    try:
        reply = agent.ask(
            request.message,
            persona=request.persona,
            history=[turn.model_dump() for turn in request.history],
        )
    except anthropic.RateLimitError as exc:
        raise HTTPException(429, "Rate limited by the model provider, try again shortly") from exc
    except anthropic.APIStatusError as exc:
        raise HTTPException(502, f"Model provider error ({exc.status_code})") from exc
    except anthropic.APIConnectionError as exc:
        raise HTTPException(503, "Could not reach the model provider") from exc
    return ChatResponse(
        answer=reply.text,
        stop_reason=reply.stop_reason,
        tool_calls=[
            ToolCallOut(name=c.name, input=c.input, is_error=c.is_error) for c in reply.tool_calls
        ],
    )

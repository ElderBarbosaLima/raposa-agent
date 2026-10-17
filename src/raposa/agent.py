"""The agent loop: Claude decides which tools to call, we execute them and feed results back.

The loop is written by hand (rather than with the SDK's beta tool runner) because every
tool call is recorded in an `AgentReply` trace, which the eval suite uses to check that
answers are grounded in tool outputs.
"""

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import anthropic

from raposa.config import Settings, get_settings
from raposa.personas import DEFAULT_PERSONA, get_persona
from raposa.tools import Toolbox, ToolError

log = logging.getLogger(__name__)

# Server-side retry on Anthropic's recommended model if a request is declined by a safety
# classifier; routing is decided per refusal category.
FALLBACK_BETA = "server-side-fallback-2026-07-01"

REFUSAL_MESSAGE = "Sorry, I can't help with that one. Ask me anything about Cruzeiro!"
MAX_ROUNDS_MESSAGE = "I couldn't finish looking that up. Could you rephrase the question?"


@dataclass
class ToolCall:
    name: str
    input: dict[str, Any]
    output: str
    is_error: bool = False


@dataclass
class AgentReply:
    text: str
    stop_reason: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)


class RaposaAgent:
    def __init__(
        self,
        toolbox: Toolbox,
        *,
        client: anthropic.Anthropic | None = None,
        settings: Settings | None = None,
        today: date | None = None,
    ) -> None:
        self._toolbox = toolbox
        self._client = client or anthropic.Anthropic()
        self._settings = settings or get_settings()
        self._today = today

    def ask(
        self,
        question: str,
        *,
        persona: str = DEFAULT_PERSONA,
        history: list[dict[str, Any]] | None = None,
    ) -> AgentReply:
        """Answer one user question. `history` holds earlier {"role", "content"} turns."""
        messages: list[dict[str, Any]] = [*(history or []), {"role": "user", "content": question}]
        system = self._system(persona)
        calls: list[ToolCall] = []

        for _ in range(self._settings.max_tool_rounds):
            response = self._client.beta.messages.create(
                model=self._settings.model,
                max_tokens=16000,
                system=system,
                tools=self._toolbox.definitions,
                messages=messages,
                output_config={"effort": self._settings.effort},
                betas=[FALLBACK_BETA],
                fallbacks="default",
            )

            if response.stop_reason == "refusal":
                log.warning("Request declined: %s", response.stop_details)
                return AgentReply(REFUSAL_MESSAGE, response.stop_reason, calls)

            if response.stop_reason != "tool_use":
                return AgentReply(_text(response.content), response.stop_reason, calls)

            messages.append({"role": "assistant", "content": _echoable(response.content)})
            messages.append({"role": "user", "content": self._run_tools(response.content, calls)})

        return AgentReply(MAX_ROUNDS_MESSAGE, "max_tool_rounds", calls)

    def _system(self, persona: str) -> list[dict[str, Any]]:
        today = self._today or date.today()
        return [
            # Stable prefix (tools + persona) is cached; the date goes after the breakpoint.
            {
                "type": "text",
                "text": get_persona(persona).system_prompt,
                "cache_control": {"type": "ephemeral"},
            },
            {"type": "text", "text": f"Today's date is {today.isoformat()}."},
        ]

    def _run_tools(self, content: list[Any], calls: list[ToolCall]) -> list[dict[str, Any]]:
        # All results go back in a single user message so Claude keeps calling tools in parallel.
        results = []
        for block in content:
            if block.type != "tool_use":
                continue
            try:
                output, is_error = self._toolbox.run(block.name, block.input), False
            except ToolError as exc:
                output, is_error = f"Error: {exc}", True
            calls.append(ToolCall(block.name, dict(block.input), output, is_error))
            results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": output,
                    "is_error": is_error,
                }
            )
        return results


def _text(content: list[Any]) -> str:
    return "".join(block.text for block in content if block.type == "text").strip()


def _echoable(content: list[Any]) -> list[Any]:
    """Drop non-text blocks produced before a mid-output fallback, as the API requires."""
    boundary = max((i for i, b in enumerate(content) if b.type == "fallback"), default=None)
    if boundary is None:
        return content
    before = [b for b in content[:boundary] if b.type == "text"]
    return before + content[boundary + 1 :]

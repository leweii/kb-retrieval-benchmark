"""Uniform chat-with-tools interface over the four models under test.

  haiku, sonnet  -> Anthropic API (anthropic SDK), effort medium, top-level prompt caching
  deepseek       -> DeepSeek official API (OpenAI-compatible), model deepseek-flash
  qwen           -> OpenRouter (OpenAI-compatible), model qwen/qwen3.8-flash, provider Alibaba

A Conversation keeps provider-native message history (Claude content blocks are passed back unchanged, as the
API requires for thinking blocks). Each turn returns text, tool calls and a Usage with dollar cost computed from
the provider's price rules (Claude cache read/write tiers and Haiku's >100K-prompt tier; DeepSeek peak/off-peak;
Qwen as reported by OpenRouter). No server-side model fallbacks: a refusal is recorded, never silently rerouted.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

import anthropic
from openai import OpenAI

MODELS = {
    "haiku": {"provider": "anthropic", "id": "claude-haiku-5-5"},
    "sonnet": {"provider": "anthropic", "id": "claude-sonnet-5-5"},
    "deepseek": {"provider": "deepseek", "id": "deepseek-flash"},
    "qwen": {"provider": "openrouter", "id": "qwen/qwen3.8-flash"},
}
EFFORT = "medium"  # Claude; DeepSeek has no medium (default high); Qwen default

# $/M tokens: (input, output, cache_read, cache_write_5m)
CLAUDE_PRICES = {
    "claude-sonnet-5-5": (2.00, 10.00, 0.10, 2.50),
    "claude-haiku-5-5": (0.10, 0.50, 0.01, 0.125),
}
HAIKU_LONG = (0.50, 2.50, 0.05, 0.625)  # prompts over 100K tokens
# DeepSeek deepseek-flash: (cache_hit, cache_miss, output) peak / off-peak
DEEPSEEK_PRICES = {"peak": (0.006, 0.30, 1.20), "off": (0.003, 0.15, 0.60)}


def deepseek_period(ts: float) -> str:
    t = datetime.fromtimestamp(ts, timezone.utc)
    if t.weekday() < 5 and (1 <= t.hour < 4 or 6 <= t.hour < 10):
        return "peak"
    return "off"


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read: int = 0
    cache_write: int = 0
    cost: float = 0.0
    calls: int = 0

    def add(self, other: "Usage") -> None:
        for k in ("input_tokens", "output_tokens", "cache_read", "cache_write", "cost", "calls"):
            setattr(self, k, getattr(self, k) + getattr(other, k))


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict


@dataclass
class Turn:
    text: str
    tool_calls: list[ToolCall]
    usage: Usage
    stop: str  # end | tool_use | max_tokens | refusal | error
    latency: float = 0.0


@dataclass
class Tool:
    name: str
    description: str
    schema: dict  # JSON schema of the arguments


class Conversation:
    def __init__(self, model: str, system: str, tools: list[Tool]):
        self.model = model
        self.spec = MODELS[model]
        self.system = system
        self.tools = tools
        self.messages: list = []
        self.usage = Usage()
        if self.spec["provider"] == "anthropic":
            self.client = anthropic.Anthropic(max_retries=6)
        elif self.spec["provider"] == "deepseek":
            self.client = OpenAI(base_url="https://api.deepseek.com", api_key=os.environ["DEEPSEEK_API_KEY"],
                                 max_retries=6, timeout=600)
        else:
            self.client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"],
                                 max_retries=6, timeout=600)

    # ------------------------------------------------------------------ public

    def user(self, text: str) -> Turn:
        if self.spec["provider"] == "anthropic":
            self.messages.append({"role": "user", "content": text})
        else:
            if not self.messages:
                self.messages.append({"role": "system", "content": self.system})
            self.messages.append({"role": "user", "content": text})
        return self._call()

    def tool_results(self, results: list[tuple[ToolCall, str]]) -> Turn:
        if self.spec["provider"] == "anthropic":
            self.messages.append({"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": tc.id, "content": out} for tc, out in results]})
        else:
            for tc, out in results:
                self.messages.append({"role": "tool", "tool_call_id": tc.id, "content": out})
        return self._call()

    # ------------------------------------------------------------------ providers

    def _call(self) -> Turn:
        t0 = time.time()
        try:
            turn = self._anthropic() if self.spec["provider"] == "anthropic" else self._openai_compat()
        except (anthropic.BadRequestError, anthropic.APIStatusError) as e:  # non-retryable after SDK retries
            turn = Turn(text=f"[error] {type(e).__name__}: {str(e)[:300]}", tool_calls=[], usage=Usage(), stop="error")
        turn.latency = time.time() - t0
        self.usage.add(turn.usage)
        return turn

    def _anthropic(self) -> Turn:
        tools = [{"name": t.name, "description": t.description, "input_schema": t.schema} for t in self.tools]
        kwargs = dict(model=self.spec["id"], max_tokens=16000, system=self.system, messages=self.messages,
                      output_config={"effort": EFFORT}, cache_control={"type": "ephemeral"})
        if tools:
            kwargs["tools"] = tools
        r = self.client.messages.create(**kwargs)
        self.messages.append({"role": "assistant", "content": r.content})  # unchanged (thinking blocks included)
        u = r.usage
        cr = getattr(u, "cache_read_input_tokens", 0) or 0
        cw = getattr(u, "cache_creation_input_tokens", 0) or 0
        prompt_total = u.input_tokens + cr + cw
        pin, pout, pcr, pcw = CLAUDE_PRICES[self.spec["id"]]
        if self.spec["id"] == "claude-haiku-5-5" and prompt_total > 100_000:
            pin, pout, pcr, pcw = HAIKU_LONG
        cost = (u.input_tokens * pin + u.output_tokens * pout + cr * pcr + cw * pcw) / 1e6
        usage = Usage(u.input_tokens, u.output_tokens, cr, cw, cost, 1)
        text = "".join(b.text for b in r.content if b.type == "text")
        calls = [ToolCall(b.id, b.name, dict(b.input)) for b in r.content if b.type == "tool_use"]
        stop = {"end_turn": "end", "tool_use": "tool_use", "max_tokens": "max_tokens",
                "refusal": "refusal"}.get(r.stop_reason, r.stop_reason or "end")
        return Turn(text, calls, usage, stop)

    def _openai_compat(self) -> Turn:
        tools = [{"type": "function", "function": {"name": t.name, "description": t.description,
                                                   "parameters": t.schema}} for t in self.tools]
        kwargs = dict(model=self.spec["id"], messages=self.messages, max_tokens=16000)
        if tools:
            kwargs["tools"] = tools
        extra = {}
        if self.spec["provider"] == "openrouter":
            extra = {"usage": {"include": True}, "provider": {"order": ["Alibaba"], "allow_fallbacks": False}}
        ts = time.time()
        r = self.client.chat.completions.create(**kwargs, extra_body=extra or None)
        m = r.choices[0].message
        msg = {"role": "assistant", "content": m.content or ""}
        if m.tool_calls:
            msg["tool_calls"] = [{"id": tc.id, "type": "function",
                                  "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                                 for tc in m.tool_calls]
        self.messages.append(msg)
        u = r.usage
        if self.spec["provider"] == "deepseek":
            hit = getattr(u, "prompt_cache_hit_tokens", 0) or 0
            miss = getattr(u, "prompt_cache_miss_tokens", u.prompt_tokens - hit) or 0
            ph, pm, po = DEEPSEEK_PRICES[deepseek_period(ts)]
            cost = (hit * ph + miss * pm + u.completion_tokens * po) / 1e6
            usage = Usage(miss, u.completion_tokens, hit, 0, cost, 1)
        else:
            cached = getattr(getattr(u, "prompt_tokens_details", None), "cached_tokens", 0) or 0
            cost = float(getattr(u, "cost", 0) or 0)
            usage = Usage(u.prompt_tokens - cached, u.completion_tokens, cached, 0, cost, 1)
        calls = []
        for tc in m.tool_calls or []:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {"_invalid_json": tc.function.arguments}
            calls.append(ToolCall(tc.id, tc.function.name, args))
        fr = r.choices[0].finish_reason
        stop = "tool_use" if calls else {"stop": "end", "length": "max_tokens",
                                         "content_filter": "refusal"}.get(fr, fr or "end")
        return Turn(m.content or "", calls, usage, stop)

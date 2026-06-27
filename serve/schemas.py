from __future__ import annotations

from pydantic import BaseModel


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int | None = None
    temperature: float | None = None
    top_k: int | None = None
    top_p: float | None = None


class GenerateResponse(BaseModel):
    text: str
    tokens: list[int]


class ChatMessage(BaseModel):
    role: str
    content: str = ""


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    max_new_tokens: int | None = None
    temperature: float | None = None
    tools: bool = False


class ChatResponse(BaseModel):
    message: ChatMessage
    tool_calls: list[dict] = []


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    vocab_size: int | None = None
    n_params: int | None = None

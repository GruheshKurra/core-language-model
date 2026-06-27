from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Settings:
    model_dir: str | None
    host: str
    port: int
    max_new_tokens: int
    temperature: float
    top_k: int | None
    top_p: float | None
    sandbox_dir: str
    allow_shell: bool
    allow_write: bool


def _opt_int(name: str) -> int | None:
    value = os.environ.get(name)
    return int(value) if value else None


def _opt_float(name: str) -> float | None:
    value = os.environ.get(name)
    return float(value) if value else None


def get_settings() -> Settings:
    return Settings(
        model_dir=os.environ.get("MODEL_DIR"),
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "8000")),
        max_new_tokens=int(os.environ.get("MAX_NEW_TOKENS", "128")),
        temperature=float(os.environ.get("TEMPERATURE", "0.8")),
        top_k=_opt_int("TOP_K"),
        top_p=_opt_float("TOP_P"),
        sandbox_dir=os.environ.get("TOOL_SANDBOX", str(Path.cwd() / "serve" / "sandbox")),
        allow_shell=os.environ.get("ALLOW_SHELL", "0") == "1",
        allow_write=os.environ.get("ALLOW_WRITE", "0") == "1",
    )

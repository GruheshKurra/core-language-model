from __future__ import annotations

from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException

from serve import schemas
from serve.config import get_settings
from zyn.agent import ToolAgent
from zyn.bundle import load_bundle
from zyn.chat import build_prompt
from zyn.generate import generate
from zyn.tools import Sandbox


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="zyn core language model")
    app.state.settings = settings
    app.state.bundle = None
    if settings.model_dir and Path(settings.model_dir).exists():
        app.state.bundle = load_bundle(settings.model_dir)

    def _require_bundle():
        if app.state.bundle is None:
            raise HTTPException(status_code=503, detail="model not loaded")
        return app.state.bundle

    @app.get("/health", response_model=schemas.HealthResponse)
    def health() -> schemas.HealthResponse:
        bundle = app.state.bundle
        if bundle is None:
            return schemas.HealthResponse(status="ok", model_loaded=False)
        return schemas.HealthResponse(
            status="ok",
            model_loaded=True,
            vocab_size=bundle.config.vocab_size,
            n_params=bundle.model.num_params(),
        )

    @app.post("/generate", response_model=schemas.GenerateResponse)
    def generate_endpoint(req: schemas.GenerateRequest) -> schemas.GenerateResponse:
        bundle = _require_bundle()
        s = app.state.settings
        try:
            ids = bundle.tokenizer.encode(req.prompt, add_bos=True)
            out = generate(
                bundle.model,
                np.asarray(ids),
                max_new_tokens=req.max_new_tokens or s.max_new_tokens,
                temperature=req.temperature if req.temperature is not None else s.temperature,
                top_k=req.top_k if req.top_k is not None else s.top_k,
                top_p=req.top_p if req.top_p is not None else s.top_p,
                eos_id=bundle.tokenizer.eos_id,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        new_ids = out[0, len(ids):].tolist()
        return schemas.GenerateResponse(
            text=bundle.tokenizer.decode(new_ids, skip_specials=True),
            tokens=new_ids,
        )

    @app.post("/chat", response_model=schemas.ChatResponse)
    def chat_endpoint(req: schemas.ChatRequest) -> schemas.ChatResponse:
        bundle = _require_bundle()
        s = app.state.settings
        messages = [m.model_dump() for m in req.messages]
        temperature = req.temperature if req.temperature is not None else s.temperature
        max_new = req.max_new_tokens or s.max_new_tokens

        try:
            if req.tools:
                sandbox = Sandbox(s.sandbox_dir, allow_write=s.allow_write, allow_shell=s.allow_shell)
                agent = ToolAgent(
                    bundle.tokenizer,
                    sandbox,
                    max_new_tokens=max_new,
                    temperature=temperature,
                )
                result = agent.run(bundle.cached_model(), messages)
                return schemas.ChatResponse(
                    message=schemas.ChatMessage(role="assistant", content=result["answer"]),
                    tool_calls=result["tool_calls"],
                )

            ids = build_prompt(messages, bundle.tokenizer)
            out = generate(
                bundle.model,
                np.asarray(ids),
                max_new_tokens=max_new,
                temperature=temperature,
                top_k=s.top_k,
                top_p=s.top_p,
                eos_id=bundle.tokenizer.eos_id,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        new_ids = out[0, len(ids):].tolist()
        return schemas.ChatResponse(
            message=schemas.ChatMessage(
                role="assistant",
                content=bundle.tokenizer.decode(new_ids, skip_specials=True),
            ),
            tool_calls=[],
        )

    return app


app = create_app()

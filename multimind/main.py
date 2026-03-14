from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from multimind.config import APP_NAME, DEFAULT_HOST, DEFAULT_PORT, PIPELINE_STEPS, STATIC_DIR, TEMPLATE_DIR
from multimind.discovery import ProviderInfo, discover_providers, normalize_base_url, select_default_provider
from multimind.llm_client import LocalLLMClient
from multimind.pipeline import run_pipeline, run_council_pipeline
from multimind.org_pipeline import run_org_pipeline


class SettingsPayload(BaseModel):
    provider_name: str = ""
    provider_kind: str = "ollama"
    base_url: str = "http://127.0.0.1:11434"
    ollama_think: bool = False
    model_map: dict[str, str] = Field(default_factory=dict)
    council_models: list[str] = Field(default_factory=list)
    judge_model: str = ""
    org_model: str = ""


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    mode: str = Field(default="hard")


@dataclass
class RuntimeState:
    providers: list[ProviderInfo] = field(default_factory=list)
    settings: SettingsPayload = field(default_factory=SettingsPayload)

    def to_dict(self) -> dict[str, Any]:
        return {
            "providers": [provider.to_dict() for provider in self.providers],
            "settings": self.settings.model_dump(),
        }


app = FastAPI(title=APP_NAME)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.state.runtime = RuntimeState()
app.state.llm_client = LocalLLMClient()


def _default_model_map(provider: ProviderInfo | None) -> dict[str, str]:
    model = provider.models[0] if provider and provider.models else ""
    return {step: model for step in PIPELINE_STEPS}


def _merge_model_map(existing: dict[str, str], fallback: dict[str, str]) -> dict[str, str]:
    merged: dict[str, str] = {}
    for step in PIPELINE_STEPS:
        merged[step] = existing.get(step) or fallback.get(step) or ""
    return merged


async def _refresh_runtime() -> None:
    providers = await discover_providers()
    runtime: RuntimeState = app.state.runtime
    runtime.providers = providers
    selected_provider = select_default_provider(providers)

    if not runtime.settings.provider_name:
        if selected_provider:
            runtime.settings.provider_name = selected_provider.name
            runtime.settings.provider_kind = selected_provider.kind
            runtime.settings.base_url = selected_provider.base_url
            runtime.settings.model_map = _default_model_map(selected_provider)
            default_model = selected_provider.models[0] if selected_provider.models else ""
            runtime.settings.council_models = [default_model] if default_model else []
            runtime.settings.judge_model = default_model
            runtime.settings.org_model = default_model
        return

    matching_provider = next(
        (
            provider
            for provider in providers
            if provider.name == runtime.settings.provider_name
            and normalize_base_url(provider.base_url) == normalize_base_url(runtime.settings.base_url)
        ),
        None,
    )

    fallback_map = _default_model_map(matching_provider or selected_provider)
    runtime.settings.model_map = _merge_model_map(runtime.settings.model_map, fallback_map)

    if not runtime.settings.council_models:
        default_model = (matching_provider or selected_provider).models[0] if (matching_provider or selected_provider) and (matching_provider or selected_provider).models else ""
        if default_model:
            runtime.settings.council_models = [default_model]

    if not runtime.settings.judge_model:
        default_model = (matching_provider or selected_provider).models[0] if (matching_provider or selected_provider) and (matching_provider or selected_provider).models else ""
        if default_model:
            runtime.settings.judge_model = default_model

    if not runtime.settings.org_model:
        default_model = (matching_provider or selected_provider).models[0] if (matching_provider or selected_provider) and (matching_provider or selected_provider).models else ""
        if default_model:
            runtime.settings.org_model = default_model


@app.on_event("startup")
async def startup_event() -> None:
    await _refresh_runtime()


@app.on_event("shutdown")
async def shutdown_event() -> None:
    await app.state.llm_client.aclose()


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(Path(TEMPLATE_DIR) / "index.html")


@app.get("/api/health")
async def health() -> JSONResponse:
    return JSONResponse({"ok": True})


@app.get("/api/settings")
async def get_settings() -> JSONResponse:
    runtime: RuntimeState = app.state.runtime
    return JSONResponse(runtime.to_dict())


@app.post("/api/providers/refresh")
async def refresh_providers() -> JSONResponse:
    await _refresh_runtime()
    runtime: RuntimeState = app.state.runtime
    return JSONResponse(runtime.to_dict())


@app.post("/api/settings")
async def update_settings(payload: SettingsPayload) -> JSONResponse:
    runtime: RuntimeState = app.state.runtime
    runtime.settings = SettingsPayload(
        provider_name=payload.provider_name,
        provider_kind=payload.provider_kind,
        base_url=normalize_base_url(payload.base_url),
        ollama_think=payload.ollama_think,
        model_map=_merge_model_map(payload.model_map, runtime.settings.model_map),
        council_models=payload.council_models or runtime.settings.council_models,
        judge_model=payload.judge_model or runtime.settings.judge_model,
        org_model=payload.org_model or runtime.settings.org_model,
    )
    return JSONResponse(runtime.to_dict())


@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    runtime: RuntimeState = app.state.runtime
    settings = runtime.settings
    message = request.message.strip()
    mode = request.mode.strip().lower()

    if mode not in {"off", "medium", "hard", "council", "org"}:
        raise HTTPException(status_code=400, detail="Mode must be one of off, medium, hard, council, or org.")

    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    if mode == "council":
        if not settings.council_models:
            raise HTTPException(status_code=400, detail="At least one council model must be selected.")
        if not settings.judge_model:
            raise HTTPException(status_code=400, detail="A judge model must be selected.")

    if mode == "org":
        if not settings.org_model:
            raise HTTPException(status_code=400, detail="An org model must be selected.")

    async def event_stream():
        try:
            if mode == "org":
                async for event in run_org_pipeline(
                    client=app.state.llm_client,
                    provider_kind=settings.provider_kind,
                    base_url=settings.base_url,
                    model=settings.org_model,
                    ollama_think=settings.ollama_think,
                    user_message=message,
                ):
                    yield json.dumps(event) + "\n"
            elif mode == "council":
                async for event in run_council_pipeline(
                    client=app.state.llm_client,
                    provider_kind=settings.provider_kind,
                    base_url=settings.base_url,
                    council_models=settings.council_models,
                    judge_model=settings.judge_model,
                    ollama_think=settings.ollama_think,
                    user_message=message,
                ):
                    yield json.dumps(event) + "\n"
            else:
                async for event in run_pipeline(
                    client=app.state.llm_client,
                    provider_kind=settings.provider_kind,
                    base_url=settings.base_url,
                    model_map=settings.model_map,
                    ollama_think=settings.ollama_think,
                    user_message=message,
                    mode=mode,
                ):
                    yield json.dumps(event) + "\n"
        except Exception as exc:
            yield json.dumps({"type": "error", "message": str(exc)}) + "\n"

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")


def run() -> None:
    uvicorn.run(
        "multimind.main:app",
        host=DEFAULT_HOST,
        port=DEFAULT_PORT,
        reload=False,
    )
"""Core benchmark harness for MultiMind pipeline evaluation.

Runs test cases through each pipeline mode (off, medium, hard, council, org)
and captures structured results for scoring and reporting.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from multimind.llm_client import LocalLLMClient
from multimind.pipeline import run_council_pipeline, run_pipeline
from multimind.org_pipeline import run_org_pipeline


# ---------------------------------------------------------------------------
# Result data class
# ---------------------------------------------------------------------------

@dataclass
class RunResult:
    """Result of a single benchmark run."""

    model: str
    mode: str
    suite: str
    question_id: str
    question: str
    expected_answer: str
    final_answer: str
    all_stage_outputs: dict[str, str] = field(default_factory=dict)
    wall_time_seconds: float = 0.0
    total_output_chars: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Pipeline consumption helpers
# ---------------------------------------------------------------------------

async def _consume_pipeline(pipeline_iter) -> tuple[str, dict[str, str]]:
    """Consume an async pipeline generator, return (final_answer, stage_outputs)."""
    outputs: dict[str, str] = {}
    final = ""

    async for event in pipeline_iter:
        etype = event.get("type", "")

        if etype == "step-complete":
            outputs[event["step"]] = event.get("content", "")
        elif etype == "org-node-complete":
            outputs[event["node_id"]] = event.get("content", "")
        elif etype == "answer-complete":
            outputs[event["step"]] = event.get("content", "")
        elif etype == "run-complete":
            final = event.get("outputs", {}).get("final", "")

    return final, outputs


# ---------------------------------------------------------------------------
# Single-run execution
# ---------------------------------------------------------------------------

async def run_single(
    *,
    client: LocalLLMClient,
    provider_kind: str,
    base_url: str,
    model: str,
    mode: str,
    question: str,
    question_id: str,
    expected_answer: str,
    suite: str,
    council_models: list[str] | None = None,
    judge_model: str | None = None,
    plan_model: str | None = None,
    execute_model: str | None = None,
    critique_model: str | None = None,
    ollama_think: bool = False,
) -> RunResult:
    """Run a single question through a specific pipeline mode."""

    start = time.perf_counter()

    if mode in ("off", "medium", "hard"):
        model_map = {
            "plan": plan_model or model,
            "execute": execute_model or model,
            "critique": critique_model or model,
        }
        pipeline = run_pipeline(
            client=client,
            provider_kind=provider_kind,
            base_url=base_url,
            model_map=model_map,
            ollama_think=ollama_think,
            user_message=question,
            mode=mode,
        )

    elif mode == "council":
        c_models = council_models or [model, model, model]
        j_model = judge_model or model
        pipeline = run_council_pipeline(
            client=client,
            provider_kind=provider_kind,
            base_url=base_url,
            council_models=c_models,
            judge_model=j_model,
            ollama_think=ollama_think,
            user_message=question,
        )

    elif mode == "org":
        pipeline = run_org_pipeline(
            client=client,
            provider_kind=provider_kind,
            base_url=base_url,
            model=model,
            ollama_think=ollama_think,
            user_message=question,
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")

    final_answer, all_outputs = await _consume_pipeline(pipeline)
    elapsed = time.perf_counter() - start

    total_chars = sum(len(v) for v in all_outputs.values())

    # Mode-specific metadata
    meta: dict[str, Any] = {}
    if mode == "hard":
        exec_out = all_outputs.get("execute", "")
        crit_out = all_outputs.get("critique", "")
        meta["self_corrected"] = exec_out.strip() != crit_out.strip()

    if mode == "council":
        # Collect individual advisor outputs for diversity analysis
        advisor_outputs = {
            k: v for k, v in all_outputs.items() if k.startswith("council-")
        }
        meta["num_advisors"] = len(advisor_outputs)
        meta["advisor_output_lengths"] = {
            k: len(v) for k, v in advisor_outputs.items()
        }

    return RunResult(
        model=model,
        mode=mode,
        suite=suite,
        question_id=question_id,
        question=question,
        expected_answer=expected_answer,
        final_answer=final_answer,
        all_stage_outputs=all_outputs,
        wall_time_seconds=elapsed,
        total_output_chars=total_chars,
        metadata=meta,
    )


# ---------------------------------------------------------------------------
# Batch execution
# ---------------------------------------------------------------------------

async def run_suite(
    *,
    client: LocalLLMClient,
    provider_kind: str,
    base_url: str,
    model: str,
    mode: str,
    suite_name: str,
    questions: list[dict[str, str]],
    council_models: list[str] | None = None,
    judge_model: str | None = None,
    plan_model: str | None = None,
    execute_model: str | None = None,
    critique_model: str | None = None,
    ollama_think: bool = False,
    on_progress: Any = None,
    output_path: Path | None = None,
) -> list[RunResult]:
    """Run an entire test suite sequentially and return all results.

    Saves results incrementally if output_path is provided.
    """
    results: list[RunResult] = []
    total = len(questions)

    for idx, q in enumerate(questions):
        result = await run_single(
            client=client,
            provider_kind=provider_kind,
            base_url=base_url,
            model=model,
            mode=mode,
            question=q["question"],
            question_id=q["id"],
            expected_answer=q["expected"],
            suite=suite_name,
            council_models=council_models,
            judge_model=judge_model,
            plan_model=plan_model,
            execute_model=execute_model,
            critique_model=critique_model,
            ollama_think=ollama_think,
        )
        results.append(result)

        # Incremental save to prevent data loss on keyboard interrupt
        if output_path:
            save_results(results, output_path)

        if on_progress:
            on_progress(idx + 1, total, result)

    return results


def save_results(results: list[RunResult], output_path: Path) -> None:
    """Save benchmark results to a JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = [r.to_dict() for r in results]
    output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))


def load_results(path: Path) -> list[dict[str, Any]]:
    """Load benchmark results from a JSON file."""
    return json.loads(path.read_text())

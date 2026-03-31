"""CLI entry point for the MultiMind benchmark suite.

Usage:
    python -m benchmarks.run_benchmarks \\
        --models qwen3.5:0.8b,qwen3.5:4b,qwen3.5:9b \\
        --modes off,medium,hard,council,org \\
        --suites gsm8k_mini,code_gen,fullstack,council_qa \\
        --output benchmarks/results/
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.harness import RunResult, load_results, run_suite, save_results
from benchmarks.scorer import compute_suite_metrics, score_code_pass, score_exact_numeric
from multimind.llm_client import LocalLLMClient


# ---------------------------------------------------------------------------
# Suite loaders
# ---------------------------------------------------------------------------

def _load_suite(name: str) -> tuple[list[dict], str]:
    """Load a test suite by name. Returns (questions, score_type)."""
    if name == "gsm8k_mini":
        from benchmarks.suites.gsm8k_mini import QUESTIONS, SCORE_TYPE
        return QUESTIONS, SCORE_TYPE
    elif name == "code_gen":
        from benchmarks.suites.code_gen import QUESTIONS, SCORE_TYPE
        return QUESTIONS, SCORE_TYPE
    elif name == "fullstack":
        from benchmarks.suites.fullstack import QUESTIONS, SCORE_TYPE
        return QUESTIONS, SCORE_TYPE
    elif name == "council_qa":
        from benchmarks.suites.council_qa import QUESTIONS, SCORE_TYPE
        return QUESTIONS, SCORE_TYPE
    else:
        raise ValueError(f"Unknown suite: {name}")


# ---------------------------------------------------------------------------
# Progress callback
# ---------------------------------------------------------------------------

def _progress(current: int, total: int, result: RunResult) -> None:
    """Print progress to stdout."""
    score_indicator = "✓" if result.final_answer.strip() else "·"
    elapsed = f"{result.wall_time_seconds:.1f}s"
    print(
        f"  [{current:3d}/{total}] {score_indicator} {result.question_id:<12s} "
        f"{elapsed:>8s}  ({result.total_output_chars:,} chars)",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------

async def run_benchmark(
    models: list[str],
    modes: list[str],
    suites: list[str],
    output_dir: Path,
    base_url: str,
    provider_kind: str,
    council_models: list[str],
    judge_model: str | None,
    ollama_think: bool,
    plan_model: str | None = None,
    execute_model: str | None = None,
    critique_model: str | None = None,
    limit: int | None = None,
) -> None:
    """Run the full benchmark matrix."""
    client = LocalLLMClient()
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, list[dict]] = {}
    run_start = time.perf_counter()
    timestamp = datetime.now(timezone.utc).isoformat()

    for suite_name in suites:
        questions, score_type = _load_suite(suite_name)
        print(f"\n{'='*60}")
        print(f"Suite: {suite_name} ({len(questions)} questions, scoring: {score_type})")
        print(f"{'='*60}")

        for model in models:
            for mode in modes:
                # Skip incompatible combinations
                if mode == "org" and suite_name in ("gsm8k_mini", "code_gen"):
                    # Org mode is only meaningful for fullstack suite
                    if suite_name != "fullstack":
                        continue
                if mode == "council" and suite_name == "fullstack":
                    # Council mode is not designed for org-style prompts
                    continue

                run_key = f"{suite_name}__{model}__{mode}"
                run_file = output_dir / f"{run_key}.json"
                print(f"\n▶ {model} | mode={mode}")

                # Apply limit if specified
                active_questions = questions
                if limit and limit > 0:
                    active_questions = questions[:limit]

                results = await run_suite(
                    client=client,
                    provider_kind=provider_kind,
                    base_url=base_url,
                    model=model,
                    mode=mode,
                    suite_name=suite_name,
                    questions=active_questions,
                    council_models=council_models if mode == "council" else None,
                    judge_model=judge_model if mode == "council" else None,
                    plan_model=plan_model,
                    execute_model=execute_model,
                    critique_model=critique_model,
                    ollama_think=ollama_think,
                    on_progress=_progress,
                    output_path=run_file,
                )

                # Re-load results to get the dict format with scores if needed
                # (Though run_suite returned them, we want to finalize them)
                result_dicts = [r.to_dict() for r in results]
                if score_type == "exact_numeric":
                    for rd in result_dicts:
                        rd["score"] = score_exact_numeric(
                            rd["expected_answer"], rd["final_answer"]
                        )
                elif score_type == "code_pass":
                    for rd, q in zip(result_dicts, questions):
                        test_code = q.get("test_code", "")
                        code_result = score_code_pass(rd["final_answer"], test_code)
                        rd["score"] = 1.0 if code_result["passed"] else 0.0
                        rd["metadata"]["code_error"] = code_result.get("error")
                else:
                    # Rubric: score is 0 (needs human review)
                    for rd in result_dicts:
                        rd["score"] = 0.0

                all_results[run_key] = result_dicts

                # Compute and print metrics
                metrics = compute_suite_metrics(result_dicts, score_type)
                if score_type in ("exact_numeric", "code_pass"):
                    acc = metrics.get("accuracy", 0)
                    correct = metrics.get("correct", 0)
                    total = metrics.get("total_questions", 0)
                    print(
                        f"  ✔ Accuracy: {acc:.1%} ({correct}/{total}) | "
                        f"Median time: {metrics.get('median_time_seconds', 0):.1f}s"
                    )
                else:
                    print(
                        f"  ✔ {metrics.get('total_questions', 0)} responses collected | "
                        f"Median time: {metrics.get('median_time_seconds', 0):.1f}s | "
                        f"(requires human rubric scoring)"
                    )

                # Save per-run results
                run_file = output_dir / f"{run_key}.json"
                run_file.write_text(
                    json.dumps(result_dicts, indent=2, ensure_ascii=False)
                )

    # Save aggregate summary
    summary = {
        "timestamp": timestamp,
        "models": models,
        "modes": modes,
        "suites": suites,
        "council_models": council_models,
        "total_time_seconds": round(time.perf_counter() - run_start, 2),
        "runs": {},
    }

    for run_key, result_dicts in all_results.items():
        parts = run_key.split("__")
        suite_name = parts[0]
        _, score_type = _load_suite(suite_name)
        summary["runs"][run_key] = compute_suite_metrics(result_dicts, score_type)

    summary_file = output_dir / "summary.json"
    summary_file.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    total_time = time.perf_counter() - run_start
    print(f"\n{'='*60}")
    print(f"Done! Total time: {total_time:.1f}s")
    print(f"Results saved to: {output_dir.resolve()}")
    print(f"Summary: {summary_file.resolve()}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="MultiMind AI Benchmark Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--models",
        default="qwen3.5:4b",
        help="Comma-separated list of models (default: qwen3.5:4b)",
    )
    parser.add_argument(
        "--modes",
        default="off,medium,hard",
        help="Comma-separated list of modes (default: off,medium,hard)",
    )
    parser.add_argument(
        "--suites",
        default="gsm8k_mini",
        help="Comma-separated list of suites (default: gsm8k_mini)",
    )
    parser.add_argument(
        "--output",
        default="benchmarks/results",
        help="Output directory for results (default: benchmarks/results)",
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:11434",
        help="Ollama base URL (default: http://127.0.0.1:11434)",
    )
    parser.add_argument(
        "--provider",
        default="ollama",
        choices=["ollama", "openai"],
        help="Provider kind (default: ollama)",
    )
    parser.add_argument(
        "--council-models",
        default="qwen3.5:4b,ministral-3:3b,qwen3-4b-instruct,llama3.2:3b",
        help="Comma-separated council advisor models",
    )
    parser.add_argument(
        "--judge-model",
        default=None,
        help="Judge model for council mode (default: same as main model)",
    )
    parser.add_argument(
        "--ollama-think",
        action="store_true",
        help="Enable Ollama native thinking mode",
    )
    parser.add_argument(
        "--plan-model",
        default=None,
        help="Planning model for thinking pipeline (overrides main model)",
    )
    parser.add_argument(
        "--execute-model",
        default=None,
        help="Execution model for thinking pipeline (overrides main model)",
    )
    parser.add_argument(
        "--critique-model",
        default=None,
        help="Critique model for thinking pipeline (overrides main model)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of questions per suite",
    )

    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",")]
    modes = [m.strip() for m in args.modes.split(",")]
    suites = [s.strip() for s in args.suites.split(",")]
    council_models = [m.strip() for m in args.council_models.split(",")]

    print("╔══════════════════════════════════════════════════════════╗")
    print("║           MultiMind AI — Benchmark Suite                ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"  Models:  {models}")
    print(f"  Modes:   {modes}")
    print(f"  Suites:  {suites}")
    print(f"  Council: {council_models}")
    print(f"  Output:  {args.output}")

    asyncio.run(
        run_benchmark(
            models=models,
            modes=modes,
            suites=suites,
            output_dir=Path(args.output),
            base_url=args.base_url,
            provider_kind=args.provider,
            council_models=council_models,
            judge_model=args.judge_model,
            plan_model=args.plan_model,
            execute_model=args.execute_model,
            critique_model=args.critique_model,
            ollama_think=args.ollama_think,
            limit=args.limit,
        )
    )


if __name__ == "__main__":
    main()

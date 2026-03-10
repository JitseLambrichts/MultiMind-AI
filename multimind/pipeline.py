from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from multimind.config import PIPELINE_STEPS
from multimind.llm_client import LocalLLMClient
from multimind.markdown_render import render_markdown_to_html


STEP_LABELS = {
    "plan": "Planning",
    "execute": "Executing",
    "critique": "Critiquing",
}

def _system_prompt_for_council_member() -> str:
    return (
        "You are an expert advisor on an AI Agent Council.\n"
        "Your task: provide the most accurate, rigorous, and comprehensive answer to the user's request.\n"
        "Instructions:\n"
        "1. DEEP REASONING: Think step-by-step. Break down the problem, identify core assumptions, and consider potential edge cases.\n"
        "2. EXPERT PERSPECTIVE: Bring deep technical expertise to bear. Highlight nuances, trade-offs, or alternative approaches.\n"
        "3. STRUCTURE: Organize your response logically. If providing code or steps, ensure they are complete and verifiable.\n"
        "4. DIRECTNESS: Be direct. Avoid introductory filler or pleasantries. Get straight to the analysis and solution."
    )

def _system_prompt_for_judge() -> str:
    return (
        "You are the Lead Synthesizer and Judge of an AI Agent Council.\n"
        "Your task: critically review the answers provided by several expert advisors and synthesize them into a single, definitive, and superior final answer.\n"
        "Instructions:\n"
        "1. CRITICAL ANALYSIS: Evaluate each advisor's response. Identify the strongest arguments, most accurate facts, and most elegant solutions.\n"
        "2. CONFLICT RESOLUTION: If advisors disagree, rigorously cross-examine their claims. Resolve contradictions using logic, established facts, and technical accuracy.\n"
        "3. SUPERIOR SYNTHESIS: Do not just copy-paste. Combine the best elements of all responses into a cohesive, comprehensive, and perfectly structured final answer that is better than any individual advisor's response.\n"
        "4. NO META-COMMENTARY: Output ONLY the final answer. Do not include summaries of the council's process, or phrases like 'Advisor 1 said'. Just provide the definitive solution."
    )

def _build_messages_for_council_member(user_message: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": _system_prompt_for_council_member()},
        {"role": "user", "content": user_message},
    ]

def _build_messages_for_judge(user_message: str, council_answers: list[str]) -> list[dict[str, str]]:
    answers_text = "\n\n---\n\n".join(
        f"Advisor {i+1}:\n{answer}" for i, answer in enumerate(council_answers)
    )
    return [
        {"role": "system", "content": _system_prompt_for_judge()},
        {
            "role": "user",
            "content": (
                f"User request:\n{user_message}\n\n"
                f"Council Answers:\n{answers_text}\n\n"
                "Provide the definitive final answer."
            ),
        },
    ]

def _system_prompt_for_plan() -> str:
    return (
        "You are the expert analytical planning stage in a multi-step reasoning pipeline.\n"
        "Your task: carefully research the request and map out a grounded, step-by-step strategy.\n"
        "Instructions:\n"
        "1. GROUNDED RESEARCH: Start by identifying the core requirements, necessary technologies, and architectural dependencies.\n"
        "2. LOGICAL BREAKDOWN: Use numbered pseudocode to represent execution steps. Group tasks by component or layer.\n"
        "3. EDGE CASES: Explicitly identify 3-5 likely edge cases (e.g., empty states, timeouts, auth failures) and integrate them into the plan.\n"
        "4. DESIGN AESTHETICS: If the task involves UI, plan for 'Rich Aesthetics'—vibrant palettes, modern typography, and dynamic micro-animations.\n"
        "5. VERIFICATION PLAN: Include specific criteria for success and a strategy for testing each step.\n"
        "6. DO NOT provide the final answer. Focus entirely on the technical roadmap and 'HOW' to solve it."
    )


def _system_prompt_for_execute(mode: str) -> str:
    # 'hard' mode emphasizes rigor, symbol tracing, and strict adherence to conventions
    if mode == "hard":
        return (
            "You are the expert execution stage in a reasoning pipeline.\n"
            "Your task: follow the provided plan with absolute technical rigor to produce a comprehensive solution.\n"
            "Instructions:\n"
            "1. PLAN ADHERENCE: Abide strictly by the provided technical roadmap. If complexity requires a design change, note it clearly.\n"
            "2. CONVENTION MIMICRY: Mirror the project's existing code style, naming conventions, and idiomatic patterns. NEVER assume a library is available—verify usage first.\n"
            "3. SYMBOL TRACING: Trace every symbol to its definition and usage to ensure complete consistency across the codebase.\n"
            "4. STEP-BY-STEP REASONING: Think through the implementation logic out loud to ensure all edge cases identified in the plan are handled.\n"
            "5. DEFENSIVE SECURITY: Apply security best practices—sanitize inputs, avoid logging PII, and protect sensitive credentials.\n"
            "6. FULL OUTPUT: Provide the complete, functional solution without gaps or placeholders."
        )

    # Standard mode focuses on high polish and direct utility
    return (
        "You are the expert execution stage in a reasoning pipeline. "
        "Use the provided plan to smoothly produce a final, highly polished, and directly useful user-facing answer. "
        "Mimic existing conventions and trace all logic dependencies to ensure accuracy. "
        "Think step-by-step but keep the final output clean, idiomatic, and ready for use."
    )


def _system_prompt_for_critique() -> str:
    return (
        "You are the expert critique and revision stage in a reasoning pipeline.\n"
        "Your task: rigorously audit the draft answer against the original request, the technical plan, and engineering standards.\n"
        "Instructions:\n"
        "1. DIFFERENTIAL REVIEW: Review all modifications for factual errors, weak logic, or failure to follow constraints.\n"
        "2. VERIFICATION CHECK: Confirm the solution meets the 'Verification Plan' from the planning stage and handles all identified edge cases.\n"
        "3. OMISSIONS & LOGIC: Actively look for missed references, inconsistent naming, or unoptimized logic.\n"
        "4. SUPERIOR SYNTHESIS: Consolidate all corrections into a single, superior final response that is significantly improved.\n"
        "5. NO META-COMMENTARY: Output ONLY the improved final answer. Do not include introductory filler, summaries of changes, or explanations."
    )



def _build_messages_for_plan(user_message: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": _system_prompt_for_plan()},
        {"role": "user", "content": user_message},
    ]


def _build_messages_for_execute(user_message: str, plan: str, mode: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": _system_prompt_for_execute(mode)},
        {
            "role": "user",
            "content": (
                f"User request:\n{user_message}\n\n"
                f"Plan:\n{plan or 'No plan was generated.'}\n\n"
                "Write the best possible answer."
            ),
        },
    ]


def _build_messages_for_critique(user_message: str, plan: str, draft: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": _system_prompt_for_critique()},
        {
            "role": "user",
            "content": (
                f"User request:\n{user_message}\n\n"
                f"Plan:\n{plan or 'No plan was generated.'}\n\n"
                f"Draft answer:\n{draft or 'No draft answer was generated.'}\n\n"
                "Provide the corrected final answer."
            ),
        },
    ]


def _pipeline_for_mode(mode: str) -> tuple[str, ...]:
    if mode == "off":
        return ("execute",)
    if mode == "medium":
        return ("plan", "execute")
    return PIPELINE_STEPS


async def _stream_council_member(
    client: LocalLLMClient,
    provider_kind: str,
    base_url: str,
    model: str,
    index: int,
    user_message: str,
    queue: asyncio.Queue,
) -> None:
    step_id = f"council-{index}"
    try:
        await queue.put(
            {
                "type": "step-start",
                "step": step_id,
                "label": f"Advisor {index + 1}",
                "model": model,
                "thought": True,
            }
        )

        messages = _build_messages_for_council_member(user_message)
        buffer: list[str] = []
        partial_content = ""

        async for token in client.stream_chat(
            provider_kind=provider_kind,
            base_url=base_url,
            model=model,
            messages=messages,
        ):
            buffer.append(token)
            partial_content += token
            await queue.put(
                {
                    "type": "step-delta",
                    "step": step_id,
                    "delta": token,
                    "content": partial_content,
                    "html": render_markdown_to_html(partial_content),
                }
            )

        final_content = "".join(buffer).strip()
        await queue.put(
            {
                "type": "step-complete",
                "step": step_id,
                "content": final_content,
                "html": render_markdown_to_html(final_content),
            }
        )
        await queue.put({"type": "internal-member-done", "index": index, "content": final_content})

    except Exception as exc:
        await queue.put(
            {
                "type": "step-complete",
                "step": step_id,
                "content": f"Failed: {exc}",
                "html": f"<p>Failed: {exc}</p>",
            }
        )
        await queue.put({"type": "internal-member-done", "index": index, "content": ""})

async def run_council_pipeline(
    *,
    client: LocalLLMClient,
    provider_kind: str,
    base_url: str,
    council_models: list[str],
    judge_model: str,
    user_message: str,
) -> AsyncIterator[dict]:
    steps = [f"council-{i}" for i in range(len(council_models))] + ["judge"]
    yield {"type": "run-start", "mode": "council", "steps": steps}

    queue: asyncio.Queue = asyncio.Queue()
    tasks = [
        asyncio.create_task(
            _stream_council_member(
                client, provider_kind, base_url, model, i, user_message, queue
            )
        )
        for i, model in enumerate(council_models)
    ]

    council_answers: list[str] = [""] * len(council_models)
    completed_members = 0

    while completed_members < len(council_models):
        event = await queue.get()
        if event["type"] == "internal-member-done":
            council_answers[event["index"]] = event["content"]
            completed_members += 1
        else:
            yield event

    await asyncio.gather(*tasks, return_exceptions=True)

    # Judge execution
    yield {
        "type": "step-start",
        "step": "judge",
        "label": "Judge",
        "model": judge_model,
        "thought": False,
    }
    yield {
        "type": "answer-start",
        "step": "judge",
        "label": "Judge",
        "model": judge_model,
    }

    messages = _build_messages_for_judge(user_message, [ans for ans in council_answers if ans])
    buffer: list[str] = []
    partial_content = ""

    try:
        async for token in client.stream_chat(
            provider_kind=provider_kind,
            base_url=base_url,
            model=judge_model,
            messages=messages,
        ):
            buffer.append(token)
            partial_content += token
            yield {
                "type": "step-delta",
                "step": "judge",
                "delta": token,
                "content": partial_content,
                "html": render_markdown_to_html(partial_content),
            }
            yield {
                "type": "answer-delta",
                "step": "judge",
                "delta": token,
                "content": partial_content,
                "html": render_markdown_to_html(partial_content),
            }

        final_content = "".join(buffer).strip()
        yield {
            "type": "step-complete",
            "step": "judge",
            "content": final_content,
            "html": render_markdown_to_html(final_content),
        }
        yield {
            "type": "answer-complete",
            "step": "judge",
            "content": final_content,
            "html": render_markdown_to_html(final_content),
        }
    except Exception as exc:
        yield {
            "type": "step-complete",
            "step": "judge",
            "content": f"Judge failed: {exc}",
            "html": f"<p>Judge failed: {exc}</p>",
        }
        yield {
            "type": "answer-complete",
            "step": "judge",
            "content": f"Judge failed: {exc}",
            "html": f"<p>Judge failed: {exc}</p>",
        }

    outputs = {"final": final_content if "final_content" in locals() else ""}
    yield {"type": "run-complete", "outputs": outputs}

async def run_pipeline(
    *,
    client: LocalLLMClient,
    provider_kind: str,
    base_url: str,
    model_map: dict[str, str],
    user_message: str,
    mode: str,
) -> AsyncIterator[dict]:
    steps = _pipeline_for_mode(mode)
    outputs: dict[str, str] = {}

    yield {"type": "run-start", "mode": mode, "steps": list(steps)}

    for step in steps:
        model = model_map.get(step) or model_map.get("execute") or ""
        if not model:
            raise ValueError(f"No model configured for step '{step}'.")

        yield {
            "type": "step-start",
            "step": step,
            "label": STEP_LABELS[step],
            "model": model,
            "thought": step != "execute" or mode == "hard",
        }

        if step == "execute" and mode != "hard":
            yield {"type": "answer-start", "step": step, "label": STEP_LABELS[step], "model": model}
        if step == "critique":
            yield {"type": "answer-start", "step": step, "label": STEP_LABELS[step], "model": model}

        if step == "plan":
            messages = _build_messages_for_plan(user_message)
        elif step == "execute":
            messages = _build_messages_for_execute(user_message, outputs.get("plan", ""), mode)
        else:
            messages = _build_messages_for_critique(
                user_message,
                outputs.get("plan", ""),
                outputs.get("execute", ""),
            )

        buffer: list[str] = []
        partial_content = ""
        async for token in client.stream_chat(
            provider_kind=provider_kind,
            base_url=base_url,
            model=model,
            messages=messages,
        ):
            buffer.append(token)
            partial_content += token
            yield {
                "type": "step-delta",
                "step": step,
                "delta": token,
                "content": partial_content,
                "html": render_markdown_to_html(partial_content),
            }

            if (step == "execute" and mode != "hard") or step == "critique":
                yield {
                    "type": "answer-delta",
                    "step": step,
                    "delta": token,
                    "content": partial_content,
                    "html": render_markdown_to_html(partial_content),
                }

        outputs[step] = "".join(buffer).strip()
        yield {
            "type": "step-complete",
            "step": step,
            "content": outputs[step],
            "html": render_markdown_to_html(outputs[step]),
        }

        if (step == "execute" and mode != "hard") or step == "critique":
            yield {
                "type": "answer-complete",
                "step": step,
                "content": outputs[step],
                "html": render_markdown_to_html(outputs[step]),
            }

    if mode == "off":
        outputs["final"] = outputs.get("execute", "")
    elif mode == "medium":
        outputs["final"] = outputs.get("execute", "")
    else:
        outputs["final"] = outputs.get("critique", "")

    yield {"type": "run-complete", "outputs": outputs}
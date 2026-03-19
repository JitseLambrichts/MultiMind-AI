from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator

from multimind.llm_client import LocalLLMClient
from multimind.markdown_render import render_markdown_to_html
from multimind.org_roles import (
    VALID_DEPARTMENTS,
    get_department_employees,
    get_department_roster_summary,
)

COMPANY_NAME = "MultiMind Corp"


def _system_prompt_ceo() -> str:
    roster = get_department_roster_summary()
    return (
        "You are the CEO of a multi-department AI organization. "
        "Your job is not to do the work — it is to ensure the right departments get the right sub-tasks, "
        "defined clearly enough that each can execute independently without coming back to you.\n\n"

        "How you work:\n"
        "1. Identify the true core of the request. Strip out noise.\n"
        "2. Map the request to departments. Only involve a department if its output is genuinely needed "
        "for the final deliverable — not because it could theoretically contribute.\n"
        "3. Write each sub-task as a complete, self-contained brief. "
        "The department must be able to act on it without reading the original request.\n"
        "4. If a department's work depends on another's output, note that dependency in the task description.\n\n"

        "Available departments and their specialists:\n"
        f"{roster}\n\n"

        "Output rules — these are absolute:\n"
        "- Respond with ONLY a valid JSON array. No preamble, no explanation, no trailing text.\n"
        "- Each object has exactly two string keys: \"department\" and \"task\".\n"
        f"- Valid department names: {', '.join(sorted(VALID_DEPARTMENTS))}.\n"
        "- 2–5 departments maximum. More is not better — unfocused delegation produces incoherent results.\n\n"

        "Example:\n"
        '[{"department": "Engineering", "task": "Build a REST API with endpoints for user registration, '
        'login (JWT), and profile retrieval. Use Node.js + Express. Auth must handle token expiry."}, '
        '{"department": "Design", "task": "Create a mobile-first landing page mockup. '
        'Target audience: B2B SaaS buyers. Emphasize trust signals and a clear CTA above the fold."}]'
    )


def _system_prompt_ceo_scope_tasks(department: str, roles: list[str]) -> str:
    role_list = ", ".join(roles)
    return (
        f"You are the CEO. You previously assigned a sub-task to the {department} department. "
        f"That department has these specialists: {role_list}.\n\n"

        "Your job now is to split the sub-task into individual assignments — one per specialist who is "
        "genuinely needed. Not every specialist must be used. Only assign work that requires "
        "that specific role's expertise.\n\n"

        "Each assignment must be fully self-contained: include all context the employee needs, "
        "because they will not see the original request.\n\n"

        "Output rules — these are absolute:\n"
        "- Respond with ONLY a valid JSON array. No preamble, no explanation, no trailing text.\n"
        "- Each object has exactly two string keys: \"role\" and \"task\".\n"
        f"- Valid role names: {role_list}.\n"
        f"- 1–{len(roles)} employees maximum. If the task can be done by one person, assign one.\n\n"

        "Example:\n"
        '[{"role": "Backend Developer", "task": "Implement JWT-based user authentication. '
        'Endpoints: POST /auth/register, POST /auth/login, GET /auth/me. '
        'Handle token expiry with a 401 response and refresh token flow."}, '
        '{"role": "QA Engineer", "task": "Write integration tests for the auth endpoints listed above. '
        'Cover: valid login, wrong password, expired token, and missing header cases."}]'
    )


def _system_prompt_ceo_synthesize() -> str:
    return (
        "You are the CEO, and your departments have delivered their work. "
        "Your final responsibility is to synthesize these outputs into a single, authoritative response "
        "to the user's original request — one that reads as if it came from one expert, not a committee.\n\n"

        "How you work:\n"
        "1. Read all department outputs before writing anything.\n"
        "2. Identify the narrative spine: what is the user actually trying to accomplish, "
        "and how do these outputs collectively answer that?\n"
        "3. Merge overlapping content once. If two departments covered the same ground, "
        "keep the more complete version and discard the redundancy.\n"
        "4. Preserve every critical detail — technical specs, constraints, recommendations — "
        "but cut ceremonial language, department headers, and role attributions.\n"
        "5. Sequence the content so it flows logically for the user, not in the order departments were invoked.\n\n"

        "Output standards: The user should receive a cohesive, well-structured final answer. "
        "No meta-commentary about the process, no 'the Engineering team said...', no synthesis summaries. "
        "Just the best possible answer to what they asked."
    )


def _parse_json_array(text: str) -> list[dict]:
    """Extract a JSON array from LLM output, handling markdown fences and preamble."""
    # Try to find JSON array in the text
    # First, try direct parse
    text = text.strip()
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass

    # Try to extract from markdown code fence
    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if fence_match:
        try:
            result = json.loads(fence_match.group(1).strip())
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

    # Try to find the first [ ... ] block
    bracket_match = re.search(r"\[.*\]", text, re.DOTALL)
    if bracket_match:
        try:
            result = json.loads(bracket_match.group(0))
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

    return []


def _slug(text: str) -> str:
    """Create a URL-safe slug from text."""
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


async def run_org_pipeline(
    *,
    client: LocalLLMClient,
    provider_kind: str,
    base_url: str,
    model: str,
    ollama_think: bool,
    user_message: str,
) -> AsyncIterator[dict]:
    """Run the multi-agent org chart pipeline with pre-trained employees."""

    # ── Phase 1: CEO decomposes the task into departments ─────────
    ceo_id = "org-ceo"
    yield {
        "type": "org-run-start",
        "mode": "org",
        "root": ceo_id,
    }

    yield {
        "type": "org-node-start",
        "node_id": ceo_id,
        "parent_id": None,
        "role": "CEO",
        "slug": "chief-executive-officer",
        "department": None,
        "model": model,
        "reports": 0,
    }

    # Stream CEO decomposition
    ceo_messages = [
        {"role": "system", "content": _system_prompt_ceo()},
        {"role": "user", "content": user_message},
    ]

    ceo_buffer: list[str] = []
    ceo_partial = ""

    async for token in client.stream_chat(
        provider_kind=provider_kind,
        base_url=base_url,
        model=model,
        messages=ceo_messages,
        ollama_think=ollama_think,
    ):
        ceo_buffer.append(token)
        ceo_partial += token
        yield {
            "type": "org-node-delta",
            "node_id": ceo_id,
            "delta": token,
            "content": ceo_partial,
            "html": render_markdown_to_html(ceo_partial),
        }

    ceo_output = "".join(ceo_buffer).strip()

    # Parse departments from CEO output
    departments = _parse_json_array(ceo_output)
    if not departments:
        # Fallback: treat entire output as a single task
        departments = [{"department": "Engineering", "task": user_message}]

    yield {
        "type": "org-node-complete",
        "node_id": ceo_id,
        "content": ceo_output,
        "html": render_markdown_to_html(ceo_output),
        "reports": len(departments),
    }

    # ── Phase 2 & 3: For each department, scope tasks then execute ─
    all_department_results: list[dict] = []

    for dept_idx, dept_info in enumerate(departments):
        dept_name = dept_info.get("department", f"Department {dept_idx + 1}")
        dept_task = dept_info.get("task", "")
        dept_id = f"org-dept-{_slug(dept_name)}-{dept_idx}"

        # Look up the pre-trained employees for this department
        roster = get_department_employees(dept_name)
        role_names = [e["role"] for e in roster]

        # Emit a synthetic department group node (preserves 3-level visual hierarchy)
        yield {
            "type": "org-node-start",
            "node_id": dept_id,
            "parent_id": ceo_id,
            "role": f"{dept_name} Department",
            "slug": _slug(dept_name),
            "department": dept_name,
            "model": model,
            "reports": 0,
        }

        # Phase 2: CEO scopes tasks for this department's specialists
        scope_messages = [
            {"role": "system", "content": _system_prompt_ceo_scope_tasks(dept_name, role_names)},
            {"role": "user", "content": dept_task},
        ]

        scope_buffer: list[str] = []
        scope_partial = ""

        async for token in client.stream_chat(
            provider_kind=provider_kind,
            base_url=base_url,
            model=model,
            messages=scope_messages,
            ollama_think=ollama_think,
        ):
            scope_buffer.append(token)
            scope_partial += token
            yield {
                "type": "org-node-delta",
                "node_id": dept_id,
                "delta": token,
                "content": scope_partial,
                "html": render_markdown_to_html(scope_partial),
            }

        scope_output = "".join(scope_buffer).strip()
        assignments = _parse_json_array(scope_output)
        if not assignments:
            # Fallback: assign the full task to the first employee
            assignments = [{"role": roster[0]["role"], "task": dept_task}]

        yield {
            "type": "org-node-complete",
            "node_id": dept_id,
            "content": scope_output,
            "html": render_markdown_to_html(scope_output),
            "reports": len(assignments),
        }

        # Phase 3: Employees execute their assignments using pre-trained prompts
        employee_results: list[dict] = []

        # Build a lookup from role name → system_prompt
        role_prompt_map = {e["role"]: e["system_prompt"] for e in roster}

        for emp_idx, emp_info in enumerate(assignments):
            emp_role = emp_info.get("role", roster[0]["role"])
            emp_task = emp_info.get("task", "")
            emp_id = f"org-emp-{_slug(dept_name)}-{_slug(emp_role)}-{emp_idx}"

            # Use pre-trained system prompt, or fall back to a generic one
            emp_system_prompt = role_prompt_map.get(
                emp_role,
                (
                    f"You are a {emp_role} in the {dept_name} department. "
                    "You are a specialist. Execute the assignment with full depth — "
                    "no placeholders, no deferred work. Lead with the deliverable."
                ),
            )

            yield {
                "type": "org-node-start",
                "node_id": emp_id,
                "parent_id": dept_id,
                "role": emp_role,
                "slug": _slug(emp_role),
                "department": dept_name,
                "model": model,
                "reports": 0,
            }

            emp_messages = [
                {"role": "system", "content": emp_system_prompt},
                {"role": "user", "content": emp_task},
            ]

            emp_buffer: list[str] = []
            emp_partial = ""

            async for token in client.stream_chat(
                provider_kind=provider_kind,
                base_url=base_url,
                model=model,
                messages=emp_messages,
                ollama_think=ollama_think,
            ):
                emp_buffer.append(token)
                emp_partial += token
                yield {
                    "type": "org-node-delta",
                    "node_id": emp_id,
                    "delta": token,
                    "content": emp_partial,
                    "html": render_markdown_to_html(emp_partial),
                }

            emp_output = "".join(emp_buffer).strip()
            yield {
                "type": "org-node-complete",
                "node_id": emp_id,
                "content": emp_output,
                "html": render_markdown_to_html(emp_output),
                "reports": 0,
            }

            employee_results.append({
                "role": emp_role,
                "task": emp_task,
                "result": emp_output,
            })

        all_department_results.append({
            "department": dept_name,
            "task": dept_task,
            "employees": employee_results,
        })

    # ── Phase 4: CEO synthesizes the final answer ────────────────────
    synthesis_sections = []
    for dept_result in all_department_results:
        section = f"## {dept_result['department']}\n"
        for emp in dept_result["employees"]:
            section += f"### {emp['role']}\n{emp['result']}\n\n"
        synthesis_sections.append(section)

    synthesis_input = (
        f"Original user request:\n{user_message}\n\n"
        f"Department outputs:\n\n{'---\n\n'.join(synthesis_sections)}"
    )

    synthesis_messages = [
        {"role": "system", "content": _system_prompt_ceo_synthesize()},
        {"role": "user", "content": synthesis_input},
    ]

    yield {"type": "answer-start", "step": "org-synthesis", "label": "CEO Synthesis", "model": model}

    synth_buffer: list[str] = []
    synth_partial = ""

    async for token in client.stream_chat(
        provider_kind=provider_kind,
        base_url=base_url,
        model=model,
        messages=synthesis_messages,
        ollama_think=ollama_think,
    ):
        synth_buffer.append(token)
        synth_partial += token
        yield {
            "type": "answer-delta",
            "step": "org-synthesis",
            "delta": token,
            "content": synth_partial,
            "html": render_markdown_to_html(synth_partial),
        }

    final_content = "".join(synth_buffer).strip()
    yield {
        "type": "answer-complete",
        "step": "org-synthesis",
        "content": final_content,
        "html": render_markdown_to_html(final_content),
    }

    yield {"type": "run-complete", "outputs": {"final": final_content}}

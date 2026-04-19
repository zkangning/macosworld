import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import OpenAI


DEFAULT_API_KEY = "MAASace45968cdbf4afeb71d07ecef846c94"
DEFAULT_BASE_URL = "https://maas.devops.xiaohongshu.com/v1"
DEFAULT_MODEL = "qwen3.5-397b-a17b"

REPO_ROOT = Path(__file__).resolve().parents[1]
SKILL_ROOT = REPO_ROOT / "mm_skills_download" / "mac_4_18_skills"
TASK_SOURCES = [
    REPO_ROOT / "tasks" / "baseline5_snapshot_used_en",
    REPO_ROOT / "tasks" / "baseline5_snapshot_usedApps_en",
]
DOMAINS = [
    "file_management",
    "media",
    "productivity",
    "sys_and_interface",
    "sys_apps",
]
DEFAULT_OUTPUT_PATH = SKILL_ROOT / "qwen35_top5_skill_mapping.json"


@dataclass
class SkillRecord:
    skill_id: str
    display_name: str
    description: str
    skill_dir: Path


@dataclass
class TaskRecord:
    task_id: str
    domain: str
    task_text: str
    source_file: Path
    source_package: str
    app_hints: list[str]
    pre_command_hint: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Map baseline5 macOSWorld tasks to top-5 relevant skills with Qwen3.5."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api-key", default=os.getenv("QWEN_TEST_API_KEY", DEFAULT_API_KEY))
    parser.add_argument("--base-url", default=os.getenv("QWEN_TEST_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument(
        "--skill-root",
        type=Path,
        default=SKILL_ROOT,
        help="Root directory that contains the 5 domain skill folders.",
    )
    parser.add_argument(
        "--task-source",
        action="append",
        type=Path,
        default=None,
        help="Task directory to include. May be supplied multiple times.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="JSON file to write incrementally.",
    )
    parser.add_argument(
        "--domain",
        action="append",
        choices=DOMAINS,
        default=None,
        help="Restrict processing to one or more domains. May be supplied multiple times.",
    )
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument(
        "--thinking-mode",
        choices=["auto", "on", "off"],
        default="on",
        help="auto omits the flag, on sends enable_thinking=true, off sends enable_thinking=false",
    )
    parser.add_argument("--limit", type=int, default=0, help="Process only the first N tasks for debugging.")
    parser.add_argument("--retry", type=int, default=3, help="Retries per task on parse/validation failures.")
    parser.add_argument("--sleep-seconds", type=float, default=0.0, help="Optional delay between API calls.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Ignore existing output JSON and recompute all tasks.",
    )
    return parser.parse_args()


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def extract_front_matter(skill_md: Path) -> dict[str, str]:
    text = skill_md.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        return {}

    parts = text.split("---\n", 2)
    if len(parts) < 3:
        return {}

    front_matter = {}
    for raw_line in parts[1].splitlines():
        line = raw_line.strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        front_matter[key.strip()] = value.strip().strip('"').strip("'")
    return front_matter


def load_skill_catalog(skill_root: Path) -> dict[str, list[SkillRecord]]:
    catalog: dict[str, list[SkillRecord]] = {}
    for domain in DOMAINS:
        domain_dir = skill_root / domain
        skills: list[SkillRecord] = []
        for skill_dir in sorted(p for p in domain_dir.iterdir() if p.is_dir()):
            skill_md = skill_dir / "SKILL.md"
            front_matter = extract_front_matter(skill_md) if skill_md.exists() else {}
            display_name = front_matter.get("name") or skill_dir.name.replace("_", " ")
            description = front_matter.get("description") or display_name
            skills.append(
                SkillRecord(
                    skill_id=skill_dir.name,
                    display_name=normalize_text(display_name),
                    description=normalize_text(description),
                    skill_dir=skill_dir,
                )
            )
        catalog[domain] = skills
    return catalog


def build_task_domain_lookup(repo_root: Path) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for domain in DOMAINS:
        for task_path in sorted((repo_root / "tasks" / domain).glob("*.json")):
            task_id = task_path.stem
            previous = lookup.get(task_id)
            if previous and previous != domain:
                raise ValueError(f"Task id {task_id} is duplicated across domains: {previous}, {domain}")
            lookup[task_id] = domain
    return lookup


def get_english_field(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("en", "")).strip()
    if value is None:
        return ""
    return str(value).strip()


def extract_app_hints(pre_command_text: str) -> list[str]:
    matches = re.findall(r'tell application "([^"]+)"', pre_command_text)
    seen = set()
    app_hints = []
    for match in matches:
        app_name = match.strip()
        if app_name and app_name not in seen:
            seen.add(app_name)
            app_hints.append(app_name)
    return app_hints


def shorten_pre_command(pre_command_text: str, max_len: int = 280) -> str:
    compact = normalize_text(pre_command_text)
    if len(compact) <= max_len:
        return compact
    return compact[: max_len - 3] + "..."


def load_tasks(task_sources: list[Path], domain_lookup: dict[str, str]) -> list[TaskRecord]:
    tasks: list[TaskRecord] = []
    seen_ids = set()

    for task_source in task_sources:
        for task_path in sorted(task_source.glob("*.json")):
            payload = read_json(task_path)
            task_id = payload["id"]
            if task_id in seen_ids:
                continue
            seen_ids.add(task_id)

            domain = domain_lookup.get(task_id)
            if not domain:
                raise KeyError(f"Could not resolve domain for task id {task_id} from {task_path}")

            task_text = get_english_field(payload.get("task"))
            pre_command_text = get_english_field(payload.get("pre_command"))
            tasks.append(
                TaskRecord(
                    task_id=task_id,
                    domain=domain,
                    task_text=normalize_text(task_text),
                    source_file=task_path,
                    source_package=task_source.name,
                    app_hints=extract_app_hints(pre_command_text),
                    pre_command_hint=shorten_pre_command(pre_command_text),
                )
            )

    return tasks


def build_skill_block(skills: list[SkillRecord]) -> str:
    lines = []
    for idx, skill in enumerate(skills, start=1):
        lines.append(
            f"{idx}. skill_id: {skill.skill_id}\n"
            f"   name: {skill.display_name}\n"
            f"   description: {skill.description}"
        )
    return "\n".join(lines)


def build_messages(task: TaskRecord, skills: list[SkillRecord]) -> list[dict[str, str]]:
    system_prompt = (
        "You are a precise macOSWorld skill router. "
        "Given one benchmark task and all candidate skills from the already known domain, "
        "select the 5 most relevant skills. "
        "Keep your reasoning concise and reserve enough tokens for the final answer. "
        "You must only output valid JSON."
    )

    app_hint_text = ", ".join(task.app_hints) if task.app_hints else "none"
    user_prompt = (
        "Choose the top 5 most relevant skills for this benchmark task.\n\n"
        f"Known domain: {task.domain}\n"
        f"Task ID: {task.task_id}\n"
        f"Task source package: {task.source_package}\n"
        f"Task text: {task.task_text}\n"
        f"App hints from setup: {app_hint_text}\n"
        f"Prep command hint: {task.pre_command_hint or 'none'}\n\n"
        "Selection rules:\n"
        "- Prefer skills that match the app, surface, and concrete edit/navigation intent.\n"
        "- Prefer skills whose descriptions align with the requested end state.\n"
        "- Return exactly 5 unique skills.\n"
        "- Use exact skill_id strings from the candidate list.\n\n"
        "Candidate skills:\n"
        f"{build_skill_block(skills)}\n\n"
        "Return JSON only with this schema:\n"
        '{\n'
        '  "top_5_skills": ["skill_id_1", "skill_id_2", "skill_id_3", "skill_id_4", "skill_id_5"],\n'
        '  "brief_reason": "one short sentence"\n'
        '}'
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def extract_message_content(message: Any) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                parts.append(item["text"])
            elif hasattr(item, "text"):
                parts.append(getattr(item, "text"))
        return "\n".join(part for part in parts if part).strip()
    return str(content or "").strip()


def extract_first_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if not stripped:
        raise ValueError("Model returned empty content.")

    code_block_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, flags=re.DOTALL)
    if code_block_match:
        return json.loads(code_block_match.group(1))

    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"Could not find a JSON object in model content: {stripped[:500]}")
    return json.loads(stripped[start : end + 1])


def normalize_skill_selection(raw_skills: Any, valid_skill_ids: list[str]) -> list[str]:
    valid_set = set(valid_skill_ids)
    normalized: list[str] = []
    for item in raw_skills or []:
        skill_id = str(item).strip()
        if skill_id in valid_set and skill_id not in normalized:
            normalized.append(skill_id)
        if len(normalized) == 5:
            break
    return normalized


def heuristic_fill(task: TaskRecord, skills: list[SkillRecord], selected: list[str]) -> list[str]:
    selected_set = set(selected)
    task_terms = set(re.findall(r"[a-z0-9]+", task.task_text.lower()))
    task_terms.update(term.lower() for term in task.app_hints)
    task_terms.update(re.findall(r"[a-z0-9]+", task.pre_command_hint.lower()))

    scored = []
    for idx, skill in enumerate(skills):
        if skill.skill_id in selected_set:
            continue
        haystack_terms = set(re.findall(r"[a-z0-9]+", f"{skill.display_name} {skill.description} {skill.skill_id}".lower()))
        overlap = len(task_terms & haystack_terms)
        app_bonus = 3 if any(app.lower() in haystack_terms for app in task.app_hints) else 0
        scored.append((overlap + app_bonus, -idx, skill.skill_id))

    scored.sort(reverse=True)
    for _, _, skill_id in scored:
        if skill_id not in selected_set:
            selected.append(skill_id)
            selected_set.add(skill_id)
        if len(selected) == 5:
            break
    return selected[:5]


def query_top5_skills(
    client: OpenAI,
    task: TaskRecord,
    skills: list[SkillRecord],
    model: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    thinking_mode: str,
    retries: int,
) -> tuple[list[str], dict[str, Any]]:
    valid_skill_ids = [skill.skill_id for skill in skills]
    messages = build_messages(task, skills)
    last_error = None
    last_response_payload: dict[str, Any] | None = None

    for attempt in range(1, retries + 1):
        attempt_max_tokens = min(max_tokens * (2 ** (attempt - 1)), 12288)
        request_kwargs = {
            "model": model,
            "messages": messages,
            "stream": False,
            "max_tokens": attempt_max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
        if thinking_mode == "on":
            request_kwargs["extra_body"] = {"enable_thinking": True}
        elif thinking_mode == "off":
            request_kwargs["extra_body"] = {"enable_thinking": False}

        response = client.chat.completions.create(**request_kwargs)
        message = response.choices[0].message
        content = extract_message_content(message)
        usage = response.usage.model_dump() if getattr(response, "usage", None) else {}
        last_response_payload = {
            "content": content,
            "usage": usage,
            "finish_reason": response.choices[0].finish_reason,
            "reasoning_content": getattr(message, "reasoning_content", None),
        }

        try:
            parsed = extract_first_json_object(content)
            normalized = normalize_skill_selection(parsed.get("top_5_skills"), valid_skill_ids)
            normalized = heuristic_fill(task, skills, normalized)
            if len(normalized) != 5:
                raise ValueError(f"Expected 5 skills after normalization, got {len(normalized)}")
            return normalized, last_response_payload
        except Exception as exc:
            last_error = exc
            messages.append({"role": "assistant", "content": content})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Your previous answer was invalid for routing.\n"
                        f"Validation error: {exc}\n"
                        "Reply again with JSON only, using exactly 5 unique skill_id values from the candidate list. "
                        "Keep the reasoning short and make sure the final JSON is present."
                    ),
                }
            )

    raise RuntimeError(
        f"Failed to obtain a valid top-5 skill list for task {task.task_id} after {retries} attempts: {last_error}\n"
        f"Last response: {json.dumps(last_response_payload, ensure_ascii=False)[:1200]}"
    )


def load_existing_output(output_path: Path) -> dict[str, Any]:
    if not output_path.exists():
        return {}
    return read_json(output_path)


def init_output_document(existing: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if existing and not args.overwrite:
        for domain in DOMAINS:
            existing.setdefault(domain, {})
        existing.setdefault(
            "_meta",
            {
                "model": args.model,
                "thinking_mode": args.thinking_mode,
            },
        )
        return existing

    return {
        "_meta": {
            "model": args.model,
            "base_url": args.base_url,
            "thinking_mode": args.thinking_mode,
            "task_sources": [str(path) for path in (args.task_source or TASK_SOURCES)],
            "skill_root": str(args.skill_root),
        },
        **{domain: {} for domain in DOMAINS},
    }


def save_output(output_path: Path, payload: dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    task_sources = args.task_source or TASK_SOURCES

    domain_lookup = build_task_domain_lookup(REPO_ROOT)
    skill_catalog = load_skill_catalog(args.skill_root)
    tasks = load_tasks(task_sources, domain_lookup)
    if args.domain:
        allowed_domains = set(args.domain)
        tasks = [task for task in tasks if task.domain in allowed_domains]
    if args.limit > 0:
        tasks = tasks[: args.limit]

    existing = load_existing_output(args.output)
    output_doc = init_output_document(existing, args)

    client = OpenAI(api_key=args.api_key, base_url=args.base_url)

    total = len(tasks)
    processed = 0
    skipped = 0

    for idx, task in enumerate(tasks, start=1):
        domain_bucket = output_doc.setdefault(task.domain, {})
        if not args.overwrite and task.task_id in domain_bucket:
            skipped += 1
            print(f"[{idx}/{total}] skip {task.task_id} ({task.domain})", flush=True)
            continue

        print(f"[{idx}/{total}] route {task.task_id} ({task.domain})", flush=True)
        top_5_skills, debug_payload = query_top5_skills(
            client=client,
            task=task,
            skills=skill_catalog[task.domain],
            model=args.model,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            thinking_mode=args.thinking_mode,
            retries=args.retry,
        )

        domain_bucket[task.task_id] = top_5_skills
        output_doc["_meta"]["last_written_task_id"] = task.task_id
        output_doc["_meta"]["last_written_domain"] = task.domain
        output_doc["_meta"]["last_model_usage"] = debug_payload.get("usage", {})
        save_output(args.output, output_doc)

        processed += 1
        print(f"  -> {top_5_skills}", flush=True)
        if args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)

    output_doc["_meta"]["processed_count"] = processed
    output_doc["_meta"]["skipped_count"] = skipped
    output_doc["_meta"]["total_tasks_seen"] = total
    save_output(args.output, output_doc)
    print(f"Saved mapping to {args.output}", flush=True)


if __name__ == "__main__":
    main()

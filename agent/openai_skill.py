import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from agent.llm_utils import pil_to_b64
from agent.openai import GPT_SYSTEM_PROMPT, OpenAI_General_Agent
from agent.skill_loader import SkillLoader, SkillStateSelection
from utils.log import print_message
from utils.timeout import timeout


logger = logging.getLogger("macosworld.openai_skill")

MM_BRANCH_MODE = "mm_branch"

MAX_SKILL_CONSULTS_PER_SKILL = 2
MAX_MAIN_RESPONSE_ROUNDS = 6
MAX_STATE_VIEW_SELECTION_ROUNDS = 4
MAX_PLANNER_ROUNDS = 4
MAX_STAGE1_SELECTED_STATES = 3
MAX_STAGE1_SELECTED_VIEWS = 6
MAIN_STATE_CARD_PREVIEW_LIMIT = 3
ACTIVE_PLANNER_MEMO_TTL_STEPS = 5
MAX_HISTORY_RECORDS = 32


OPENAI_SKILL_MAIN_SYSTEM_PROMPT = (
    GPT_SYSTEM_PROMPT
    + "\n\n"
    + f"""
Additional skill rule:
- You may optionally request exactly one skill by returning `LOAD_SKILL("EXACT_SKILL_NAME")` in a fenced plaintext block.
- Only request a skill if the current screenshot, recent screenshot-action history, and the skill previews suggest that extra procedural guidance would help with the next immediate action.
- If you request a skill, do not output any other action in the same round.
- Skills are references only. Do not copy coordinates from them.
- After a skill branch returns planner notes, use the CURRENT screenshot as the source of truth for the real next action.
- Each skill may be consulted at most {MAX_SKILL_CONSULTS_PER_SKILL} times in the same trajectory.
- When clicking a target, emit `move_to x y` before the click action unless the cursor is already clearly positioned.
- In high-confidence simple UI states, it is acceptable to emit a compact 2-4 action block if the actions are directly grounded by the current screenshot.
""".strip()
)


OPENAI_SKILL_STAGE1_SYSTEM_PROMPT = f"""
You are inside a temporary state-view selection branch for a single Mac desktop step.
Your job is to decide whether any specific runtime state-card views are worth loading before planner reasoning.

Branch rules:
- Do NOT return raw GUI actions, WAIT, DONE, FAIL, planner JSON, or LOAD_SKILL.
- The CURRENT screenshot is authoritative.
- The loaded skill text and runtime state cards are supplemental references only.
- Request state views only when the CURRENT screenshot appears close enough to one or more state cards that the reference views will reduce ambiguity.
- If the current screenshot is clearly far from the referenced states, or the cards already provide enough information, request no views.
- Use exact `state_id` and exact `view_type` values from the provided runtime state cards.
- You may request multiple complementary views from the same state.
- Keep the request minimal: at most {MAX_STAGE1_SELECTED_STATES} states and at most {MAX_STAGE1_SELECTED_VIEWS} total views.

Output format:
- Return ONLY one code block.
- The code block must contain exactly one `LOAD_STATE_VIEWS([...])` call.
- The payload must be a JSON list of objects. Each object must contain:
  - `"state_id"`: an exact state ID from the runtime state-card manifest
  - `"views"`: a non-empty list of exact view types for that state
  - `"reason"`: one short sentence explaining why those views are needed
- The list may be empty.
- Do not return prose outside the code block.

Correct example with state views:
```text
LOAD_STATE_VIEWS([
  {{
    "state_id": "rename_sheet_dialog_open",
    "views": ["full_frame", "focus_crop", "before", "after"],
    "reason": "I need complementary global, local, and transition views before planning the rename flow."
  }}
])
```

When those exact view types are present in the runtime state-card manifest, `views` may include transition-oriented references such as `before` and `after` in addition to `full_frame` and `focus_crop`.

Correct example without state views:
```text
LOAD_STATE_VIEWS([])
```
""".strip()


OPENAI_SKILL_STAGE2_SYSTEM_PROMPT = """
You are inside a temporary planner-only skill consultation branch for a single Mac desktop step.
Your job is NOT to return a GUI action. Your job is to return a structured planner summary for the CURRENT state.

Branch rules:
- Do not return raw GUI actions, WAIT, DONE, FAIL, LOAD_SKILL, or LOAD_STATE_VIEWS.
- Do not request another skill in this branch.
- The main agent will choose the real GUI action after reading your planner summary.
- Use the CURRENT screenshot first. Skill text, runtime state cards, the stage-1 selection record, and selected reference views are supplemental references only.
- If a skill is ineffective for the CURRENT state, say so clearly and avoid forcing the plan toward the skill.
- Treat selected reference views as state references, never as coordinate templates.
- `subgoal` must be the next immediate local milestone for the user instruction under the CURRENT state.
- Keep `subgoal` short, local, and near-term.
- `plan` must be the longer-range route for solving the user instruction from the CURRENT state after integrating the loaded skill materials and selected reference views.
- `plan` must not collapse into the same content as `subgoal`.
- `plan` must identify the currently relevant UI surface or control area, the next 2 to 4 key actions, checks, or transitions that matter, and the visible cue that means advance versus re-plan.
- If the live UI is not yet at the local state assumed by the skill, say what must be corrected first.
- `expected_state` must describe visible screenshot cues the main agent should aim to reveal next.
- `completion_scope` must be judged against the full user instruction, not only the local subgoal.

Output format:
- Return ONLY one code block.
- The code block must contain exactly one JSON object with these keys:
  - `"skill_applicability"`: one of `"effective"`, `"ineffective"`, `"uncertain"`
  - `"subgoal"`: a short local milestone string
  - `"plan"`: a longer-range behavior plan grounded in the current state
  - `"expected_state"`: a short string describing visible screenshot cues the main agent should aim for next
  - `"completion_scope"`: one of `"local_only"`, `"needs_verification"`, `"maybe_complete"`
- Do not return prose outside the code block.

Correct example:
```json
{
  "skill_applicability": "effective",
  "subgoal": "open the settings surface that exposes the requested control",
  "plan": "stay on the currently visible settings path, operate the row or submenu that should reveal the requested control, then verify the control itself becomes visible before editing it. If the click opens an unrelated panel or the expected control still does not appear, stop following the skill pattern and re-plan from the live UI instead of repeating blindly.",
  "expected_state": "The requested control is visible and ready to edit on the active settings surface",
  "completion_scope": "local_only"
}
```
""".strip()


class OpenAISkillAgent(OpenAI_General_Agent):
    def __init__(
        self,
        model: str,
        remote_client,
        screenshot_rolling_window: int,
        top_p: float,
        temperature: float,
        skill_mode: str = MM_BRANCH_MODE,
        skills_library_dir: str = "skills_library",
    ):
        super().__init__(
            model=model,
            system_prompt=OPENAI_SKILL_MAIN_SYSTEM_PROMPT,
            remote_client=remote_client,
            screenshot_rolling_window=screenshot_rolling_window,
            top_p=top_p,
            temperature=temperature,
        )
        if skill_mode != MM_BRANCH_MODE:
            raise ValueError("Only mm_branch mode is supported for OpenAI skills.")
        self.skill_mode = skill_mode
        self.skills_library_dir = skills_library_dir
        self._skill_loader = SkillLoader(skills_library_dir=skills_library_dir)

        self.responses: List[str] = []
        self.actions: List[str] = []
        self._history_records: List[dict] = []

        self._task_skill_names: List[str] = []
        self._task_skill_metadatas = []
        self._consulted_skills = set()
        self._skill_consult_counts: Dict[str, int] = {}
        self._current_step_planner_summaries: List[Dict[str, Any]] = []
        self._active_skill_state: Optional[Dict[str, Any]] = None
        self._conversation_log: List[dict] = []
        self._skill_invocation_log: List[dict] = []
        self._skill_invocation_counter = 0
        self._result_dir: Optional[str] = None
        self._skill_usage_summary = self._empty_skill_usage_summary()

    def _empty_skill_usage_summary(self) -> Dict[str, object]:
        return {
            "architecture_version": "openai_skill_agent_mm_branch_v1",
            "skill_mode": self.skill_mode,
            "task_skill_names": [],
            "consulted_skill_names": [],
            "load_skill_calls": 0,
            "load_skill_successes": 0,
            "load_state_view_calls": 0,
            "load_state_view_successes": 0,
            "skill_branch_invocations": 0,
            "skill_branch_successes": 0,
            "skill_consult_counts": {},
            "selected_state_views_per_skill": {},
            "total_selected_state_views": 0,
            "active_skill_state": None,
        }

    def set_task_skills(self, skill_names: List[str]):
        self._task_skill_names = list(skill_names or [])
        meta_map = {Path(meta.directory).name: meta for meta in self._skill_loader.discover_all_skills()}
        self._task_skill_metadatas = [meta_map[name] for name in self._task_skill_names if name in meta_map]
        self._consulted_skills = set()
        self._skill_consult_counts = {}
        self._current_step_planner_summaries = []
        self._active_skill_state = None
        self._conversation_log = []
        self._skill_invocation_log = []
        self._skill_invocation_counter = 0
        self.responses = []
        self.actions = []
        self._history_records = []
        self._skill_usage_summary = self._empty_skill_usage_summary()
        self._skill_usage_summary["task_skill_names"] = list(self._task_skill_names)
        logger.info("[OpenAISkill] Task skills resolved: %s", self._task_skill_names)

    def _append_conversation_log(self, entry_type: str, payload: dict):
        self._conversation_log.append({"entry_type": entry_type, **payload})

    def _format_content_elements(self, elements: List[Any]) -> List[dict]:
        formatted: List[dict] = []
        for element in elements:
            if isinstance(element, dict):
                formatted.append(element)
            elif isinstance(element, str):
                formatted.append({"type": "text", "text": element})
            elif isinstance(element, Image.Image):
                formatted.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": pil_to_b64(element, add_prefix=True),
                            "detail": "high",
                        },
                    }
                )
            else:
                raise TypeError(f"Unsupported prompt element type for OpenAI skill agent: {type(element)}")
        return formatted

    def _serialize_content_for_json(self, user_content: List[dict]) -> List[dict]:
        serialized: List[dict] = []
        for item in user_content:
            if not isinstance(item, dict):
                serialized.append({"type": "unknown"})
                continue
            if item.get("type") == "text":
                serialized.append({"type": "text", "text": item.get("text", "")})
                continue
            if item.get("type") == "image_url":
                image_url = item.get("image_url", {}) or {}
                serialized.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "detail": image_url.get("detail", ""),
                            "url": "<omitted>",
                        },
                    }
                )
                continue
            serialized.append({"type": str(item.get("type", "unknown"))})
        return serialized

    def _call_messages(self, system_prompt: str, user_content: List[dict]) -> str:
        response = self.prompt_client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            top_p=self.top_p,
            temperature=self.temperature,
        )
        content = response.choices[0].message.content
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            text = "\n".join(
                item.get("text", "")
                for item in content
                if isinstance(item, dict) and "text" in item
            ).strip()
        else:
            text = str(content or "").strip()
        if not text:
            raise RuntimeError(f"Empty OpenAI skill response for model {self.model}.")
        return text

    def _available_skills_text(self, include_state_previews: bool = False) -> str:
        if not self._task_skill_metadatas:
            return "- None"
        lines: List[str] = []
        for meta in self._task_skill_metadatas:
            skill_name = Path(meta.directory).name
            consult_count = self._skill_consult_counts.get(skill_name, 0)
            header = (
                f"- {skill_name}: {(meta.description or '').strip() or '(no description)'} "
                f"[consulted {consult_count}/{MAX_SKILL_CONSULTS_PER_SKILL}]"
            )
            if not include_state_previews:
                lines.append(header)
                continue
            state_cards = self._skill_loader.load_state_cards(skill_name, runtime=True)
            preview = self._skill_loader.summarize_state_cards_for_preview(
                state_cards,
                max_cards=MAIN_STATE_CARD_PREVIEW_LIMIT,
            )
            lines.append("\n".join([header, preview]))
        return "\n".join(lines)

    def _consulted_skills_text(self) -> str:
        return ", ".join(sorted(self._consulted_skills)) if self._consulted_skills else "None"

    @staticmethod
    def _count_code_blocks(text: str) -> int:
        return len(re.findall(r"```(?:[\w+-]+)?\s*[\s\S]*?```", text or ""))

    @staticmethod
    def _extract_first_code_block_text(text: str) -> Optional[str]:
        if not text:
            return None
        match = re.search(r"```(?:[\w+-]+)?\s*([\s\S]*?)```", text)
        if not match:
            return None
        return match.group(1).strip()

    @staticmethod
    def _normalize_non_codeblock_response(text: str) -> str:
        return (text or "").strip()

    @staticmethod
    def _strip_code_fence(text: str) -> str:
        text = (text or "").strip()
        if text.startswith("```") and text.endswith("```"):
            text = text[3:-3].strip()
        return text.strip("`").strip()

    def _extract_skill_request(self, response_text: str) -> Optional[str]:
        code_body = None
        if self._count_code_blocks(response_text) == 1:
            code_body = self._extract_first_code_block_text(response_text)
        elif self._count_code_blocks(response_text) == 0:
            code_body = self._normalize_non_codeblock_response(response_text)
        if not code_body:
            return None
        raw = self._strip_code_fence(code_body)
        match = re.fullmatch(r'LOAD_SKILL\(\s*"([^"]+)"\s*\)', raw, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()
        match = re.fullmatch(r"LOAD_SKILL\(\s*'([^']+)'\s*\)", raw, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return None

    def _extract_load_state_views_request(
        self,
        response: str,
    ) -> Tuple[Optional[List[Dict[str, object]]], Optional[str]]:
        if not response:
            return None, "The state-view selection response was empty."
        if self._count_code_blocks(response) == 1:
            code_body = self._extract_first_code_block_text(response)
        elif self._count_code_blocks(response) == 0:
            code_body = self._normalize_non_codeblock_response(response)
        else:
            return None, "The state-view selection response must contain exactly one code block with LOAD_STATE_VIEWS([...])."
        if not code_body:
            return None, "The state-view selection response code block was empty."

        normalized = "\n".join(
            line.strip()
            for line in code_body.splitlines()
            if line.strip() and not line.strip().startswith("#")
        ).strip()
        match = re.fullmatch(r"LOAD_STATE_VIEWS\((.*)\)\s*;?", normalized, re.DOTALL)
        if not match:
            return None, "The state-view selection response must be a LOAD_STATE_VIEWS([...]) call."

        payload = match.group(1).strip()
        if not payload:
            return [], None

        try:
            parsed = json.loads(payload)
        except Exception as exc:
            return None, f"LOAD_STATE_VIEWS(...) must contain a valid JSON list. Parse error: {exc}"
        if not isinstance(parsed, list):
            return None, "LOAD_STATE_VIEWS(...) must contain a JSON list of objects."

        merged: Dict[str, Dict[str, object]] = {}
        for item in parsed:
            if not isinstance(item, dict):
                return None, "Each LOAD_STATE_VIEWS(...) item must be a JSON object."
            state_id = str(item.get("state_id", "") or "").strip()
            if not state_id:
                return None, "Each LOAD_STATE_VIEWS(...) item must include a non-empty `state_id`."
            raw_views = item.get("views", [])
            if not isinstance(raw_views, list):
                return None, f"`views` for state '{state_id}' must be a list of strings."
            deduped_views: List[str] = []
            for raw_view in raw_views:
                view_type = str(raw_view).strip()
                if view_type and view_type not in deduped_views:
                    deduped_views.append(view_type)
            if not deduped_views:
                return None, f"`views` for state '{state_id}' must contain at least one non-empty view type."
            reason = str(item.get("reason", "") or "").strip()
            existing = merged.get(state_id)
            if existing is None:
                merged[state_id] = {"state_id": state_id, "views": deduped_views, "reason": reason}
            else:
                existing_views = list(existing.get("views", []))
                for view_type in deduped_views:
                    if view_type not in existing_views:
                        existing_views.append(view_type)
                existing["views"] = existing_views
                if reason and not str(existing.get("reason", "") or "").strip():
                    existing["reason"] = reason

        normalized_items = list(merged.values())
        if len(normalized_items) > MAX_STAGE1_SELECTED_STATES:
            return None, (
                f"Select at most {MAX_STAGE1_SELECTED_STATES} states in LOAD_STATE_VIEWS(...), "
                f"but received {len(normalized_items)}."
            )
        total_view_count = sum(len(item.get("views", [])) for item in normalized_items)
        if total_view_count > MAX_STAGE1_SELECTED_VIEWS:
            return None, (
                f"Select at most {MAX_STAGE1_SELECTED_VIEWS} total views in LOAD_STATE_VIEWS(...), "
                f"but received {total_view_count}."
            )
        return normalized_items, None

    def _extract_planner_summary(self, response: str) -> Tuple[Optional[Dict[str, str]], Optional[str]]:
        if not response:
            return None, "The planner response was empty."
        if self._count_code_blocks(response) == 1:
            code_body = self._extract_first_code_block_text(response)
        elif self._count_code_blocks(response) == 0:
            code_body = self._normalize_non_codeblock_response(response)
        else:
            return None, (
                "The planner response must contain exactly one code block with a JSON object containing "
                "`skill_applicability`, `subgoal`, `plan`, `expected_state`, and `completion_scope`."
            )
        if not code_body:
            return None, "The planner response code block was empty."
        try:
            payload = json.loads(code_body)
        except Exception as exc:
            return None, f"The planner response must be valid JSON. Parse error: {exc}"
        if not isinstance(payload, dict):
            return None, "The planner response JSON must be an object."

        applicability = str(payload.get("skill_applicability", "") or "").strip().lower()
        if applicability not in {"effective", "ineffective", "uncertain"}:
            return None, "The `skill_applicability` field must be one of: effective, ineffective, uncertain."

        subgoal = str(payload.get("subgoal", "") or "").strip()
        plan = str(payload.get("plan", "") or "").strip()
        expected_state = str(payload.get("expected_state", "") or "").strip()
        completion_scope = str(payload.get("completion_scope", "") or "").strip().lower()
        if not subgoal:
            return None, "The `subgoal` field must be a non-empty string."
        if not plan:
            return None, "The `plan` field must be a non-empty string."
        if not expected_state:
            return None, "The `expected_state` field must be a non-empty string."
        if completion_scope not in {"local_only", "needs_verification", "maybe_complete"}:
            return None, "The `completion_scope` field must be one of: local_only, needs_verification, maybe_complete."

        return {
            "skill_applicability": applicability,
            "subgoal": subgoal,
            "plan": plan,
            "expected_state": expected_state,
            "completion_scope": completion_scope,
        }, None

    def _build_repetition_warning_text(self) -> Optional[str]:
        recent_actions = [action for action in self.actions[-3:] if action and action != "No valid action"]
        if len(recent_actions) >= 2 and recent_actions[-1] == recent_actions[-2]:
            return (
                "Recent steps already repeated the same action sequence. Do not repeat the same coordinates, "
                "menu path, or hotkey again unless the CURRENT screenshot shows clear new evidence that it is now correct."
            )
        if len(recent_actions) >= 3 and len(set(recent_actions[-3:])) == 1:
            return (
                "The recent trajectory is looping on the same action sequence. Break the loop by grounding on the "
                "CURRENT screenshot and choosing a meaningfully different next step."
            )
        return None

    def _visible_active_skill_state(self) -> Optional[Dict[str, Any]]:
        state = self._active_skill_state
        if not state:
            return None
        try:
            last_consult_step = int(state.get("last_consult_step"))
        except (TypeError, ValueError):
            last_consult_step = 0
        if last_consult_step <= 0:
            return state
        current_step_number = len(self.responses) + 1
        steps_since_consult = current_step_number - last_consult_step
        if steps_since_consult > ACTIVE_PLANNER_MEMO_TTL_STEPS:
            return None
        return state

    def _active_skill_state_text(self) -> str:
        state = self._visible_active_skill_state()
        if not state:
            return "None"
        return "\n".join(
            [
                f"- Skill: {state.get('skill_name', 'Unknown')}",
                f"- Applicability: {state.get('skill_applicability', 'unknown')}",
                f"- Plan: {state.get('plan', 'None')}",
                f"- Expected state: {state.get('expected_state', 'None')}",
                f"- Completion scope: {state.get('completion_scope', 'needs_verification')}",
                f"- Last consulted at outer step: {state.get('last_consult_step', 'unknown')}",
                f"- Consult count: {state.get('consult_count', 0)}/{MAX_SKILL_CONSULTS_PER_SKILL}",
            ]
        )

    def _current_step_planner_summaries_text(self) -> str:
        if not self._current_step_planner_summaries:
            return "None"
        chunks: List[str] = []
        for idx, item in enumerate(self._current_step_planner_summaries, start=1):
            chunks.append(
                "\n".join(
                    [
                        f"Planner note {idx}:",
                        f"- Skill: {item.get('skill_name', 'Unknown')}",
                        f"- Applicability: {item.get('skill_applicability', 'unknown')}",
                        f"- Subgoal: {item.get('subgoal', 'None')}",
                        f"- Plan: {item.get('plan', 'None')}",
                        f"- Expected state: {item.get('expected_state', 'None')}",
                        f"- Completion scope: {item.get('completion_scope', 'needs_verification')}",
                        f"- Consult count: {item.get('consult_count', 0)}/{MAX_SKILL_CONSULTS_PER_SKILL}",
                    ]
                )
            )
        return "\n\n".join(chunks)

    def _planner_summary_to_record(self, skill_name: str, summary: Dict[str, str]) -> Dict[str, Any]:
        return {
            "skill_name": skill_name,
            "skill_applicability": summary["skill_applicability"],
            "subgoal": summary["subgoal"],
            "plan": summary["plan"],
            "expected_state": summary["expected_state"],
            "completion_scope": summary["completion_scope"],
            "consult_count": self._skill_consult_counts.get(skill_name, 0),
        }

    def _upsert_current_step_planner_summary(self, planner_note: Dict[str, Any]) -> None:
        for idx, existing in enumerate(self._current_step_planner_summaries):
            if existing.get("skill_name") == planner_note.get("skill_name"):
                self._current_step_planner_summaries[idx] = planner_note
                return
        self._current_step_planner_summaries.append(planner_note)

    def _update_active_skill_state(self, planner_note: Dict[str, Any]) -> None:
        applicability = planner_note.get("skill_applicability")
        if applicability == "ineffective":
            active_skill_name = self._active_skill_state.get("skill_name") if self._active_skill_state else None
            if active_skill_name == planner_note.get("skill_name"):
                self._active_skill_state = None
            return
        self._active_skill_state = {
            "skill_name": planner_note.get("skill_name"),
            "skill_applicability": applicability,
            "plan": planner_note.get("plan"),
            "expected_state": planner_note.get("expected_state"),
            "completion_scope": planner_note.get("completion_scope"),
            "consult_count": planner_note.get("consult_count", 0),
            "last_consult_step": len(self.responses) + 1,
        }
        self._skill_usage_summary["active_skill_state"] = dict(self._active_skill_state)

    def _build_previous_history_parts(self) -> List[Any]:
        history_records = self._history_records[-max(0, self.screenshot_rolling_window - 1) :]
        if not history_records:
            return ["No previous screenshot-action history is available."]

        parts: List[Any] = ["Recent grounded interaction history below is ordered as screenshot then issued action text."]
        for record in history_records:
            parts.extend(
                [
                    f"\nHistorical screenshot from outer step {record['step']}:",
                    record["screenshot"],
                    "\nAction issued for the screenshot above:\n```text\n"
                    + str(record.get("response") or record.get("action_summary") or "No valid action").strip()
                    + "\n```",
                ]
            )
        return parts

    def _build_main_user_content(
        self,
        task: str,
        current_screenshot: Image.Image,
        round_feedback: Optional[List[str]] = None,
    ) -> List[dict]:
        active_memo_text = self._active_skill_state_text()
        active_skill_name = self._active_skill_state.get("skill_name") if self._active_skill_state else None
        current_step_skills = {item.get("skill_name") for item in self._current_step_planner_summaries}
        if active_skill_name and active_skill_name in current_step_skills:
            active_memo_text = "Covered by the planner notes returned in this same outer step."

        elements: List[Any] = [
            "Please decide the next grounded response for the CURRENT screenshot. Return either the next raw GUI action block or `LOAD_SKILL(...)` when extra procedural guidance is useful.",
            "\nInstruction:\n" + task,
            "\nAvailable skills for this task (descriptions + compact runtime state-card previews):\n"
            + self._available_skills_text(include_state_previews=True),
            "\nPreviously consulted skills:\n" + self._consulted_skills_text(),
            "\nActive planner memo:\n" + active_memo_text,
            "\nPlanner notes returned in this step:\n" + self._current_step_planner_summaries_text(),
        ]
        if round_feedback:
            feedback_lines = "\n".join(f"- {item}" for item in round_feedback if item)
            if feedback_lines:
                elements.append("\nFeedback for this step:\n" + feedback_lines)
        repetition_warning = self._build_repetition_warning_text()
        if repetition_warning:
            elements.append("\nLoop warning:\n" + repetition_warning)

        elements.extend(self._build_previous_history_parts())
        elements.extend(
            [
                "\nCurrent screenshot (authoritative for the next decision):",
                current_screenshot,
                "\nRules:\n"
                "- Ground every action in the CURRENT screenshot.\n"
                "- Planner notes are fallible references only. They may still be incomplete or partially wrong for the live UI.\n"
                "- Re-decide the real action from the CURRENT screenshot plus recent screenshot-action history before acting.\n"
                "- Treat skills, runtime state cards, selected views, and planner notes as references only, never as coordinate templates.\n"
                f"- Do not reload a skill after {MAX_SKILL_CONSULTS_PER_SKILL} consults.\n"
                "- If planner notes already exist for this step, use them before consulting again.\n"
                "- If recent actions repeated without progress, change strategy.\n"
                "- Before DONE, verify the full instruction, not just a local subgoal.\n"
                "- To click a target, emit `move_to x y` followed by the click action.\n"
                "- In simple high-confidence states, compact 2-4 action blocks are encouraged.",
            ]
        )
        return self._format_content_elements(elements)

    def _build_branch_reference_elements(
        self,
        skill_name: str,
        main_trigger_response: str,
        skill_payload: Dict[str, Any],
    ) -> List[Any]:
        content = skill_payload["content"]
        state_cards = skill_payload.get("runtime_state_cards")
        return [
            "Skill reference package for this temporary planner branch.",
            f"\nRequested skill in the main context: LOAD_SKILL(\"{skill_name}\")",
            "\nMain-context response that triggered this branch:\n```text\n" + main_trigger_response.strip() + "\n```",
            (
                "\nLoaded skill reference: "
                f"{content.name} ({skill_name})\n"
                f"Description: {(content.description or '').strip() or '(no description)'}\n\n"
                "Treat the material below as supplemental procedural knowledge only.\n"
                "Use it to understand workflow stages, state cues, likely subgoals, and success/failure signals.\n"
                "Do NOT treat the text or state cards as coordinate templates.\n"
                "The CURRENT screenshot remains authoritative for concrete GUI actions.\n\n"
                f"{content.text}"
            ),
            "\nRuntime state-card manifest:\n" + self._skill_loader.format_state_cards_for_branch(state_cards),
        ]

    def _build_selected_state_view_elements(
        self,
        skill_name: str,
        selected_selections: List[SkillStateSelection],
    ) -> List[Any]:
        if not selected_selections:
            return ["\nNo specific state views were selected in stage 1."]

        elements: List[Any] = [
            "\nStage-1 selection record for this branch.\n"
            "Use the requested states, requested view types, and selection reasons as part of the planner evidence.\n"
            "The loaded reference views below are supplemental references only. They are never coordinate templates."
        ]
        for selection in selected_selections:
            requested_view_text = ", ".join(selection.requested_view_types) if selection.requested_view_types else "None"
            elements.append(
                "\n".join(
                    [
                        f"[Stage-1 Selection - {skill_name}/{selection.state.state_id}]",
                        f"stage: {selection.state.stage or '(unknown)'}",
                        f"requested_views: {requested_view_text}",
                        f"reason: {selection.reason or '(none provided)'}",
                        f"when_to_use: {selection.state.when_to_use or '(missing)'}",
                        f"verification_cue: {selection.state.verification_cue or '(missing)'}",
                    ]
                )
            )
            for loaded_view in selection.loaded_views:
                filename, b64_data, mime_type = loaded_view.image
                cues = ", ".join(selection.state.visible_cues[:4]) if selection.state.visible_cues else "(no visible cues listed)"
                elements.extend(
                    [
                        "\n".join(
                            [
                                f"[Selected State View - {skill_name}/{selection.state.state_id}/{loaded_view.view.view_type} -> {filename}]",
                                f"state_name: {selection.state.state_name or selection.state.state_id}",
                                f"stage: {selection.state.stage or '(unknown)'}",
                                f"use_for: {loaded_view.view.use_for or '(missing)'}",
                                f"label: {loaded_view.view.label or '(missing)'}",
                                f"visible_cues: {cues}",
                                f"verification_cue: {selection.state.verification_cue or '(missing)'}",
                                f"visual_risk: {selection.state.visual_risk or '(missing)'}",
                            ]
                        ),
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{b64_data}",
                                "detail": "high",
                            },
                        },
                    ]
                )
        return elements

    def _build_stage1_user_content(
        self,
        task: str,
        current_screenshot: Image.Image,
        trigger_skill_name: str,
        main_trigger_response: str,
        skill_payload: Dict[str, Any],
        round_feedback: Optional[List[str]] = None,
    ) -> List[dict]:
        elements: List[Any] = self._build_branch_reference_elements(
            skill_name=trigger_skill_name,
            main_trigger_response=main_trigger_response,
            skill_payload=skill_payload,
        )
        elements.extend(
            [
                "\nPlease inspect the CURRENT UI screenshot and decide whether any specific state views are worth loading.",
                "\nInstruction:\n" + task,
            ]
        )
        if round_feedback:
            feedback_lines = "\n".join(f"- {item}" for item in round_feedback if item)
            if feedback_lines:
                elements.append("\nAdditional feedback for this state-view selection round:\n" + feedback_lines)
        repetition_warning = self._build_repetition_warning_text()
        if repetition_warning:
            elements.append("\nLoop warning:\n" + repetition_warning)
        elements.extend(self._build_previous_history_parts())
        elements.extend(
            [
                "\nCurrent screenshot (authoritative for state-view selection):",
                current_screenshot,
                "\nRules:\n"
                "- Return `LOAD_STATE_VIEWS([...])` only.\n"
                "- The list may be empty when the runtime state cards already suffice or the screenshot is too different.\n"
                f"- Select at most {MAX_STAGE1_SELECTED_STATES} states and at most {MAX_STAGE1_SELECTED_VIEWS} total views.\n"
                "- Prefer exact `state_id` and `view_type` values from the runtime state-card manifest.\n"
                "- Do not request reference views that mainly duplicate one another without adding new evidence.\n"
                "- Do not assume that a visually similar state is helpful if the card's `when_not_to_use` or `visual_risk` warns against it.",
            ]
        )
        return self._format_content_elements(elements)

    def _build_stage2_user_content(
        self,
        task: str,
        current_screenshot: Image.Image,
        trigger_skill_name: str,
        main_trigger_response: str,
        skill_payload: Dict[str, Any],
        selected_selections: List[SkillStateSelection],
        round_feedback: Optional[List[str]] = None,
    ) -> List[dict]:
        elements: List[Any] = self._build_branch_reference_elements(
            skill_name=trigger_skill_name,
            main_trigger_response=main_trigger_response,
            skill_payload=skill_payload,
        )
        elements.extend(self._build_selected_state_view_elements(trigger_skill_name, selected_selections))
        selected_text = (
            ", ".join(
                f"{selection.state.state_id}/{loaded_view.view.view_type}"
                for selection in selected_selections
                for loaded_view in selection.loaded_views
            )
            if selected_selections
            else "None"
        )
        elements.extend(
            [
                "\nPlease inspect the CURRENT UI screenshot and return planner JSON only.",
                "\nInstruction:\n" + task,
                f"\nSelected state views for this branch: {selected_text}",
            ]
        )
        if round_feedback:
            feedback_lines = "\n".join(f"- {item}" for item in round_feedback if item)
            if feedback_lines:
                elements.append("\nAdditional feedback for this planner round:\n" + feedback_lines)
        repetition_warning = self._build_repetition_warning_text()
        if repetition_warning:
            elements.append("\nLoop warning:\n" + repetition_warning)
        elements.extend(self._build_previous_history_parts())
        elements.extend(
            [
                "\nCurrent screenshot (authoritative for planner reasoning):",
                current_screenshot,
                "\nRules:\n"
                "- Keep the planner grounded in the CURRENT screenshot.\n"
                "- Use the loaded skill only for the specific procedural knowledge that matters now.\n"
                "- Use the stage-1 selection record as evidence about why these references were loaded.\n"
                "- `subgoal` should stay local and immediate: the next small milestone under the live UI.\n"
                "- `plan` should be the longer-range behavior route for solving the user instruction from the CURRENT state after incorporating the loaded skill materials and selected reference views.\n"
                "- `plan` should cover the currently relevant UI surface, the next 2 to 4 key actions/checks or transitions, and what visible cue means advance versus re-plan.\n"
                "- Do not let `plan` collapse into the same content as `subgoal`.\n"
                "- Do not let skill examples or selected views override what is actually visible now.\n"
                "- If the live UI is not yet at the local state assumed by the skill, say what must be corrected first.\n"
                "- `expected_state` must describe visible cues, not an abstract end goal.",
            ]
        )
        return self._format_content_elements(elements)

    def _record_selected_branch_state_views(
        self,
        skill_name: str,
        selected_selections: List[SkillStateSelection],
    ) -> None:
        selected_per_skill = dict(self._skill_usage_summary.get("selected_state_views_per_skill", {}))
        selected_count = sum(len(selection.loaded_views) for selection in selected_selections)
        if selected_count > 0:
            selected_per_skill[skill_name] = selected_per_skill.get(skill_name, 0) + selected_count
        self._skill_usage_summary["selected_state_views_per_skill"] = selected_per_skill
        self._skill_usage_summary["total_selected_state_views"] = sum(selected_per_skill.values())

    def _load_skill_for_branch(
        self,
        skill_name: str,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        self._skill_usage_summary["load_skill_calls"] = int(self._skill_usage_summary.get("load_skill_calls", 0)) + 1
        if not skill_name:
            return None, "Missing skill name in LOAD_SKILL(...)."
        if skill_name not in self._task_skill_names:
            return None, f"Unknown skill '{skill_name}'. Use only a skill from the available skill list."
        if self._skill_consult_counts.get(skill_name, 0) >= MAX_SKILL_CONSULTS_PER_SKILL:
            return (
                None,
                f"Skill '{skill_name}' has already been consulted {MAX_SKILL_CONSULTS_PER_SKILL} times in this trajectory. "
                "Do not load it again. Continue with grounded GUI interaction using the CURRENT screenshot and recent screenshot-action history.",
            )

        content = self._skill_loader.load_skill_content(skill_name)
        if content is None:
            return None, f"Failed to load skill text for '{skill_name}'."
        runtime_state_cards = self._skill_loader.load_state_cards(skill_name, runtime=True)
        if runtime_state_cards is None:
            runtime_state_cards = self._skill_loader.load_state_cards(skill_name, runtime=False)

        self._skill_consult_counts[skill_name] = self._skill_consult_counts.get(skill_name, 0) + 1
        self._consulted_skills.add(skill_name)
        self._skill_usage_summary["load_skill_successes"] = int(self._skill_usage_summary.get("load_skill_successes", 0)) + 1
        self._skill_usage_summary["consulted_skill_names"] = sorted(self._consulted_skills)
        self._skill_usage_summary["skill_consult_counts"] = dict(self._skill_consult_counts)
        return {
            "content": content,
            "runtime_state_cards": runtime_state_cards,
        }, None

    def _run_skill_branch(
        self,
        task: str,
        current_screenshot: Image.Image,
        trigger_skill_name: str,
        main_trigger_response: str,
        step_idx: int,
    ) -> Dict[str, Any]:
        self._skill_invocation_counter += 1
        branch_id = self._skill_invocation_counter
        selected_state_view_requests: List[Dict[str, object]] = []
        selected_selections: List[SkillStateSelection] = []
        branch_rounds: List[dict] = []
        stage1_feedback: List[str] = []
        planner_feedback: List[str] = []
        final_response = ""
        final_summary: Optional[Dict[str, str]] = None
        success = False

        skill_payload, load_error = self._load_skill_for_branch(trigger_skill_name)
        if skill_payload is None:
            branch_log = {
                "architecture_version": self._skill_usage_summary.get("architecture_version"),
                "branch_id": branch_id,
                "step": step_idx,
                "skill_mode": self.skill_mode,
                "trigger_skill_name": trigger_skill_name,
                "main_trigger_response": main_trigger_response,
                "success": False,
                "loaded_skills": [],
                "selected_state_view_requests": [],
                "selected_state_view_ids": [],
                "selected_state_view_paths": [],
                "selected_state_view_count": 0,
                "rounds": [],
                "final_response": "",
                "final_summary": None,
                "final_feedback": load_error or "Failed to load the requested skill.",
            }
            return {
                "success": False,
                "response": "",
                "summary": None,
                "feedback": load_error or "Failed to load the requested skill.",
                "log": branch_log,
            }

        for round_idx in range(MAX_STATE_VIEW_SELECTION_ROUNDS):
            user_content = self._build_stage1_user_content(
                task=task,
                current_screenshot=current_screenshot,
                trigger_skill_name=trigger_skill_name,
                main_trigger_response=main_trigger_response,
                skill_payload=skill_payload,
                round_feedback=stage1_feedback,
            )
            response = self._call_messages(OPENAI_SKILL_STAGE1_SYSTEM_PROMPT, user_content)
            final_response = response or ""
            requested_items, parse_error = self._extract_load_state_views_request(final_response)
            round_record = {
                "stage": "state_view_selection",
                "round": round_idx + 1,
                "timestamp": time.time(),
                "system_message": OPENAI_SKILL_STAGE1_SYSTEM_PROMPT,
                "contents": self._serialize_content_for_json(user_content),
                "response": final_response,
            }
            if parse_error:
                stage1_feedback.append(parse_error)
                round_record["status"] = "invalid_state_view_selection"
                round_record["error"] = parse_error
                branch_rounds.append(round_record)
                continue

            selected_state_view_requests = list(requested_items or [])
            self._skill_usage_summary["load_state_view_calls"] = int(
                self._skill_usage_summary.get("load_state_view_calls", 0)
            ) + 1
            selected_selections, missing_items = self._skill_loader.load_selected_state_views(
                trigger_skill_name,
                selected_state_view_requests,
                runtime=True,
            )
            round_record["requested_state_view_requests"] = list(selected_state_view_requests)
            round_record["selected_state_view_ids"] = [
                f"{selection.state.state_id}/{loaded_view.view.view_type}"
                for selection in selected_selections
                for loaded_view in selection.loaded_views
            ]
            round_record["selected_state_view_paths"] = [
                loaded_view.view.image_path
                for selection in selected_selections
                for loaded_view in selection.loaded_views
            ]
            round_record["missing_state_views"] = list(missing_items)
            if missing_items:
                stage1_feedback.append(
                    "Some requested state IDs or view types could not be resolved. Use exact `state_id` and `view_type` values from the runtime state-card manifest."
                )
                round_record["status"] = "state_view_selection_missing_identifiers"
                branch_rounds.append(round_record)
                selected_selections = []
                continue

            self._skill_usage_summary["load_state_view_successes"] = int(
                self._skill_usage_summary.get("load_state_view_successes", 0)
            ) + 1
            self._record_selected_branch_state_views(trigger_skill_name, selected_selections)
            round_record["status"] = "state_view_selection_returned"
            round_record["selected_state_view_count"] = sum(
                len(selection.loaded_views) for selection in selected_selections
            )
            branch_rounds.append(round_record)
            break
        else:
            branch_log = {
                "architecture_version": self._skill_usage_summary.get("architecture_version"),
                "branch_id": branch_id,
                "step": step_idx,
                "skill_mode": self.skill_mode,
                "trigger_skill_name": trigger_skill_name,
                "main_trigger_response": main_trigger_response,
                "success": False,
                "loaded_skills": [trigger_skill_name],
                "selected_state_view_requests": selected_state_view_requests,
                "selected_state_view_ids": [],
                "selected_state_view_paths": [],
                "selected_state_view_count": 0,
                "rounds": branch_rounds,
                "final_response": final_response,
                "final_summary": None,
                "final_feedback": stage1_feedback[-1] if stage1_feedback else "State-view selection stage failed.",
            }
            return {
                "success": False,
                "response": final_response,
                "summary": None,
                "feedback": stage1_feedback[-1] if stage1_feedback else "State-view selection stage failed.",
                "log": branch_log,
            }

        for round_idx in range(MAX_PLANNER_ROUNDS):
            user_content = self._build_stage2_user_content(
                task=task,
                current_screenshot=current_screenshot,
                trigger_skill_name=trigger_skill_name,
                main_trigger_response=main_trigger_response,
                skill_payload=skill_payload,
                selected_selections=selected_selections,
                round_feedback=planner_feedback,
            )
            response = self._call_messages(OPENAI_SKILL_STAGE2_SYSTEM_PROMPT, user_content)
            final_response = response or ""
            summary, parse_error = self._extract_planner_summary(final_response)
            round_record = {
                "stage": "planner",
                "round": round_idx + 1,
                "timestamp": time.time(),
                "system_message": OPENAI_SKILL_STAGE2_SYSTEM_PROMPT,
                "contents": self._serialize_content_for_json(user_content),
                "response": final_response,
                "selected_state_view_requests": list(selected_state_view_requests),
                "selected_state_view_ids": [
                    f"{selection.state.state_id}/{loaded_view.view.view_type}"
                    for selection in selected_selections
                    for loaded_view in selection.loaded_views
                ],
            }
            if parse_error:
                planner_feedback.append(parse_error)
                round_record["status"] = "invalid_planner_summary"
                round_record["error"] = parse_error
                branch_rounds.append(round_record)
                continue

            success = True
            final_summary = summary
            round_record["status"] = "planner_summary_returned"
            round_record["planner_summary"] = dict(summary)
            branch_rounds.append(round_record)
            break

        branch_log = {
            "architecture_version": self._skill_usage_summary.get("architecture_version"),
            "branch_id": branch_id,
            "step": step_idx,
            "skill_mode": self.skill_mode,
            "trigger_skill_name": trigger_skill_name,
            "main_trigger_response": main_trigger_response,
            "success": success,
            "loaded_skills": [trigger_skill_name],
            "selected_state_view_requests": list(selected_state_view_requests),
            "selected_state_view_ids": [
                f"{selection.state.state_id}/{loaded_view.view.view_type}"
                for selection in selected_selections
                for loaded_view in selection.loaded_views
            ],
            "selected_state_view_paths": [
                loaded_view.view.image_path
                for selection in selected_selections
                for loaded_view in selection.loaded_views
            ],
            "selected_state_view_count": sum(len(selection.loaded_views) for selection in selected_selections),
            "rounds": branch_rounds,
            "final_response": final_response,
            "final_summary": dict(final_summary) if final_summary else None,
            "final_feedback": planner_feedback[-1] if planner_feedback else None,
        }
        return {
            "success": success,
            "response": final_response,
            "summary": final_summary,
            "feedback": planner_feedback[-1] if planner_feedback else "Skill planner branch did not produce a valid summary.",
            "log": branch_log,
        }

    def step(
        self,
        task_id: int,
        current_step: int,
        max_steps: int,
        env_language: str,
        task_language: str,
        task: str,
        task_step_timeout: int,
        save_dir: str,
    ):
        self._result_dir = save_dir
        with timeout(task_step_timeout):
            print_message(
                title=f"Task {task_id}/{env_language}/{task_language} Step {current_step}/{max_steps}",
                content="Capturing screenshot...",
            )
            current_screenshot = self.remote_client.capture_screenshot().copy()
            self.screenshots.append(current_screenshot.copy())
            self.screenshots = self.screenshots[-self.screenshot_rolling_window :]

            print_message(
                title=f"Task {task_id}/{env_language}/{task_language} Step {current_step}/{max_steps}",
                content="Calling GUI agent...",
            )
            self._current_step_planner_summaries = []
            round_feedback: List[str] = []
            parsed_actions: List[dict] = []
            final_response = ""

            for round_idx in range(MAX_MAIN_RESPONSE_ROUNDS):
                user_content = self._build_main_user_content(
                    task=task,
                    current_screenshot=current_screenshot,
                    round_feedback=round_feedback,
                )
                main_response = self._call_messages(OPENAI_SKILL_MAIN_SYSTEM_PROMPT, user_content)
                final_response = main_response
                self._append_conversation_log(
                    "main_round",
                    {
                        "step": current_step,
                        "round": round_idx + 1,
                        "response": main_response,
                        "feedback_before_round": list(round_feedback),
                        "contents": self._serialize_content_for_json(user_content),
                    },
                )

                skill_request = self._extract_skill_request(main_response)
                if skill_request:
                    branch_result = self._run_skill_branch(
                        task=task,
                        current_screenshot=current_screenshot,
                        trigger_skill_name=skill_request,
                        main_trigger_response=main_response,
                        step_idx=current_step,
                    )
                    self._skill_invocation_log.append(branch_result["log"])
                    self._append_conversation_log(
                        "skill_branch",
                        {
                            "step": current_step,
                            "round": round_idx + 1,
                            "skill_request": skill_request,
                            "branch_success": branch_result["success"],
                            "branch_feedback": branch_result["feedback"],
                            "branch_summary": branch_result["summary"],
                        },
                    )
                    if branch_result["success"] and branch_result["summary"]:
                        planner_note = self._planner_summary_to_record(skill_request, branch_result["summary"])
                        self._upsert_current_step_planner_summary(planner_note)
                        self._update_active_skill_state(planner_note)
                        self._skill_usage_summary["skill_branch_invocations"] = len(self._skill_invocation_log)
                        self._skill_usage_summary["skill_branch_successes"] = sum(
                            1 for item in self._skill_invocation_log if item.get("success")
                        )
                        round_feedback.append(
                            f"Planner note loaded from skill '{skill_request}'. Use the planner note plus the CURRENT screenshot to decide the real next action. Do not return LOAD_SKILL unless another skill is still necessary."
                        )
                    else:
                        round_feedback.append(
                            branch_result["feedback"] or f"Skill branch for '{skill_request}' failed."
                        )
                    continue

                parsed_actions = self.parse_agent_output(main_response)
                if parsed_actions:
                    break

                round_feedback.append(
                    "Invalid action format. Use only supported action names. To click a target, emit `move_to x y` followed by the click action."
                )

            print_message(
                title=f"Task {task_id}/{env_language}/{task_language} Step {current_step}/{max_steps}",
                content="Actuating...",
            )
            status, _ = self.execute_actions(parsed_actions)

        current_screenshot.save(os.path.join(save_dir, "context", f"step_{str(current_step).zfill(3)}.png"))
        with open(os.path.join(save_dir, "context", f"step_{str(current_step).zfill(3)}_raw_response.txt"), "w") as f:
            f.write(final_response)
        with open(os.path.join(save_dir, "context", f"step_{str(current_step).zfill(3)}_parsed_actions.json"), "w") as f:
            json.dump(parsed_actions, f, indent=4)

        action_summary = (
            " | ".join(json.dumps(action, ensure_ascii=False) for action in parsed_actions)
            if parsed_actions
            else "No valid action"
        )
        self.responses.append(final_response)
        self.actions.append(action_summary)
        self._history_records.append(
            {
                "step": current_step,
                "screenshot": current_screenshot.copy(),
                "response": final_response,
                "action_summary": action_summary,
            }
        )
        self._history_records = self._history_records[-MAX_HISTORY_RECORDS:]
        self._skill_usage_summary["consulted_skill_names"] = sorted(self._consulted_skills)
        self._skill_usage_summary["skill_consult_counts"] = dict(self._skill_consult_counts)
        self._skill_usage_summary["active_skill_state"] = dict(self._active_skill_state) if self._active_skill_state else None
        return status

    def save_conversation_history(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        if self._conversation_log:
            with open(os.path.join(save_dir, "conversation.json"), "w", encoding="utf-8") as f:
                json.dump(self._conversation_log, f, indent=2, ensure_ascii=False)
        if self._skill_invocation_log:
            with open(os.path.join(save_dir, "skill_invocations.json"), "w", encoding="utf-8") as f:
                json.dump(self._skill_invocation_log, f, indent=2, ensure_ascii=False)
        payload = dict(self._skill_usage_summary)
        payload.update(
            {
                "task_skill_names": list(self._task_skill_names),
                "consulted_skill_names": sorted(self._consulted_skills),
                "skill_consult_counts": dict(self._skill_consult_counts),
                "steps_recorded": len(self.actions),
                "final_actions": list(self.actions),
                "skill_branch_invocations": len(self._skill_invocation_log),
                "skill_branch_successes": sum(1 for item in self._skill_invocation_log if item.get("success")),
                "active_skill_state": dict(self._active_skill_state) if self._active_skill_state else None,
                "max_skill_consults_per_skill": MAX_SKILL_CONSULTS_PER_SKILL,
            }
        )
        with open(os.path.join(save_dir, "skill_usage_summary.json"), "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image

from agent.gemini import GEMINI_SYSTEM_PROMPT, Gemini_General_Agent
from agent.skill_loader import SkillLoader
from utils.log import print_message


logger = logging.getLogger("macosworld.gemini_skill")

TEXT_INLINE_MODE = "text_inline"
TEXT_BRANCH_MODE = "text_branch"
MM_BRANCH_MODE = "mm_branch"

MAX_SKILL_BRANCH_ROUNDS = 6
MAX_INLINE_SKILLS = 5
MAX_INLINE_SKILL_CHARS = 4000
MAX_BRANCH_IMAGE_COUNT = 6
MAX_RUNTIME_PREVIEW_STATES = 3


def _truncate_text(text: str, max_chars: int) -> str:
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n\n[truncated]"


GEMINI_SKILL_MAIN_SYSTEM_PROMPT = (
    GEMINI_SYSTEM_PROMPT
    + "\n\n"
    + """
Additional skill rule:
- You may optionally request exactly one skill by returning `LOAD_SKILL("EXACT_SKILL_NAME")` in a fenced plaintext block.
- Only request a skill if the current screenshot and task suggest procedural guidance would help with the next immediate action.
- If you request a skill, do not output any other action in the same round.
- Skills are references only. Do not copy coordinates from them.
- After a skill is loaded, use the current screenshot as the source of truth.
- When clicking a target, emit `move_to x y` before the click action unless the cursor is already clearly positioned.
- In high-confidence, simple UI states, it is acceptable to emit a compact 2-4 action block if the actions are directly grounded by the current screenshot.
""".strip()
)


GEMINI_SKILL_BRANCH_SYSTEM_PROMPT = (
    GEMINI_SYSTEM_PROMPT
    + "\n\n"
    + """
You are inside a temporary skill-guidance branch.
- Do not request another skill in this branch.
- Use the loaded skill as procedural guidance only.
- If the skill conflicts with the current screenshot, trust the current screenshot.
- Return only executable Mac action commands in a fenced plaintext block.
- Prefer click sequences grounded on the current screenshot, usually `move_to` then `left_click`.
- In simple, high-confidence cases, a short 2-4 action block is preferred over a single under-specified action.
""".strip()
)


class GeminiSkillAgent(Gemini_General_Agent):
    def __init__(
        self,
        model: str,
        remote_client,
        only_n_most_recent_images: int,
        max_tokens: int,
        top_p: float,
        temperature: float,
        safety_config: dict,
        skill_mode: str = TEXT_INLINE_MODE,
        skills_library_dir: str = "skills_library",
    ):
        super().__init__(
            model=model,
            system_prompt=GEMINI_SYSTEM_PROMPT,
            remote_client=remote_client,
            only_n_most_recent_images=only_n_most_recent_images,
            max_tokens=max_tokens,
            top_p=top_p,
            temperature=temperature,
            safety_config=safety_config,
        )
        if skill_mode not in {TEXT_INLINE_MODE, TEXT_BRANCH_MODE, MM_BRANCH_MODE}:
            raise ValueError("Unsupported Gemini skill mode")
        self.skill_mode = skill_mode
        self.skills_library_dir = skills_library_dir
        self._skill_loader = SkillLoader(skills_library_dir=skills_library_dir)
        self._task_skill_names: List[str] = []
        self._task_skill_metadatas = []
        self._consulted_skills = set()
        self._action_history: List[str] = []
        self._conversation_log: List[dict] = []
        self._skill_invocation_log: List[dict] = []

    def set_task_skills(self, skill_names: List[str]):
        self._task_skill_names = list(skill_names or [])
        meta_map = {Path(meta.directory).name: meta for meta in self._skill_loader.discover_all_skills()}
        self._task_skill_metadatas = [meta_map[name] for name in self._task_skill_names if name in meta_map]
        self._consulted_skills = set()
        self._action_history = []
        self._conversation_log = []
        self._skill_invocation_log = []
        logger.info("[GeminiSkill] Task skills resolved: %s", self._task_skill_names)

    def _append_conversation_log(self, entry_type: str, payload: dict):
        self._conversation_log.append({"entry_type": entry_type, **payload})

    def _available_skills_text(self, include_runtime_preview: bool = False) -> str:
        if not self._task_skill_metadatas:
            return "- None"
        lines = []
        for meta in self._task_skill_metadatas:
            skill_name = Path(meta.directory).name
            header = f"- {skill_name}: {(meta.description or '').strip() or '(no description)'}"
            if include_runtime_preview:
                preview = self._skill_loader.summarize_runtime_state_cards(
                    skill_name, max_states=MAX_RUNTIME_PREVIEW_STATES
                )
                lines.append("\n".join([header, preview]))
            else:
                lines.append(header)
        return "\n".join(lines)

    def _consulted_skills_text(self) -> str:
        return ", ".join(sorted(self._consulted_skills)) if self._consulted_skills else "None"

    def _build_inline_skill_context(self) -> str:
        if not self._task_skill_names:
            return "No task skills are available."
        chunks: List[str] = []
        for skill_name in self._task_skill_names[:MAX_INLINE_SKILLS]:
            full_skill = self._skill_loader.load_full_skill(skill_name)
            if full_skill is None:
                continue
            content = full_skill["content"]
            runtime_preview = self._skill_loader.summarize_runtime_state_cards(
                skill_name, max_states=MAX_RUNTIME_PREVIEW_STATES
            )
            chunks.append(
                "\n".join(
                    [
                        f"Skill: {skill_name}",
                        f"Description: {(content.description or '').strip() or '(no description)'}",
                        f"Runtime state preview:\n{runtime_preview}",
                        "Skill reference:",
                        _truncate_text(content.text, MAX_INLINE_SKILL_CHARS),
                    ]
                )
            )
        return "\n\n".join(chunks) if chunks else "No task skills are available."

    def _call_with_prompt_elements(self, system_prompt: str, prompt_elements: List):
        if self.use_gateway:
            request_parts = []
            for element in prompt_elements:
                if isinstance(element, str):
                    request_parts.append({"text": element})
                elif isinstance(element, Image.Image):
                    from agent.gemini import pil_to_gateway_inline_data

                    request_parts.append({"inlineData": pil_to_gateway_inline_data(element)})
                else:
                    raise TypeError(f"Unsupported prompt element type for Gemini skill gateway: {type(element)}")

            response = self.prompt_client.generate_content(
                parts=request_parts,
                system_text=system_prompt,
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
                top_p=self.top_p,
                seed=self.gateway_seed,
                thinking_level=self.gateway_thinking_level,
                include_thoughts=self.gateway_include_thoughts,
            )
            usage_metadata = response.get("usageMetadata", {})
            prompt_tokens = usage_metadata.get("promptTokenCount", 0)
            candidate_tokens = usage_metadata.get("candidatesTokenCount", 0)
            self.token_usage.append(
                {
                    "prompt_token_count": prompt_tokens,
                    "candidates_token_count": candidate_tokens,
                }
            )
            self.total_prompt_tokens += prompt_tokens
            self.total_candidates_tokens += candidate_tokens
            response_parts = response.get("candidates", [{}])[0].get("content", {}).get("parts", [])
            response_text = "\n".join(
                part.get("text", "")
                for part in response_parts
                if isinstance(part, dict) and "text" in part
            ).strip()
            if response_text == "":
                raise RuntimeError("Empty Gemini skill gateway response.")
            return response_text

        prompt = [system_prompt, *prompt_elements]
        response = self.prompt_client.generate_content(
            prompt,
            generation_config=dict(
                candidate_count=1,
                max_output_tokens=self.max_tokens,
                top_p=self.top_p,
                temperature=self.temperature,
            ),
            safety_settings=self.safety_config,
        )
        self.token_usage.append(
            {
                "prompt_token_count": response.usage_metadata.prompt_token_count,
                "candidates_token_count": response.usage_metadata.candidates_token_count,
            }
        )
        self.total_prompt_tokens += response.usage_metadata.prompt_token_count
        self.total_candidates_tokens += response.usage_metadata.candidates_token_count
        return response.candidates[0].content.parts[0].text

    @staticmethod
    def _strip_code_fence(text: str) -> str:
        text = text.strip()
        if text.startswith("```") and text.endswith("```"):
            text = text[3:-3].strip()
        return text.strip("`").strip()

    def _extract_skill_request(self, response_text: str) -> Optional[str]:
        raw = self._strip_code_fence(response_text)
        match = re.fullmatch(r'LOAD_SKILL\(\s*"([^"]+)"\s*\)', raw, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()
        match = re.fullmatch(r"LOAD_SKILL\(\s*'([^']+)'\s*\)", raw, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return None

    def _construct_main_prompt_elements(
        self,
        task: str,
        screenshots: List[Image.Image],
        round_feedback: Optional[List[str]] = None,
    ) -> List:
        prompt_parts: List = [
            "Task: " + task,
            "\nAvailable skills:\n"
            + self._available_skills_text(include_runtime_preview=self.skill_mode == MM_BRANCH_MODE),
            "\nPreviously consulted skills: " + self._consulted_skills_text(),
            "\nPrevious action summaries:\n" + ("\n".join(self._action_history[-5:]) if self._action_history else "None"),
        ]
        if self.skill_mode == TEXT_INLINE_MODE:
            prompt_parts.append("\nLoaded text skill references:\n" + self._build_inline_skill_context())
        if round_feedback:
            prompt_parts.append("\nFeedback from previous invalid response(s):\n" + "\n".join(f"- {x}" for x in round_feedback))

        if len(screenshots) == 1:
            prompt_parts.extend(["\nCurrent screenshot: ", screenshots[0]])
        else:
            prompt_parts.extend(
                [
                    "\nRolling window of historical screenshots in chronological order: ",
                    *screenshots[:-1],
                    "\nCurrent screenshot: ",
                    screenshots[-1],
                ]
            )
        return prompt_parts

    def _construct_branch_prompt_elements(
        self,
        task: str,
        current_screenshot: Image.Image,
        skill_name: str,
        full_skill: Dict,
    ) -> List:
        content = full_skill["content"]
        prompt_parts: List = [
            "Task: " + task,
            "\nCurrent screenshot: ",
            current_screenshot,
            (
                "\nLoaded skill reference: "
                f"{content.name} ({skill_name})\n"
                f"Description: {(content.description or '').strip() or '(no description)'}\n\n"
                f"Runtime state preview:\n{self._skill_loader.summarize_runtime_state_cards(skill_name, max_states=MAX_RUNTIME_PREVIEW_STATES)}\n\n"
                f"Skill content:\n{content.text}"
            ),
        ]

        if self.skill_mode == MM_BRANCH_MODE:
            for idx, (filename, b64_data, mime_type) in enumerate(full_skill.get("images", [])[:MAX_BRANCH_IMAGE_COUNT], start=1):
                prompt_parts.append(f"\n[Skill visual reference {idx}: {skill_name}/{filename}]")
                skill_image_path = self._resolve_skill_image_path(Path(full_skill["content"].directory), filename)
                if skill_image_path is None:
                    continue
                with Image.open(skill_image_path) as image:
                    prompt_parts.append(image.copy())
        return prompt_parts

    @staticmethod
    def _resolve_skill_image_path(skill_dir: Path, filename: str) -> Optional[Path]:
        direct = skill_dir / "Images" / filename
        if direct.exists():
            return direct
        alt = skill_dir / "images" / filename
        if alt.exists():
            return alt
        for child in skill_dir.iterdir():
            candidate = child / filename
            if child.is_dir() and child.name.lower() == "images" and candidate.exists():
                return candidate
        return None

    def _run_skill_branch(
        self,
        task: str,
        current_screenshot: Image.Image,
        trigger_skill_name: str,
        main_response: str,
        current_step: int,
    ) -> Tuple[Optional[str], Optional[List[dict]], Optional[str]]:
        if trigger_skill_name not in self._task_skill_names:
            return None, None, f'Unknown skill "{trigger_skill_name}".'
        full_skill = self._skill_loader.load_full_skill(trigger_skill_name)
        if full_skill is None:
            return None, None, f'Failed to load skill "{trigger_skill_name}".'

        self._consulted_skills.add(trigger_skill_name)
        branch_elements = self._construct_branch_prompt_elements(
            task=task,
            current_screenshot=current_screenshot,
            skill_name=trigger_skill_name,
            full_skill=full_skill,
        )
        branch_response = self._call_with_prompt_elements(GEMINI_SKILL_BRANCH_SYSTEM_PROMPT, branch_elements)
        parsed_actions = self.parse_agent_output(branch_response)
        self._skill_invocation_log.append(
            {
                "step": current_step,
                "trigger_skill_name": trigger_skill_name,
                "main_response": main_response,
                "branch_response": branch_response,
                "parsed_actions": parsed_actions,
                "loaded_image_count": len(full_skill.get("images", [])) if self.skill_mode == MM_BRANCH_MODE else 0,
                "success": len(parsed_actions) > 0,
            }
        )
        if not parsed_actions:
            return branch_response, None, "Loaded skill branch returned no valid action."
        return branch_response, parsed_actions, None

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
        print_message(
            title=f"Task {task_id}/{env_language}/{task_language} Step {current_step}/{max_steps}",
            content="Capturing screenshot...",
        )
        current_screenshot = self.remote_client.capture_screenshot()
        self.screenshots.append(current_screenshot.copy())
        if self.only_n_most_recent_images > 0:
            self.screenshots = self.screenshots[-self.only_n_most_recent_images :]

        print_message(
            title=f"Task {task_id}/{env_language}/{task_language} Step {current_step}/{max_steps}",
            content="Calling GUI agent...",
        )

        final_response = ""
        parsed_actions: List[dict] = []
        round_feedback: List[str] = []
        system_prompt = GEMINI_SYSTEM_PROMPT if self.skill_mode == TEXT_INLINE_MODE else GEMINI_SKILL_MAIN_SYSTEM_PROMPT

        for round_idx in range(MAX_SKILL_BRANCH_ROUNDS):
            main_elements = self._construct_main_prompt_elements(
                task=task,
                screenshots=self.screenshots,
                round_feedback=round_feedback,
            )
            main_response = self._call_with_prompt_elements(system_prompt, main_elements)
            skill_request = None if self.skill_mode == TEXT_INLINE_MODE else self._extract_skill_request(main_response)

            self._append_conversation_log(
                "main_round",
                {
                    "step": current_step,
                    "round": round_idx + 1,
                    "response": main_response,
                    "feedback_before_round": list(round_feedback),
                },
            )

            if skill_request:
                branch_response, branch_actions, branch_error = self._run_skill_branch(
                    task=task,
                    current_screenshot=current_screenshot,
                    trigger_skill_name=skill_request,
                    main_response=main_response,
                    current_step=current_step,
                )
                self._append_conversation_log(
                    "skill_branch",
                    {
                        "step": current_step,
                        "round": round_idx + 1,
                        "skill_request": skill_request,
                        "branch_response": branch_response,
                        "branch_error": branch_error,
                    },
                )
                if branch_actions:
                    final_response = branch_response or ""
                    parsed_actions = branch_actions
                    break
                round_feedback.append(branch_error or "Skill branch returned no valid action.")
                continue

            parsed_actions = self.parse_agent_output(main_response)
            if parsed_actions:
                final_response = main_response
                break
            round_feedback.append(
                "Invalid action format. Use only supported action names. To click a target, emit `move_to x y` followed by `left_click`."
            )

        print_message(
            title=f"Task {task_id}/{env_language}/{task_language} Step {current_step}/{max_steps}",
            content="Actuating...",
        )
        status, _ = self.execute_actions(parsed_actions)

        if parsed_actions:
            self._action_history.append(" | ".join(json.dumps(action, ensure_ascii=False) for action in parsed_actions))
        else:
            self._action_history.append("No valid action")

        current_screenshot.save(os.path.join(save_dir, "context", f"step_{str(current_step).zfill(3)}.png"))
        with open(os.path.join(save_dir, "context", f"step_{str(current_step).zfill(3)}_raw_response.txt"), "w") as f:
            f.write(final_response)
        with open(os.path.join(save_dir, "context", f"step_{str(current_step).zfill(3)}_parsed_actions.json"), "w") as f:
            json.dump(parsed_actions, f, indent=4)

        return status

    def save_conversation_history(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        if self._conversation_log:
            with open(os.path.join(save_dir, "conversation.json"), "w", encoding="utf-8") as f:
                json.dump(self._conversation_log, f, indent=2, ensure_ascii=False)
        if self._skill_invocation_log:
            with open(os.path.join(save_dir, "skill_invocations.json"), "w", encoding="utf-8") as f:
                json.dump(self._skill_invocation_log, f, indent=2, ensure_ascii=False)

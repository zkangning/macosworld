import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image

from agent.qwen import Qwen_General_Agent
from agent.skill_loader import SkillLoader
from utils.log import print_message


logger = logging.getLogger("macosworld.qwen_skill")

MAX_SKILL_BRANCH_ROUNDS = 6

QWEN_SKILL_MAIN_SYSTEM_PROMPT = """
You are a Mac desktop GUI agent that controls the computer through VNC.
At every step you receive the task, the current screenshot, recent screenshot history, and a list of optional multimodal procedural skills.

You must choose exactly one of two response modes:

1. Return an executable action block.
2. Request one skill by returning exactly `LOAD_SKILL("EXACT_SKILL_NAME")`.

Action block format:
```text
<action_name> <parameter_1> <parameter_2>
<action_name> <parameter_1> <parameter_2>
...
```

Skill request format:
```text
LOAD_SKILL("EXACT_SKILL_NAME")
```

Important action rules:
- Use only the actions listed below.
- Never invent action names.
- Never output `click`.
- `left_click`, `right_click`, `double_click`, `triple_click`, `middle_click` do not take coordinates.
- To click a specific target, first emit `move_to x y`, then emit the click action on the next line.
- All coordinates must be normalized between 0 and 1.
- In high-confidence simple situations, prefer compact 2-4 action blocks, for example:
  - `move_to ...` + `left_click`
  - `move_to ...` + `left_click` + `type_text ...`
  - `move_to ...` + `left_click` + `type_text ...` + `key_press enter`
- If you are not confident enough for a compact action block, return only the safest immediate action.
- `wait`, `fail`, and `done` must be the only command in the round.
- Only ASCII characters are allowed for `type_text`.

Available actions:
- move_to x y
- mouse_down button
- mouse_up button
- left_click
- middle_click
- right_click
- double_click
- triple_click
- drag_to x y
- scroll_down amount
- scroll_up amount
- scroll_left amount
- scroll_right amount
- type_text text
- key_press key
- wait seconds
- fail
- done

Skill usage rules:
- Skills are optional references, not ground truth.
- Only request a skill when the next immediate GUI action would benefit from procedural guidance.
- Request at most one skill per round.
- The exact skill name must come from the available skill list.
- After a skill is loaded, you will receive a temporary branch prompt containing the current screenshot plus the full skill contents.
- In that temporary branch, nested skill loading is forbidden.
- If a skill was already consulted in an earlier round, it is not still loaded. Request it again if you need it again.

Return only one fenced plaintext block and nothing else.
""".strip()


QWEN_SKILL_BRANCH_SYSTEM_PROMPT = """
You are inside a temporary skill-consultation branch for one Mac desktop step.
You receive the current screenshot, the task, and the full contents of one loaded skill.

Return exactly one executable action block.
Do not request another skill in this branch.

Action block format:
```text
<action_name> <parameter_1> <parameter_2>
<action_name> <parameter_1> <parameter_2>
...
```

Branch rules:
- Use only the supported action names.
- Never output `click`.
- Click actions never take coordinates.
- To click a specific UI target, use `move_to x y` followed by a click action.
- Prefer 2-4 action blocks in high-confidence simple situations.
- If the current screenshot conflicts with the skill text, trust the screenshot.
- Use the skill as supplemental procedural guidance, not as a script to replay blindly.
- `wait`, `fail`, and `done` must be the only command in the round.

Return only one fenced plaintext block and nothing else.
""".strip()


class QwenVLSkillAgent(Qwen_General_Agent):
    def __init__(
        self,
        model: str,
        remote_client,
        only_n_most_recent_images: int,
        max_tokens: int,
        top_p: float,
        temperature: float,
        skill_mode: str = "multimodal",
        skills_library_dir: str = "skills_library",
    ):
        super().__init__(
            model=model,
            system_prompt=QWEN_SKILL_MAIN_SYSTEM_PROMPT,
            remote_client=remote_client,
            only_n_most_recent_images=only_n_most_recent_images,
            max_tokens=max_tokens,
            top_p=top_p,
            temperature=temperature,
        )
        if skill_mode not in {"text_only", "multimodal"}:
            raise ValueError("skill_mode must be 'text_only' or 'multimodal'")
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
        logger.info("[QwenSkill] Task skills resolved: %s", self._task_skill_names)

    def _available_skills_text(self) -> str:
        if not self._task_skill_metadatas:
            return "- None"
        return "\n".join(
            f"- {Path(meta.directory).name}: {(meta.description or '').strip() or '(no description)'}"
            for meta in self._task_skill_metadatas
        )

    def _consulted_skills_text(self) -> str:
        return ", ".join(sorted(self._consulted_skills)) if self._consulted_skills else "None"

    def _call_messages(self, system_prompt: str, user_content: List[dict]) -> str:
        request_kwargs = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "top_p": self.top_p,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
        }
        if self.thinking_mode == "on":
            request_kwargs["extra_body"] = {"enable_thinking": True}
        elif self.thinking_mode == "off":
            request_kwargs["extra_body"] = {"enable_thinking": False}

        response = self.prompt_client.chat.completions.create(**request_kwargs)
        usage = getattr(response, "usage", None)
        usage_dict = usage.model_dump() if usage is not None and hasattr(usage, "model_dump") else {}
        self.token_usage.append(usage_dict)
        self.total_input_tokens += usage_dict.get("prompt_tokens", 0)
        self.total_output_tokens += usage_dict.get("completion_tokens", 0)

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
        if text == "":
            raise RuntimeError(f"Empty Qwen skill response for model {self.model}.")
        return text

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

    def _construct_main_user_prompt(
        self,
        task: str,
        screenshots: List[Image.Image],
        round_feedback: Optional[List[str]] = None,
    ) -> List[dict]:
        prompt_parts = []
        if len(screenshots) == 1:
            prompt_parts.extend(
                [
                    "Task: " + task,
                    "\nAvailable skills:\n" + self._available_skills_text(),
                    "\nPreviously consulted skills (not currently loaded): " + self._consulted_skills_text(),
                    "\nPrevious action summaries:\n" + ("\n".join(self._action_history[-5:]) if self._action_history else "None"),
                ]
            )
            if round_feedback:
                prompt_parts.append("\nFeedback from previous invalid response(s):\n" + "\n".join(f"- {x}" for x in round_feedback))
            prompt_parts.extend(["\nScreenshot: ", screenshots[0]])
            return self.format_interleaved_message(prompt_parts)

        prompt_parts.extend(
            [
                "Task: " + task,
                "\nAvailable skills:\n" + self._available_skills_text(),
                "\nPreviously consulted skills (not currently loaded): " + self._consulted_skills_text(),
                "\nPrevious action summaries:\n" + ("\n".join(self._action_history[-5:]) if self._action_history else "None"),
            ]
        )
        if round_feedback:
            prompt_parts.append("\nFeedback from previous invalid response(s):\n" + "\n".join(f"- {x}" for x in round_feedback))
        prompt_parts.extend(
            [
                "\nRolling window of historical screenshots in chronological order: ",
                *screenshots[:-1],
                "\nCurrent screenshot: ",
                screenshots[-1],
            ]
        )
        return self.format_interleaved_message(prompt_parts)

    def _construct_branch_user_prompt(
        self,
        task: str,
        current_screenshot: Image.Image,
        skill_name: str,
        full_skill: Dict,
    ) -> List[dict]:
        content = full_skill["content"]
        elements: List = [
            "Task: " + task,
            "\nCurrent screenshot: ",
            current_screenshot,
            (
                "\nLoaded skill reference: "
                f"{content.name} ({skill_name})\n\n"
                f"{content.text}"
            ),
        ]
        if self.skill_mode == "multimodal":
            for filename, b64_data, mime_type in full_skill.get("images", []):
                elements.append(f"\n[Skill visual reference: {skill_name}/{filename}]")
                elements.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{b64_data}", "detail": "high"},
                    }
                )
        formatted = []
        for element in elements:
            if isinstance(element, dict):
                formatted.append(element)
            elif isinstance(element, str):
                formatted.append({"type": "text", "text": element})
            elif isinstance(element, Image.Image):
                formatted.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": self._pil_to_b64(element), "detail": "high"},
                    }
                )
        return formatted

    @staticmethod
    def _pil_to_b64(image: Image.Image) -> str:
        from agent.llm_utils import pil_to_b64
        return pil_to_b64(image, add_prefix=True)

    def _append_conversation_log(self, entry_type: str, payload: dict):
        self._conversation_log.append({"entry_type": entry_type, **payload})

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
        user_content = self._construct_branch_user_prompt(
            task=task,
            current_screenshot=current_screenshot,
            skill_name=trigger_skill_name,
            full_skill=full_skill,
        )
        branch_response = self._call_messages(QWEN_SKILL_BRANCH_SYSTEM_PROMPT, user_content)
        parsed_actions = self.parse_agent_output(branch_response)

        self._skill_invocation_log.append(
            {
                "step": current_step,
                "trigger_skill_name": trigger_skill_name,
                "main_response": main_response,
                "branch_response": branch_response,
                "parsed_actions": parsed_actions,
                "loaded_image_count": len(full_skill.get("images", [])) if self.skill_mode == "multimodal" else 0,
                "success": len(parsed_actions) > 0,
            }
        )
        return branch_response, parsed_actions, None if parsed_actions else "Loaded skill branch returned no valid action."

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

        for round_idx in range(MAX_SKILL_BRANCH_ROUNDS):
            user_content = self._construct_main_user_prompt(task=task, screenshots=self.screenshots, round_feedback=round_feedback)
            main_response = self._call_messages(QWEN_SKILL_MAIN_SYSTEM_PROMPT, user_content)
            skill_request = self._extract_skill_request(main_response)

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
        with open(
            os.path.join(save_dir, "context", f"step_{str(current_step).zfill(3)}_parsed_actions.json"),
            "w",
        ) as f:
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

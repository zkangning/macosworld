import re


GENERAL_ACTION_OUTPUT_RULES = """

Additional output rules for any model using this agent:
- Return exactly one plaintext code block and nothing else. Do not output explanations, reasoning, JSON, XML, HTML, Markdown lists, or function-call wrappers.
- Never use tags such as <action_name>, <parameter_1>, <function_calls>, or <invoke>.
- For click-like actions, both of the following forms are legal:
  1. move_to x y
     left_click
  2. left_click x y
  The same shorthand is allowed for middle_click, right_click, double_click, and triple_click. In the shorthand form, x and y must still be normalized floats between 0 and 1.
- Use `enter`, not `return`.
- Use `esc`, not `escape`.
- Only output `done` when the task objective is visibly satisfied in the current screenshot.
- If the screenshot shows the macOS lock screen and the password field is visible, unlock first: click the password field, type `000000`, press `enter`, then wait briefly. Do not keep typing the password after the desktop or target app is visible.
"""


VALID_ACTIONS = {
    "move_to",
    "mouse_down",
    "mouse_up",
    "left_click",
    "middle_click",
    "right_click",
    "double_click",
    "triple_click",
    "drag_to",
    "scroll_down",
    "scroll_up",
    "scroll_left",
    "scroll_right",
    "type_text",
    "key_press",
    "wait",
    "fail",
    "done",
}

POSITIONAL_CLICK_ACTIONS = {
    "left_click",
    "middle_click",
    "right_click",
    "double_click",
    "triple_click",
}

INLINE_ACTIONS = sorted(
    VALID_ACTIONS - {"wait", "fail", "done"},
    key=len,
    reverse=True,
)

KEY_ALIASES = {
    "return": "enter",
    "escape": "esc",
    "delete": "del",
    "control": "ctrl",
}


def _clean_numeric_token(token: str) -> str:
    token = token.strip()
    if "=" in token:
        token = token.split("=")[-1]
    return token.strip().strip(",.;:()[]{}")


def _parse_normalized_float(token: str) -> float:
    cleaned = _clean_numeric_token(token)
    value = float(cleaned)
    if 0.0 <= value <= 1.0:
        return value
    if "." not in cleaned and cleaned.lstrip("+-").isdigit():
        digits = cleaned.lstrip("+-")
        scaled = value / (10 ** len(digits))
        if 0.0 <= scaled <= 1.0:
            return scaled
    raise ValueError(f"Expected normalized float between 0 and 1, got {token}")


def _parse_plain_float(token: str) -> float:
    return float(_clean_numeric_token(token))


def _normalize_key_name(token: str) -> str:
    key = _clean_numeric_token(token)
    return KEY_ALIASES.get(key.lower(), key)


def _normalize_agent_output(agent_output: str) -> str:
    normalized = agent_output.strip()
    normalized = re.sub(
        r"<action_name>\s*([^<]+?)\s*</action_name>",
        lambda m: m.group(1).strip(),
        normalized,
        flags=re.IGNORECASE,
    )
    normalized = re.sub(
        r"<parameter_\d+>\s*([^<]*?)\s*</parameter_\d+>",
        lambda m: " " + m.group(1).strip(),
        normalized,
        flags=re.IGNORECASE,
    )
    normalized = re.sub(r"<[^>]+>", "\n", normalized)
    normalized = normalized.replace("```", "\n").replace("`", "\n").replace("|", "\n")
    normalized = re.sub(
        rf"(?<!^)(?<!\n)(?=(?:{'|'.join(INLINE_ACTIONS)})\b)",
        "\n",
        normalized,
    )
    return normalized


def parse_gui_actions(agent_output: str) -> list[dict]:
    actions = []
    lines = _normalize_agent_output(agent_output).splitlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            tokens = line.split()
            if not tokens:
                continue
            action_cmd = tokens[0].strip().lower()
            if action_cmd not in VALID_ACTIONS:
                continue

            action_dict = {"action": action_cmd}

            if action_cmd in {"move_to", "drag_to"}:
                if len(tokens) < 3:
                    continue
                action_dict["x"] = _parse_normalized_float(tokens[1])
                action_dict["y"] = _parse_normalized_float(tokens[2])
            elif action_cmd in {"mouse_down", "mouse_up"}:
                if len(tokens) < 2:
                    continue
                action_dict["button"] = _clean_numeric_token(tokens[1]).lower()
            elif action_cmd in {"scroll_down", "scroll_up", "scroll_left", "scroll_right"}:
                if len(tokens) < 2:
                    continue
                action_dict["amount"] = _parse_normalized_float(tokens[1])
            elif action_cmd == "wait":
                if len(tokens) < 2:
                    continue
                action_dict["seconds"] = _parse_plain_float(tokens[1])
            elif action_cmd == "type_text":
                raw_text = line[len(tokens[0]):]
                if raw_text.strip() == "":
                    text = raw_text
                else:
                    text = " ".join(raw_text.split())
                action_dict["text"] = text
            elif action_cmd == "key_press":
                if len(tokens) < 2:
                    continue
                action_dict["key"] = _normalize_key_name(tokens[1])
            elif action_cmd in POSITIONAL_CLICK_ACTIONS and len(tokens) >= 3:
                try:
                    x = _parse_normalized_float(tokens[1])
                    y = _parse_normalized_float(tokens[2])
                    actions.append({"action": "move_to", "x": x, "y": y})
                except Exception:
                    pass

            actions.append(action_dict)
        except Exception:
            continue

    return actions

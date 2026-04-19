import time
from typing import Any, Optional

from PIL import Image
from io import BytesIO
import os
import json
import base64
import requests
import httpx
from openai import OpenAI

try:
    from google.api_core.exceptions import InvalidArgument
    from vertexai.preview.generative_models import Image as VertexImage
    from vertexai.preview.generative_models import (
        GenerativeModel,
        HarmBlockThreshold,
        HarmCategory,
        Content,
        Part
    )
    _VERTEXAI_AVAILABLE = True
except ModuleNotFoundError:
    InvalidArgument = Exception
    VertexImage = None
    GenerativeModel = None
    Content = None
    Part = None
    _VERTEXAI_AVAILABLE = False

    class _DummyHarmBlockThreshold:
        BLOCK_ONLY_HIGH = "BLOCK_ONLY_HIGH"

    class _DummyHarmCategory:
        HARM_CATEGORY_UNSPECIFIED = "HARM_CATEGORY_UNSPECIFIED"
        HARM_CATEGORY_HATE_SPEECH = "HARM_CATEGORY_HATE_SPEECH"
        HARM_CATEGORY_DANGEROUS_CONTENT = "HARM_CATEGORY_DANGEROUS_CONTENT"
        HARM_CATEGORY_HARASSMENT = "HARM_CATEGORY_HARASSMENT"
        HARM_CATEGORY_SEXUALLY_EXPLICIT = "HARM_CATEGORY_SEXUALLY_EXPLICIT"

    HarmBlockThreshold = _DummyHarmBlockThreshold
    HarmCategory = _DummyHarmCategory

from utils.VNCClient import VNCClient_SSH
from utils.log import print_message
from utils.timeout import timeout
from agent.llm_utils import pil_to_b64
from agent.gui_action_parser import GENERAL_ACTION_OUTPUT_RULES, parse_gui_actions



GEMINI_SYSTEM_PROMPT = """
You are an agent that performs Mac desktop computer tasks by controlling mouse and keyboard through VNC. For each step, you will receive a screenshot observation of the computer screen and should predict the next action.

Your output must be raw text commands with the following structure:
```
<action_name> <parameter_1> <parameter_2>
<action_name> <parameter_1> <parameter_2>
...
```

For example:
```
move_to 0.25 0.5
key_press command-c
left_click
```

Use only the actions listed below. Do not invent new action names.

Available actions and their parameters:

1. Mouse Actions:
- "move_to": Move cursor to normalized coordinates
  Required params: {"x": float 0-1, "y": float 0-1}
  
- "left_click": Perform left mouse click
  No params required
  
- "middle_click": Perform middle mouse click
  No params required
  
- "right_click": Perform right mouse click
  No params required
  
- "double_click": Perform double left click
  No params required

- "triple_click": Perform triple left click
  No params required

- "drag_to": Drag with the left mouse button to a specified coordinate.
  Required params: {"x": float 0-1, "y": float 0-1}

- "mouse_down": Press and hold a mouse button.
  Required params: {"button": string ("left", "middle", "right")}

- "mouse_up": Release a mouse button.
  Required params: {"button": string ("left", "middle", "right")}

- "scroll_down": Scroll down by proportion of screen height
  Required params: {"amount": float 0-1}
  
- "scroll_up": Scroll up by proportion of screen height
  Required params: {"amount": float 0-1}

- "scroll_left": Scroll left by proportion of screen width
  Required params: {"amount": float 0-1}

- "scroll_right": Scroll right by proportion of screen width
  Required params: {"amount": float 0-1}

2. Keyboard Actions:
- "type_text": Type ASCII text
  Required params: {"text": string}
  Everything after `type_text ` will be parsed as parameter 1, including spaces. No need to escape any characters.
  
- "key_press": Press a key or key combination.
  Required params: {"key": string}
  Available keys: ctrl, command, option, backspace, tab, enter, esc, del, left, up, right, down, or single ASCII characters
  When pressing a combination of keys simultaneously, connect the keys using `-`, for example, `command-c` or `ctrl-alt-del`

3. Control Actions:
- "wait": Wait for specified seconds
  Required params: {"seconds": float}
  
- "fail": Indicate task cannot be completed
  No params required
  
- "done": Indicate task is already finished
  No params required

Important Notes:
- Your username is "ec2-user" and password is "000000"
- All coordinates (x,y) should be normalized between 0 and 1
- All scroll amounts should be normalized between 0 and 1
- Only ASCII characters are allowed for text input
- The control commands (wait, fail, done) must be the only command issued in a round. If one of these commands is used, no other actions should be provided alongside it.
- Return only the actions in a backtick-wrapped plaintext code block, one line per action, no other text
"""
GEMINI_SYSTEM_PROMPT += GENERAL_ACTION_OUTPUT_RULES
GLM_SYSTEM_PROMPT = GEMINI_SYSTEM_PROMPT

GEMINI_SAFETY_CONFIG = {
    HarmCategory.HARM_CATEGORY_UNSPECIFIED: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
}

def pil_to_vertex(img: Image.Image) -> str:
    if not _VERTEXAI_AVAILABLE:
        raise RuntimeError("Vertex AI dependencies are not installed.")
    with BytesIO() as image_buffer:
        img.save(image_buffer, format="PNG")
        byte_data = image_buffer.getvalue()
        img_vertex = VertexImage.from_bytes(byte_data)
    return img_vertex


def pil_to_gateway_inline_data(img: Image.Image) -> dict:
    with BytesIO() as image_buffer:
        img.save(image_buffer, format="PNG")
        byte_data = image_buffer.getvalue()
    return {
        "mimeType": "image/png",
        "data": base64.b64encode(byte_data).decode("utf-8")
    }


class OpenAILikeGeminiGatewayClient:
    def __init__(self, api_key: str, base_url: str):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    def generate_content(
        self,
        parts: list,
        system_text: str,
        temperature: float,
        max_output_tokens: int,
        top_p: float,
        seed: int,
        thinking_level: str,
        include_thoughts: bool,
    ) -> dict:
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": parts
                }
            ],
            "systemInstruction": {
                "parts": [
                    {
                        "text": system_text
                    }
                ]
            },
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_output_tokens,
                "topP": top_p,
                "seed": seed,
            }
        }

        if thinking_level != "NONE":
            payload["generationConfig"]["thinkingConfig"] = {
                "thinkingLevel": thinking_level,
                "includeThoughts": include_thoughts,
            }

        response = requests.post(
            f"{self.base_url}/v1:generateContent",
            headers={
                "api-key": self.api_key,
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=120,
        )
        response.raise_for_status()
        return response.json()


class OpenAICompatibleVisionClient:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        proxy_url: str = "",
        default_headers: Optional[dict[str, str]] = None,
    ):
        http_client = httpx.Client(proxy=proxy_url) if proxy_url else None
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url.rstrip("/"),
            http_client=http_client,
            default_headers=default_headers or None,
        )

    def chat_completions(
        self,
        model: str,
        system_text: str,
        content: list,
        temperature: float,
        max_output_tokens: int,
        top_p: float,
        extra_body: Optional[dict[str, Any]] = None,
    ):
        request_kwargs = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_text},
                {"role": "user", "content": content},
            ],
            "temperature": temperature,
            "max_tokens": max_output_tokens,
            "top_p": top_p,
            "stream": False,
        }
        if extra_body:
            request_kwargs["extra_body"] = extra_body
        return self.client.chat.completions.create(**request_kwargs)

class Gemini_General_Agent:
    def __init__(
        self,
        model: str,
        system_prompt: str,
        remote_client: VNCClient_SSH,
        only_n_most_recent_images: int,
        max_tokens: int,
        top_p: float,
        temperature: float,
        safety_config: dict,
    ):
        self.model = model
        self.safety_config = safety_config
        self.system_prompt = system_prompt
        self.remote_client = remote_client

        self.max_tokens = max_tokens
        self.only_n_most_recent_images = only_n_most_recent_images
        self.top_p = top_p
        self.temperature = temperature
        
        self.messages = None
        self.screenshots = []

        self.token_usage = []
        self.total_prompt_tokens = 0
        self.total_candidates_tokens = 0

        self.gateway_base_url = os.environ.get("GEMINI_BASE_URL", "").strip()
        self.gateway_api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        self.gateway_thinking_level = os.environ.get("GEMINI_THINKING_LEVEL", "NONE").strip() or "NONE"
        self.gateway_include_thoughts = os.environ.get("GEMINI_INCLUDE_THOUGHTS", "false").strip().lower() == "true"
        self.gateway_seed = int(os.environ.get("GEMINI_SEED", "0"))

        if self.gateway_base_url and self.gateway_api_key:
            self.prompt_client = OpenAILikeGeminiGatewayClient(
                api_key=self.gateway_api_key,
                base_url=self.gateway_base_url,
            )
            self.use_gateway = True
        else:
            if not _VERTEXAI_AVAILABLE:
                raise RuntimeError(
                    "Gemini Vertex path requires vertexai to be installed, or set GEMINI_BASE_URL and GEMINI_API_KEY."
                )
            self.prompt_client = GenerativeModel(model)
            self.use_gateway = False

    def construct_user_prompt(self, task: str, screenshots: list):
        if len(screenshots) == 0:
            raise ValueError(f'Empty list of screenshots.')
        if len(screenshots) == 1:
            return [
                self.system_prompt,
                f'Task: {task}\nScreenshot: ',
                screenshots[0]
            ]
        return [
            self.system_prompt,
            f'Task: {task}\nRolling window of historical screenshots in chronological order: ',
            *screenshots[:-1],
            '\nCurrent screenshot: ',
            screenshots[-1]
        ]
    
    def call_agent(self, task: str):
        prompt = self.construct_user_prompt(task = task, screenshots = self.screenshots)

        if self.use_gateway:
            gateway_prompt = prompt[1:] if len(prompt) > 0 and prompt[0] == self.system_prompt else prompt
            request_parts = []
            for element in gateway_prompt:
                if isinstance(element, str):
                    request_parts.append({"text": element})
                elif isinstance(element, Image.Image):
                    request_parts.append({"inlineData": pil_to_gateway_inline_data(element)})
                else:
                    raise TypeError(f"Unsupported prompt element type for Gemini gateway: {type(element)}")

            response = self.prompt_client.generate_content(
                parts=request_parts,
                system_text=self.system_prompt,
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
                    "candidates_token_count": candidate_tokens
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
                raise RuntimeError(f"Empty Gemini gateway response: {json.dumps(response, ensure_ascii=False)[:1000]}")
            return response_text

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

        # Count token usage
        self.token_usage.append(
            {
                "prompt_token_count": response.usage_metadata.prompt_token_count,
                "candidates_token_count": response.usage_metadata.candidates_token_count
            }
        )
        self.total_prompt_tokens += response.usage_metadata.prompt_token_count
        self.total_candidates_tokens += response.usage_metadata.candidates_token_count

        response_text = response.candidates[0].content.parts[0].text
        return response_text
    
    def parse_agent_output(self, agent_output):
        return parse_gui_actions(agent_output)
    
    def execute_actions(self, actions):
        """
        Execute a list of parsed actions.
        
        For each action, the corresponding VNCClient_SSH method is called.
        If a 'fail' or 'done' command is encountered, execution stops and the function returns immediately.
        
        Returns a tuple (status, actions_executed) where status is one of:
          - "done" if a 'done' command was executed,
          - "fail" if a 'fail' command was executed,
          - "unfinished" if neither was encountered.
        
        Note: This function does not use error handling; any errors during execution will propagate.
        """
        status = "unfinished"
        for action in actions:
            act = action.get("action")
            time.sleep(self.remote_client.action_interval_seconds)
            if act == "move_to":
                self.remote_client.move_to(action["x"], action["y"])
            elif act == "mouse_down":
                self.remote_client.mouse_down(action["button"])
            elif act == "mouse_up":
                self.remote_client.mouse_up(action["button"])
            elif act == "left_click":
                self.remote_client.left_click()
            elif act == "middle_click":
                self.remote_client.middle_click()
            elif act == "right_click":
                self.remote_client.right_click()
            elif act == "double_click":
                self.remote_client.double_click()
            elif act == "triple_click":
                self.remote_client.triple_click()
            elif act == "drag_to":
                self.remote_client.drag_to(action["x"], action["y"])
            elif act == "scroll_down":
                self.remote_client.scroll_down(action["amount"])
            elif act == "scroll_up":
                self.remote_client.scroll_up(action["amount"])
            elif act == "scroll_left":
                self.remote_client.scroll_left(action["amount"])
            elif act == "scroll_right":
                self.remote_client.scroll_right(action["amount"])
            elif act == "type_text":
                self.remote_client.type_text(action["text"])
            elif act == "key_press":
                self.remote_client.key_press(action["key"])
            elif act == "wait":
                time.sleep(action["seconds"])
            elif act == "fail":
                status = "fail"
                return status, actions
            elif act == "done":
                status = "done"
                return status, actions
        return status, actions

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
        # with timeout(task_step_timeout):
        # Capture screenshot
        print_message(title = f'Task {task_id}/{env_language}/{task_language} Step {current_step}/{max_steps}', content = 'Capturing screenshot...')
        current_screenshot = self.remote_client.capture_screenshot()
        if self.use_gateway:
            self.screenshots.append(current_screenshot.copy())
        else:
            self.screenshots.append(pil_to_vertex(current_screenshot))
        if self.only_n_most_recent_images > 0:
            self.screenshots = self.screenshots[-self.only_n_most_recent_images:]
        
        # Prediction
        print_message(title = f'Task {task_id}/{env_language}/{task_language} Step {current_step}/{max_steps}', content = 'Calling GUI agent...')
        raw_response = self.call_agent(task = task)

        # Action
        print_message(title = f'Task {task_id}/{env_language}/{task_language} Step {current_step}/{max_steps}', content = 'Actuating...')
        parsed_actions = self.parse_agent_output(raw_response)
        status, _ = self.execute_actions(parsed_actions)

        # Save current_screenshot
        current_screenshot.save(os.path.join(save_dir, 'context', f'step_{str(current_step).zfill(3)}.png'))

        # Save raw_response
        with open(os.path.join(save_dir, 'context', f'step_{str(current_step).zfill(3)}_raw_response.txt'), 'w') as f:
            f.write(raw_response)

        # Dump parsed_actions
        with open(os.path.join(save_dir, 'context', f'step_{str(current_step).zfill(3)}_parsed_actions.json'), 'w') as f:
            json.dump(parsed_actions, f, indent=4)

        return status
    
    def save_conversation_history(self, save_dir: str):
        pass


class Gemini_OpenAICompat_Agent(Gemini_General_Agent):
    def __init__(
        self,
        model: str,
        system_prompt: str,
        remote_client: VNCClient_SSH,
        only_n_most_recent_images: int,
        max_tokens: int,
        top_p: float,
        temperature: float,
        base_url_env: str,
        api_key_env: str,
        thinking_mode_env: str,
        proxy_env: str = "",
    ):
        self.model = model
        self.safety_config = {}
        self.system_prompt = system_prompt
        self.remote_client = remote_client

        self.max_tokens = max_tokens
        self.only_n_most_recent_images = only_n_most_recent_images
        self.top_p = top_p
        self.temperature = temperature

        self.messages = None
        self.screenshots = []

        self.token_usage = []
        self.total_prompt_tokens = 0
        self.total_candidates_tokens = 0

        self.gateway_base_url = os.environ.get(base_url_env, "").strip()
        self.gateway_api_key = os.environ.get(api_key_env, "").strip()
        self.gateway_proxy_url = os.environ.get(proxy_env, "").strip() if proxy_env else ""
        self.gateway_thinking_mode = os.environ.get(thinking_mode_env, "auto").strip().lower() or "auto"
        self.is_openrouter = "openrouter.ai" in self.gateway_base_url.lower()

        default_headers = {}
        openrouter_site_url = os.environ.get("OPENROUTER_SITE_URL", "").strip()
        openrouter_app_name = os.environ.get("OPENROUTER_APP_NAME", "").strip()
        if self.is_openrouter and openrouter_site_url:
            default_headers["HTTP-Referer"] = openrouter_site_url
        if self.is_openrouter and openrouter_app_name:
            default_headers["X-Title"] = openrouter_app_name

        if self.gateway_base_url == "" or self.gateway_api_key == "":
            raise ValueError(
                f"Gemini-style OpenAI-compatible agent requires {base_url_env} and {api_key_env}."
            )

        self.prompt_client = OpenAICompatibleVisionClient(
            api_key=self.gateway_api_key,
            base_url=self.gateway_base_url,
            proxy_url=self.gateway_proxy_url,
            default_headers=default_headers,
        )
        self.use_gateway = False

    def _format_openai_compat_content(self, elements: list):
        formatted = []
        for element in elements:
            if isinstance(element, str):
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
                raise TypeError(f"Unsupported prompt element type for OpenAI-compatible vision client: {type(element)}")
        return formatted

    def call_agent(self, task: str):
        prompt = self.construct_user_prompt(task=task, screenshots=self.screenshots)
        prompt_body = prompt[1:] if len(prompt) > 0 and prompt[0] == self.system_prompt else prompt
        formatted_user_prompt = self._format_openai_compat_content(prompt_body)

        extra_body = None
        if self.is_openrouter:
            if self.gateway_thinking_mode == "on":
                extra_body = {"reasoning": {"enabled": True}}
            elif self.gateway_thinking_mode == "off":
                extra_body = {"reasoning": {"enabled": False}}
        else:
            if self.gateway_thinking_mode == "on":
                extra_body = {"enable_thinking": True}
            elif self.gateway_thinking_mode == "off":
                extra_body = {"enable_thinking": False}

        response = self.prompt_client.chat_completions(
            model=self.model,
            system_text=self.system_prompt,
            content=formatted_user_prompt,
            temperature=self.temperature,
            max_output_tokens=self.max_tokens,
            top_p=self.top_p,
            extra_body=extra_body,
        )

        usage = getattr(response, "usage", None)
        usage_dict = usage.model_dump() if usage is not None and hasattr(usage, "model_dump") else {}
        prompt_tokens = usage_dict.get("prompt_tokens", 0)
        completion_tokens = usage_dict.get("completion_tokens", 0)
        self.token_usage.append(
            {
                "prompt_token_count": prompt_tokens,
                "candidates_token_count": completion_tokens,
            }
        )
        self.total_prompt_tokens += prompt_tokens
        self.total_candidates_tokens += completion_tokens

        message = response.choices[0].message
        content = message.content
        if isinstance(content, str):
            response_text = content
        elif isinstance(content, list):
            response_text = "\n".join(
                item.get("text", "")
                for item in content
                if isinstance(item, dict) and "text" in item
            ).strip()
        else:
            response_text = str(content or "").strip()

        if response_text == "":
            raise RuntimeError(f"Empty OpenAI-compatible vision response for model {self.model}.")

        return response_text

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
        print_message(title = f'Task {task_id}/{env_language}/{task_language} Step {current_step}/{max_steps}', content = 'Capturing screenshot...')
        current_screenshot = self.remote_client.capture_screenshot()
        self.screenshots.append(current_screenshot.copy())
        if self.only_n_most_recent_images > 0:
            self.screenshots = self.screenshots[-self.only_n_most_recent_images:]

        print_message(title = f'Task {task_id}/{env_language}/{task_language} Step {current_step}/{max_steps}', content = 'Calling GUI agent...')
        raw_response = self.call_agent(task = task)

        print_message(title = f'Task {task_id}/{env_language}/{task_language} Step {current_step}/{max_steps}', content = 'Actuating...')
        parsed_actions = self.parse_agent_output(raw_response)
        status, _ = self.execute_actions(parsed_actions)

        current_screenshot.save(os.path.join(save_dir, 'context', f'step_{str(current_step).zfill(3)}.png'))
        with open(os.path.join(save_dir, 'context', f'step_{str(current_step).zfill(3)}_raw_response.txt'), 'w') as f:
            f.write(raw_response)
        with open(os.path.join(save_dir, 'context', f'step_{str(current_step).zfill(3)}_parsed_actions.json'), 'w') as f:
            json.dump(parsed_actions, f, indent=4)

        return status

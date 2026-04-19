from openai import OpenAI
import os
from utils.VNCClient import VNCClient_SSH
from utils.log import print_message
from agent.llm_utils import pil_to_b64
from PIL import Image
import json
from utils.timeout import timeout
import time
import httpx

from agent.gui_action_parser import GENERAL_ACTION_OUTPUT_RULES, parse_gui_actions
from agent.llm_utils import construct_user_prompt, format_interleaved_message

GPT_SYSTEM_PROMPT = """
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

- "scroll_left": Scroll up by proportion of screen width
  Required params: {"amount": float 0-1}

- "scroll_right": Scroll up by proportion of screen width
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
GPT_SYSTEM_PROMPT += GENERAL_ACTION_OUTPUT_RULES

class OpenAI_GUI_Agent:
    def __init__(self, model, system_prompt):
        self.prompt_client = OpenAI()
        self.model = model
        self.system_prompt = system_prompt

    def __call__(self, task, screenshots):
        user_prompt = construct_user_prompt(task, screenshots)
        formatted_user_prompt = format_interleaved_message(user_prompt)

        messages = [
            {
                "role": "system",
                "content": self.system_prompt
            },
            {
                "role": "user",
                "content": formatted_user_prompt
            }
        ]

        response = self.prompt_client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
        ).choices[0].message.content

        extended_messages = messages + [{"role": "assistant", "content": response}]

        return response, extended_messages

class OpenAI_General_Agent:
    def __init__(
        self, 
        model: str, 
        system_prompt: str, 
        remote_client: VNCClient_SSH,
        screenshot_rolling_window: int,
        top_p: float,
        temperature: float,
    ):
        proxy_url = os.environ.get("OPENAI_PROXY_URL")

        self.prompt_client = OpenAI() if proxy_url is None or proxy_url == "" else OpenAI(http_client=httpx.Client(proxy=proxy_url))
        self.model = model
        self.system_prompt = system_prompt
        self.remote_client = remote_client
        self.screenshot_rolling_window = screenshot_rolling_window
        self.top_p = top_p
        self.temperature = temperature

        self.messages = None
        self.screenshots = []

    def __call__(self, task, screenshots):
        user_prompt = self.construct_user_prompt(task, screenshots)
        formatted_user_prompt = self.format_interleaved_message(user_prompt)

        messages = [
            {
                "role": "system",
                "content": self.system_prompt
            },
            {
                "role": "user",
                "content": formatted_user_prompt
            }
        ]

        response = self.prompt_client.chat.completions.create(
            model=self.model,
            messages=messages,
            top_p=self.top_p,
            temperature=self.temperature
        ).choices[0].message.content

        extended_messages = messages + [{"role": "assistant", "content": response}]

        return response, extended_messages
    
    def format_interleaved_message(self, elements, b64_image_add_prefix = True):
        formatted_list = []
        for element in elements:
            if isinstance(element, str):
                formatted_list.append({"type": "text", "text": element})
            elif isinstance(element, Image.Image):
                formatted_list.append({
                    "type": "image_url",
                    "image_url": {
                        "url": pil_to_b64(element, add_prefix = b64_image_add_prefix),
                        "detail": "high"
                    }
                })
        return formatted_list

    def construct_user_prompt(self, task: str, screenshots: list):
        if len(screenshots) == 0:
            raise ValueError(f'Empty list of screenshots.')
        if len(screenshots) == 1:
            return [
                f'Task: {task}\nScreenshot: ',
                screenshots[0]
            ]
        return [
            f'Task: {task}\nRolling window of historical screenshots in chronological order: ',
            *screenshots[:-1],
            screenshots[-1]
        ]
    
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
        with timeout(task_step_timeout):
            # Capture screenshot
            print_message(title = f'Task {task_id}/{env_language}/{task_language} Step {current_step}/{max_steps}', content = 'Capturing screenshot...')
            current_screenshot = self.remote_client.capture_screenshot()
            self.screenshots.append(current_screenshot)
            self.screenshots = self.screenshots[-self.screenshot_rolling_window:]

            # Prediction
            print_message(title = f'Task {task_id}/{env_language}/{task_language} Step {current_step}/{max_steps}', content = 'Calling GUI agent...')
            raw_response, messages = self(task = task, screenshots = self.screenshots)

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

        # print_message(title = f'Task {task_id}/{env_language}/{task_language} Step {current_step}/{max_steps}', content = f'Status: {status}')

        return status

    def save_conversation_history(self, save_dir: str):
        pass

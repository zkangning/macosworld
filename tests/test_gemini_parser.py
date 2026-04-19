from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.gemini import Gemini_General_Agent
from agent.openai import OpenAI_General_Agent
from agent.qwen import Qwen_General_Agent
from agent.gui_action_parser import parse_gui_actions


def make_agent():
    return Gemini_General_Agent.__new__(Gemini_General_Agent)


def test_click_shorthand_is_rewritten_to_move_then_click():
    agent = make_agent()
    parsed = agent.parse_agent_output(
        """
        ```
        left_click 0.25 0.75
        double_click 0.5 0.125
        ```
        """
    )

    assert parsed == [
        {"action": "move_to", "x": 0.25, "y": 0.75},
        {"action": "left_click"},
        {"action": "move_to", "x": 0.5, "y": 0.125},
        {"action": "double_click"},
    ]


def test_xml_wrapped_actions_are_recovered():
    agent = make_agent()
    parsed = agent.parse_agent_output(
        """
        <action_name>key_press</action_name> <parameter_1>Return</parameter_1>
        <action_name>left_click</action_name> <parameter_1>0.508</parameter_1> <parameter_2>0.915</parameter_2>
        """
    )

    assert parsed == [
        {"action": "key_press", "key": "enter"},
        {"action": "move_to", "x": 0.508, "y": 0.915},
        {"action": "left_click"},
    ]


def test_narration_is_ignored_but_actions_are_kept():
    agent = make_agent()
    parsed = agent.parse_agent_output(
        """
        Looking at the current state, I should focus the app first.
        key_press command-space
        type_text Dictionary
        key_press enter
        """
    )

    assert parsed == [
        {"action": "key_press", "key": "command-space"},
        {"action": "type_text", "text": "Dictionary"},
        {"action": "key_press", "key": "enter"},
    ]


def test_integer_like_coordinates_are_normalized():
    agent = make_agent()
    parsed = agent.parse_agent_output(
        """
        move_to 470 87
        left_click
        """
    )

    assert parsed == [
        {"action": "move_to", "x": 0.47, "y": 0.87},
        {"action": "left_click"},
    ]


def test_pipe_separated_output_is_supported():
    agent = make_agent()
    parsed = agent.parse_agent_output("``` | left_click 0.508 0.915 | key_press Return | ```")

    assert parsed == [
        {"action": "move_to", "x": 0.508, "y": 0.915},
        {"action": "left_click"},
        {"action": "key_press", "key": "enter"},
    ]


def test_shared_parser_matches_all_agent_entrypoints():
    raw = "``` | left_click 0.508 0.915 | key_press Return | ```"
    expected = [
        {"action": "move_to", "x": 0.508, "y": 0.915},
        {"action": "left_click"},
        {"action": "key_press", "key": "enter"},
    ]

    assert parse_gui_actions(raw) == expected
    assert Gemini_General_Agent.__new__(Gemini_General_Agent).parse_agent_output(raw) == expected
    assert OpenAI_General_Agent.__new__(OpenAI_General_Agent).parse_agent_output(raw) == expected
    assert Qwen_General_Agent.__new__(Qwen_General_Agent).parse_agent_output(raw) == expected

import os

from constants import SCREEN_WIDTH, SCREEN_HEIGHT


def _get_agent_image_window(default: int = 3) -> int:
    raw = os.environ.get("MACOSWORLD_AGENT_IMAGE_WINDOW", str(default)).strip()
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _skills_enabled() -> bool:
    value = os.environ.get("MACOSWORLD_ENABLE_SKILLS", "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _qwen_agent_arch() -> str:
    return os.environ.get("MACOSWORLD_QWEN_AGENT_ARCH", "").strip().lower()

def get_gui_agent(gui_agent_name, remote_client):
    image_window = _get_agent_image_window()
    if gui_agent_name in {"gemini-skill-text-inline", "gemini-skill-text-branch", "gemini-skill-mm-branch"}:
        from agent.gemini import GEMINI_SAFETY_CONFIG
        from agent.gemini_skill import GeminiSkillAgent, MM_BRANCH_MODE, TEXT_BRANCH_MODE, TEXT_INLINE_MODE

        mode_map = {
            "gemini-skill-text-inline": TEXT_INLINE_MODE,
            "gemini-skill-text-branch": TEXT_BRANCH_MODE,
            "gemini-skill-mm-branch": MM_BRANCH_MODE,
        }
        return GeminiSkillAgent(
            model=os.environ.get("MACOSWORLD_GEMINI_SKILL_MODEL", "gemini-3-flash").strip() or "gemini-3-flash",
            remote_client=remote_client,
            only_n_most_recent_images=image_window,
            max_tokens=12800,
            top_p=0.9,
            temperature=1.0,
            safety_config=GEMINI_SAFETY_CONFIG,
            skill_mode=mode_map[gui_agent_name],
            skills_library_dir=os.environ.get(
                "MACOSWORLD_SKILLS_LIBRARY_DIR",
                "mm_skills_download/mac_4_18_skills",
            ),
        )
    if gui_agent_name == "openai-skill-mm-branch" or gui_agent_name.endswith("/skill-mm-branch"):
        from agent.openai_skill import OpenAISkillAgent

        model_name = (
            gui_agent_name.rsplit("/", 1)[0]
            if gui_agent_name.endswith("/skill-mm-branch") and gui_agent_name != "openai-skill-mm-branch"
            else os.environ.get("MACOSWORLD_OPENAI_SKILL_MODEL", "gpt-4o").strip() or "gpt-4o"
        )
        return OpenAISkillAgent(
            model=model_name,
            remote_client=remote_client,
            screenshot_rolling_window=image_window,
            top_p=0.9,
            temperature=1.0,
            skills_library_dir=os.environ.get(
                "MACOSWORLD_SKILLS_LIBRARY_DIR",
                "mm_skills_download/mac_4_18_skills",
            ),
        )
    if "gpt" in gui_agent_name and "/omniparser" in gui_agent_name:
        from agent.openai_omniparser import OpenAI_OmniParser_Agent, GPT_OMNIPARSER_SYSTEM_PROMPT
        return OpenAI_OmniParser_Agent(
            model = gui_agent_name.split('/')[0],
            system_prompt = GPT_OMNIPARSER_SYSTEM_PROMPT,
            remote_client = remote_client,
            screenshot_rolling_window = image_window,
            top_p = 0.9,
            temperature = 1.0,
            device = 'cuda'
        )
    elif "openai/computer-use-preview" in gui_agent_name:
        from agent.openai_cua import OpenAI_CUA, CUA_SYSTEM_PROMPT
        return OpenAI_CUA(
            model = gui_agent_name.split('/')[1],
            system_prompt = CUA_SYSTEM_PROMPT,
            remote_client = remote_client,
            only_n_most_recent_images = image_window,
            top_p = 0.9,
            temperature = 1.0
        )
    elif "gpt" in gui_agent_name:
        from agent.openai import OpenAI_General_Agent, GPT_SYSTEM_PROMPT
        return OpenAI_General_Agent(
            model = gui_agent_name, 
            system_prompt = GPT_SYSTEM_PROMPT,
            remote_client = remote_client,
            screenshot_rolling_window = image_window,
            top_p = 0.9,
            temperature = 1.0
        )
    elif "claude-3-7-sonnet-20250219" in gui_agent_name and "computer-use-2025-01-24" in gui_agent_name:
        from agent.anthropic import ClaudeComputerUseAgent, CLAUDE_CUA_SYSTEM_PROMPT
        return ClaudeComputerUseAgent(
            model = gui_agent_name.split('/')[0],
            betas = gui_agent_name.split('/')[1:],
            max_tokens = 8192,
            display_width = SCREEN_WIDTH,
            display_height = SCREEN_HEIGHT,
            only_n_most_recent_images = image_window,
            system_prompt = CLAUDE_CUA_SYSTEM_PROMPT,
            remote_client = remote_client
        )
    elif "UI-TARS-7B-DPO" in gui_agent_name:
        from agent.uitars import UITARS_GUI_AGENT, UITARS_COMPUTER_SYSTEM_PROMPT
        return UITARS_GUI_AGENT(
            model = "UI-TARS-7B-DPO",
            vllm_base_url = "http://127.0.0.1:8000/v1",
            system_prompt = UITARS_COMPUTER_SYSTEM_PROMPT,
            remote_client = remote_client,
            only_n_most_recent_images = image_window,
            max_tokens = 12800,
            top_p = 0.9,
            temperature = 1.0
        )
    elif "showlab/ShowUI-2B" in gui_agent_name:
        from agent.showui import ShowUI_Agent, _NAV_SYSTEM, _NAV_FORMAT
        return ShowUI_Agent(
            model_name = "showlab/ShowUI-2B",
            system_prompt = _NAV_SYSTEM + _NAV_FORMAT,
            remote_client = remote_client,
            min_pixels = 256*28*28,
            max_pixels = 1344*28*28
        )
    elif "qwen" in gui_agent_name.lower():
        if _qwen_agent_arch() == "gemini":
            from agent.gemini import Gemini_OpenAICompat_Agent, GEMINI_SYSTEM_PROMPT
            return Gemini_OpenAICompat_Agent(
                model = gui_agent_name,
                system_prompt = GEMINI_SYSTEM_PROMPT,
                remote_client = remote_client,
                only_n_most_recent_images = image_window,
                max_tokens = 12800,
                top_p = 0.9,
                temperature = 1.0,
                base_url_env = "QWEN_BASE_URL",
                api_key_env = "QWEN_API_KEY",
                thinking_mode_env = "QWEN_THINKING_MODE",
                proxy_env = "QWEN_PROXY_URL",
            )
        if "skill" in gui_agent_name.lower() or _skills_enabled():
            from agent.qwen_skill import QwenVLSkillAgent
            return QwenVLSkillAgent(
                model = gui_agent_name.replace("/skill", ""),
                remote_client = remote_client,
                only_n_most_recent_images = image_window,
                max_tokens = 12800,
                top_p = 0.9,
                temperature = 1.0,
                skill_mode = os.environ.get("MACOSWORLD_SKILL_MODE", "multimodal").strip().lower() or "multimodal",
                skills_library_dir = os.environ.get("MACOSWORLD_SKILLS_LIBRARY_DIR", "skills_library"),
            )
        else:
            from agent.qwen import Qwen_General_Agent, QWEN_SYSTEM_PROMPT
            return Qwen_General_Agent(
                model = gui_agent_name,
                system_prompt = QWEN_SYSTEM_PROMPT,
                remote_client = remote_client,
                only_n_most_recent_images = image_window,
                max_tokens = 12800,
                top_p = 0.9,
                temperature = 1.0
            )
    elif "glm" in gui_agent_name.lower():
        from agent.gemini import Gemini_OpenAICompat_Agent, GEMINI_SYSTEM_PROMPT
        return Gemini_OpenAICompat_Agent(
            model = gui_agent_name,
            system_prompt = GEMINI_SYSTEM_PROMPT,
            remote_client = remote_client,
            only_n_most_recent_images = image_window,
            max_tokens = 12800,
            top_p = 0.9,
            temperature = 1.0,
            base_url_env = "GLM_BASE_URL",
            api_key_env = "GLM_API_KEY",
            thinking_mode_env = "GLM_THINKING_MODE",
            proxy_env = "GLM_PROXY_URL",
        )
    elif "gemini" in gui_agent_name:
        from agent.gemini import Gemini_General_Agent, GEMINI_SAFETY_CONFIG, GEMINI_SYSTEM_PROMPT
        return Gemini_General_Agent(
            model = gui_agent_name,
            system_prompt = GEMINI_SYSTEM_PROMPT,
            remote_client = remote_client,
            only_n_most_recent_images = image_window,
            max_tokens = 12800,
            top_p = 0.9,
            temperature = 1.0,
            safety_config = GEMINI_SAFETY_CONFIG
        )
    raise NotImplementedError(f'Agent "{gui_agent_name}" not implemented')

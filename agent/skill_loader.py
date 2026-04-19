import base64
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


logger = logging.getLogger("macosworld.skill_loader")


@dataclass(frozen=True)
class SkillMetadata:
    name: str
    description: str
    directory: str


@dataclass
class SkillContent:
    name: str
    description: str
    text: str
    image_references: List[str] = field(default_factory=list)
    directory: str = ""


class SkillLoader:
    def __init__(self, skills_library_dir: str = "skills_library", max_skill_chars: int = 12000):
        self._skills_dir = Path(skills_library_dir).expanduser()
        if not self._skills_dir.is_absolute():
            project_root = Path(__file__).resolve().parent.parent
            self._skills_dir = project_root / skills_library_dir

        self._max_skill_chars = max_skill_chars
        self._metadata_cache: Dict[str, SkillMetadata] = {}
        self._content_cache: Dict[str, SkillContent] = {}
        self._skill_id_to_dir: Dict[str, Path] = {}
        self._basename_to_skill_ids: Dict[str, List[str]] = {}
        self._skill_index_built = False

    def discover_all_skills(self) -> List[SkillMetadata]:
        if self._metadata_cache:
            return list(self._metadata_cache.values())

        if not self._skills_dir.exists():
            logger.warning("Skills library directory not found: %s", self._skills_dir)
            return []

        self._ensure_skill_index()
        metadata_list: List[SkillMetadata] = []
        for skill_id, skill_dir in sorted(self._skill_id_to_dir.items()):
            skill_md = self._find_skill_md(skill_dir)
            if skill_md is None:
                continue
            text = skill_md.read_text(encoding="utf-8")
            frontmatter = self._parse_frontmatter(text)
            meta = SkillMetadata(
                name=frontmatter.get("name", skill_dir.name),
                description=frontmatter.get("description", ""),
                directory=str(skill_dir),
            )
            self._metadata_cache[skill_id] = meta
            metadata_list.append(meta)
        return metadata_list

    def load_full_skill(self, skill_name: str) -> Optional[Dict]:
        content = self._load_skill_content(skill_name)
        if content is None:
            return None
        images = self.load_skill_images(skill_name)
        return {"content": content, "images": images}

    def load_skill_images(self, skill_name: str) -> List[Tuple[str, str, str]]:
        resolved = self._resolve_skill_identifier_and_dir(skill_name)
        if resolved is None:
            return []
        _, skill_dir = resolved
        images_dir = skill_dir / "Images"
        if not images_dir.exists():
            for child in skill_dir.iterdir():
                if child.is_dir() and child.name.lower() == "images":
                    images_dir = child
                    break
            else:
                return []

        results: List[Tuple[str, str, str]] = []
        for path in sorted(images_dir.iterdir()):
            if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}:
                continue
            img_bytes = path.read_bytes()
            results.append(
                (
                    path.name,
                    base64.b64encode(img_bytes).decode("utf-8"),
                    self._get_mime_type(path.suffix),
                )
            )
        return results

    def load_runtime_state_cards(self, skill_name: str) -> Optional[dict]:
        resolved = self._resolve_skill_identifier_and_dir(skill_name)
        if resolved is None:
            return None
        _, skill_dir = resolved
        runtime_cards = skill_dir / "runtime_state_cards.json"
        if not runtime_cards.exists():
            return None
        try:
            import json

            return json.loads(runtime_cards.read_text(encoding="utf-8"))
        except Exception:
            return None

    def summarize_runtime_state_cards(self, skill_name: str, max_states: int = 3) -> str:
        payload = self.load_runtime_state_cards(skill_name)
        if not isinstance(payload, dict):
            return "(no runtime state summary)"
        states = payload.get("states")
        if not isinstance(states, list) or not states:
            return "(no runtime state summary)"

        lines: List[str] = []
        for state in states[:max_states]:
            if not isinstance(state, dict):
                continue
            state_name = state.get("state_name") or state.get("state_id") or "unknown_state"
            when_to_use = state.get("when_to_use") or state.get("trigger_condition") or ""
            available_views = state.get("available_views") or []
            view_types = []
            for item in available_views:
                if isinstance(item, dict) and item.get("view_type"):
                    view_types.append(str(item["view_type"]))
            view_suffix = f" [views: {', '.join(view_types)}]" if view_types else ""
            lines.append(f"- {state_name}{view_suffix}: {str(when_to_use).strip()}")
        return "\n".join(lines) if lines else "(no runtime state summary)"

    def _load_skill_content(self, skill_name: str) -> Optional[SkillContent]:
        if skill_name in self._content_cache:
            return self._content_cache[skill_name]

        resolved = self._resolve_skill_identifier_and_dir(skill_name)
        if resolved is None:
            return None
        _, skill_dir = resolved
        skill_md = self._find_skill_md(skill_dir)
        if skill_md is None:
            return None

        raw_text = skill_md.read_text(encoding="utf-8")
        frontmatter = self._parse_frontmatter(raw_text)
        body = self._strip_frontmatter(raw_text).strip()
        if len(body) > self._max_skill_chars:
            body = body[: self._max_skill_chars].rstrip() + "\n\n[truncated]"

        content = SkillContent(
            name=frontmatter.get("name", skill_dir.name),
            description=frontmatter.get("description", ""),
            text=body,
            image_references=self._extract_image_references(body),
            directory=str(skill_dir),
        )
        self._content_cache[skill_name] = content
        return content

    def _ensure_skill_index(self) -> None:
        if self._skill_index_built:
            return

        if not self._skills_dir.exists():
            self._skill_index_built = True
            return

        for root, _, files in __import__("os").walk(self._skills_dir):
            if "SKILL.md" not in files:
                continue
            skill_dir = Path(root)
            skill_id = skill_dir.relative_to(self._skills_dir).as_posix()
            self._skill_id_to_dir[skill_id] = skill_dir
            self._basename_to_skill_ids.setdefault(skill_dir.name, []).append(skill_id)

        self._skill_index_built = True

    def _resolve_skill_identifier_and_dir(self, skill_name: str) -> Optional[Tuple[str, Path]]:
        self._ensure_skill_index()
        if skill_name in self._skill_id_to_dir:
            return skill_name, self._skill_id_to_dir[skill_name]
        basename_matches = self._basename_to_skill_ids.get(skill_name, [])
        if len(basename_matches) == 1:
            skill_id = basename_matches[0]
            return skill_id, self._skill_id_to_dir[skill_id]
        if len(basename_matches) > 1:
            skill_id = basename_matches[0]
            logger.warning("Multiple skills matched '%s'; using %s", skill_name, skill_id)
            return skill_id, self._skill_id_to_dir[skill_id]
        return None

    @staticmethod
    def _find_skill_md(skill_dir: Path) -> Optional[Path]:
        direct = skill_dir / "SKILL.md"
        if direct.exists():
            return direct
        for candidate in skill_dir.glob("**/SKILL.md"):
            return candidate
        return None

    @staticmethod
    def _parse_frontmatter(text: str) -> Dict[str, str]:
        if not text.startswith("---\n"):
            return {}
        parts = text.split("\n---\n", 1)
        if len(parts) != 2:
            return {}
        raw_frontmatter = parts[0][4:]
        parsed: Dict[str, str] = {}
        for line in raw_frontmatter.splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            parsed[key.strip()] = value.strip()
        return parsed

    @staticmethod
    def _strip_frontmatter(text: str) -> str:
        if not text.startswith("---\n"):
            return text
        parts = text.split("\n---\n", 1)
        if len(parts) != 2:
            return text
        return parts[1]

    @staticmethod
    def _extract_image_references(text: str) -> List[str]:
        refs = re.findall(r"(?:^|[\s(])((?:Images|images)/[^)\s]+)", text)
        deduped: List[str] = []
        for ref in refs:
            if ref not in deduped:
                deduped.append(ref)
        return deduped

    @staticmethod
    def _get_mime_type(suffix: str) -> str:
        suffix = suffix.lower()
        if suffix == ".png":
            return "image/png"
        if suffix in {".jpg", ".jpeg"}:
            return "image/jpeg"
        if suffix == ".gif":
            return "image/gif"
        if suffix == ".webp":
            return "image/webp"
        if suffix == ".bmp":
            return "image/bmp"
        return "application/octet-stream"

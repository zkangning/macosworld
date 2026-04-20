import base64
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


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


@dataclass(frozen=True)
class SkillStateView:
    view_type: str
    image_path: str
    use_for: str = ""
    label: str = ""


@dataclass(frozen=True)
class SkillStateCard:
    state_id: str
    state_name: str
    stage: str
    image_role: str
    when_to_use: str
    when_not_to_use: str
    visible_cues: List[str]
    verification_cue: str
    visual_evidence_chain: Dict[str, Any]
    visual_risk: str
    preferred_view_order: List[str]
    available_views: List[SkillStateView]


@dataclass(frozen=True)
class LoadedSkillStateView:
    view: SkillStateView
    image: Tuple[str, str, str]


@dataclass
class SkillStateSelection:
    state: SkillStateCard
    requested_view_types: List[str]
    reason: str
    loaded_views: List[LoadedSkillStateView] = field(default_factory=list)


class SkillLoader:
    def __init__(self, skills_library_dir: str = "skills_library", max_skill_chars: int = 12000):
        self._skills_dir = Path(skills_library_dir).expanduser()
        if not self._skills_dir.is_absolute():
            project_root = Path(__file__).resolve().parent.parent
            self._skills_dir = project_root / skills_library_dir

        self._max_skill_chars = max_skill_chars
        self._metadata_cache: Dict[str, SkillMetadata] = {}
        self._content_cache: Dict[str, SkillContent] = {}
        self._state_cards_cache: Dict[Tuple[str, bool], Optional[List[SkillStateCard]]] = {}
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

    def load_skill_content(self, skill_name: str) -> Optional[SkillContent]:
        return self._load_skill_content(skill_name)

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

    def load_state_cards(self, skill_name: str, runtime: bool = True) -> Optional[List[SkillStateCard]]:
        cache_key = (skill_name, runtime)
        if cache_key in self._state_cards_cache:
            return self._state_cards_cache[cache_key]

        resolved = self._resolve_skill_identifier_and_dir(skill_name)
        if resolved is None:
            self._state_cards_cache[cache_key] = None
            return None
        _, skill_dir = resolved

        filename = "runtime_state_cards.json" if runtime else "state_cards.json"
        state_cards_path = skill_dir / filename
        if not state_cards_path.exists():
            self._state_cards_cache[cache_key] = None
            return None

        try:
            payload = json.loads(state_cards_path.read_text(encoding="utf-8"))
        except Exception:
            self._state_cards_cache[cache_key] = None
            return None

        states = payload.get("states")
        if not isinstance(states, list):
            self._state_cards_cache[cache_key] = None
            return None

        parsed_cards: List[SkillStateCard] = []
        for state in states:
            if not isinstance(state, dict):
                continue
            available_views: List[SkillStateView] = []
            for view in state.get("available_views", []) or []:
                if not isinstance(view, dict):
                    continue
                view_type = str(view.get("view_type", "") or "").strip()
                image_path = str(view.get("image_path", "") or "").strip()
                if not view_type or not image_path:
                    continue
                available_views.append(
                    SkillStateView(
                        view_type=view_type,
                        image_path=image_path,
                        use_for=str(view.get("use_for", "") or "").strip(),
                        label=str(view.get("label", "") or "").strip(),
                    )
                )
            parsed_cards.append(
                SkillStateCard(
                    state_id=str(state.get("state_id", "") or "").strip(),
                    state_name=str(state.get("state_name", "") or "").strip(),
                    stage=str(state.get("stage", "") or "").strip(),
                    image_role=str(state.get("image_role", "") or "").strip(),
                    when_to_use=str(state.get("when_to_use", "") or "").strip(),
                    when_not_to_use=str(state.get("when_not_to_use", "") or "").strip(),
                    visible_cues=[str(item).strip() for item in (state.get("visible_cues") or []) if str(item).strip()],
                    verification_cue=str(state.get("verification_cue", "") or "").strip(),
                    visual_evidence_chain=state.get("visual_evidence_chain") or {},
                    visual_risk=str(state.get("visual_risk", "") or "").strip(),
                    preferred_view_order=[
                        str(item).strip() for item in (state.get("preferred_view_order") or []) if str(item).strip()
                    ],
                    available_views=available_views,
                )
            )

        self._state_cards_cache[cache_key] = parsed_cards
        return parsed_cards

    def summarize_runtime_state_cards(self, skill_name: str, max_states: int = 3) -> str:
        state_cards = self.load_state_cards(skill_name, runtime=True)
        return self.summarize_state_cards_for_preview(state_cards, max_cards=max_states)

    def summarize_state_cards_for_preview(
        self,
        state_cards: Optional[List[SkillStateCard]],
        max_cards: int = 3,
    ) -> str:
        if not state_cards:
            return "(no runtime state summary)"
        lines: List[str] = []
        for card in state_cards[:max_cards]:
            state_name = card.state_name or card.state_id or "unknown_state"
            view_types = [view.view_type for view in card.available_views if view.view_type]
            view_suffix = f" [views: {', '.join(view_types)}]" if view_types else ""
            when_to_use = card.when_to_use or "(no when_to_use provided)"
            lines.append(f"- {state_name}{view_suffix}: {when_to_use}")
        return "\n".join(lines) if lines else "(no runtime state summary)"

    def format_state_cards_for_branch(self, state_cards: Optional[List[SkillStateCard]]) -> str:
        if not state_cards:
            return "(no runtime state-card reference)"

        sections: List[str] = ["Runtime state-card reference manifest:"]
        for card in state_cards:
            view_lines = []
            for view in card.available_views:
                view_lines.append(
                    f"  - {view.view_type}: path={view.image_path}, use_for={view.use_for or '(missing)'}, label={view.label or '(missing)'}"
                )
            visible_cues = ", ".join(card.visible_cues[:4]) if card.visible_cues else "(none listed)"
            preferred_order = ", ".join(card.preferred_view_order) if card.preferred_view_order else "(none listed)"
            sections.append(
                "\n".join(
                    [
                        f"[State - {card.state_id}]",
                        f"state_name: {card.state_name or '(missing)'}",
                        f"stage: {card.stage or '(missing)'}",
                        f"when_to_use: {card.when_to_use or '(missing)'}",
                        f"when_not_to_use: {card.when_not_to_use or '(missing)'}",
                        f"verification_cue: {card.verification_cue or '(missing)'}",
                        f"visible_cues: {visible_cues}",
                        f"visual_risk: {card.visual_risk or '(missing)'}",
                        f"preferred_view_order: {preferred_order}",
                        "available_views:",
                        *(view_lines or ["  - (none listed)"]),
                    ]
                )
            )
        return "\n\n".join(sections)

    def load_selected_state_views(
        self,
        skill_name: str,
        requests: List[Dict[str, object]],
        runtime: bool = True,
    ) -> Tuple[List[SkillStateSelection], List[str]]:
        state_cards = self.load_state_cards(skill_name, runtime=runtime)
        resolved = self._resolve_skill_identifier_and_dir(skill_name)
        if not state_cards or resolved is None:
            return [], [str(item.get("state_id", "") or "").strip() for item in requests if isinstance(item, dict)]

        _, skill_dir = resolved
        state_by_id = {card.state_id: card for card in state_cards if card.state_id}
        selections: List[SkillStateSelection] = []
        missing_items: List[str] = []

        for item in requests:
            if not isinstance(item, dict):
                continue
            state_id = str(item.get("state_id", "") or "").strip()
            requested_views = [str(view).strip() for view in (item.get("views") or []) if str(view).strip()]
            reason = str(item.get("reason", "") or "").strip()
            state = state_by_id.get(state_id)
            if state is None:
                missing_items.append(state_id or "(missing state_id)")
                continue

            loaded_views: List[LoadedSkillStateView] = []
            available_view_map = {view.view_type: view for view in state.available_views if view.view_type}
            for view_type in requested_views:
                view = available_view_map.get(view_type)
                if view is None:
                    missing_items.append(f"{state_id}/{view_type}")
                    continue
                image_tuple = self._load_state_view_image(skill_dir, view.image_path)
                if image_tuple is None:
                    missing_items.append(f"{state_id}/{view_type}")
                    continue
                loaded_views.append(LoadedSkillStateView(view=view, image=image_tuple))

            if loaded_views:
                selections.append(
                    SkillStateSelection(
                        state=state,
                        requested_view_types=requested_views,
                        reason=reason,
                        loaded_views=loaded_views,
                    )
                )

        return selections, missing_items

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

    def _load_state_view_image(self, skill_dir: Path, image_path: str) -> Optional[Tuple[str, str, str]]:
        path = skill_dir / image_path
        if not path.exists():
            normalized = image_path.replace("\\", "/")
            if normalized.lower().startswith("images/"):
                suffix = normalized.split("/", 1)[1]
                for images_dir_name in ("Images", "images"):
                    candidate = skill_dir / images_dir_name / suffix
                    if candidate.exists():
                        path = candidate
                        break
        if not path.exists():
            return None

        img_bytes = path.read_bytes()
        return (
            path.name,
            base64.b64encode(img_bytes).decode("utf-8"),
            self._get_mime_type(path.suffix),
        )

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from agent.skill_loader import SkillLoader


logger = logging.getLogger("macosworld.task_skill_resolver")

_RESOLVER_CACHE: Dict[Tuple[Optional[str], str, Optional[int]], "TaskSkillResolver"] = {}


class TaskSkillResolver:
    _PRIORITY_ORDER = {
        "P0": 0,
        "P1": 1,
        "P2": 2,
        "P3": 3,
    }

    def __init__(self, skills_library_dir: str, mapping_root: Optional[str] = None, top_k: Optional[int] = 6):
        self._skills_library_dir = skills_library_dir
        self._mapping_root = self._resolve_path(mapping_root) if mapping_root else None
        self._top_k = top_k if top_k and top_k > 0 else None
        self._domain_mapping_cache: Dict[str, Optional[dict]] = {}
        self._flat_mapping_cache: Optional[dict] = None
        self._all_skill_dirs: set[str] = set()
        self._name_to_dirs: Dict[str, List[str]] = {}
        self._normalized_name_to_dirs: Dict[str, List[str]] = {}
        self._index_built = False

    def resolve_task_skills(
        self,
        *,
        domain: Optional[str],
        task_id: Optional[str],
        fallback_skill_names: Optional[Sequence[str]] = None,
    ) -> List[str]:
        source_names: List[str] = []
        if domain and task_id:
            mapped_entries = self._lookup_mapped_entries(domain, task_id)
            if mapped_entries is not None:
                source_names = self._select_top_skills(mapped_entries)
        if not source_names and fallback_skill_names:
            source_names = list(fallback_skill_names)

        resolved_names: List[str] = []
        for raw_name in source_names:
            directory_name = self._resolve_skill_directory_name(raw_name, domain)
            if directory_name and directory_name not in resolved_names:
                resolved_names.append(directory_name)
        return resolved_names

    def _lookup_mapped_entries(self, domain: str, task_id: str) -> Optional[List[dict]]:
        flat_entries = self._lookup_flat_mapping_entries(domain, task_id)
        if flat_entries is not None:
            return flat_entries
        mapping = self._load_domain_mapping(domain)
        if not mapping:
            return None
        task_entry = mapping.get("task_to_skills", {}).get(task_id)
        if task_entry is None:
            return None
        return list(task_entry.get("skills", []))

    def _lookup_flat_mapping_entries(self, domain: str, task_id: str) -> Optional[List[dict]]:
        mapping = self._load_flat_mapping()
        if not mapping:
            return None
        domain_mapping = mapping.get(domain)
        if not isinstance(domain_mapping, dict):
            return None
        task_skills = domain_mapping.get(task_id)
        if not isinstance(task_skills, list):
            return None
        return [{"skill_name": str(name).strip(), "priority": "P0"} for name in task_skills if str(name).strip()]

    def _load_domain_mapping(self, domain: str) -> Optional[dict]:
        if domain in self._domain_mapping_cache:
            return self._domain_mapping_cache[domain]
        if not self._mapping_root:
            self._domain_mapping_cache[domain] = None
            return None
        if self._mapping_root.is_file():
            self._domain_mapping_cache[domain] = None
            return None
        mapping_file = self._mapping_root / domain / "task_skill_mapping_generated.json"
        if not mapping_file.exists():
            self._domain_mapping_cache[domain] = None
            return None
        self._domain_mapping_cache[domain] = json.loads(mapping_file.read_text(encoding="utf-8"))
        return self._domain_mapping_cache[domain]

    def _load_flat_mapping(self) -> Optional[dict]:
        if self._flat_mapping_cache is not None:
            return self._flat_mapping_cache
        if not self._mapping_root:
            self._flat_mapping_cache = None
            return None

        mapping_file: Optional[Path] = None
        if self._mapping_root.is_file():
            mapping_file = self._mapping_root
        else:
            candidate = self._mapping_root / "qwen35_top5_skill_mapping.json"
            if candidate.exists():
                mapping_file = candidate

        if mapping_file is None or not mapping_file.exists():
            self._flat_mapping_cache = None
            return None

        data = json.loads(mapping_file.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            self._flat_mapping_cache = None
            return None
        self._flat_mapping_cache = data
        return self._flat_mapping_cache

    def _select_top_skills(self, skill_entries: Sequence[dict]) -> List[str]:
        ranked_entries = sorted(
            enumerate(skill_entries),
            key=lambda item: (self._priority_rank(item[1].get("priority")), item[0]),
        )
        if self._top_k is not None:
            ranked_entries = ranked_entries[: self._top_k]
        return [
            entry.get("skill_name", "").strip()
            for _, entry in ranked_entries
            if entry.get("skill_name", "").strip()
        ]

    def _resolve_skill_directory_name(self, skill_name: str, domain: Optional[str]) -> Optional[str]:
        if not skill_name:
            return None
        self._ensure_skill_index()
        if skill_name in self._all_skill_dirs:
            return skill_name

        candidates = list(self._name_to_dirs.get(skill_name, []))
        if not candidates:
            candidates = list(self._normalized_name_to_dirs.get(self._normalize(skill_name), []))
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]

        if domain:
            domain_prefix = domain.upper().replace("_", "")
            prefixed = [candidate for candidate in candidates if candidate.startswith(f"{domain_prefix}_")]
            if len(prefixed) == 1:
                return prefixed[0]
            if prefixed:
                return prefixed[0]
        return candidates[0]

    def _ensure_skill_index(self) -> None:
        if self._index_built:
            return
        skill_loader = SkillLoader(skills_library_dir=self._skills_library_dir)
        for meta in skill_loader.discover_all_skills():
            directory_name = Path(meta.directory).name
            self._all_skill_dirs.add(directory_name)
            self._name_to_dirs.setdefault(meta.name, []).append(directory_name)
            self._normalized_name_to_dirs.setdefault(self._normalize(meta.name), []).append(directory_name)
            self._normalized_name_to_dirs.setdefault(self._normalize(directory_name), []).append(directory_name)
        self._index_built = True

    @classmethod
    def _priority_rank(cls, priority: Optional[str]) -> int:
        return cls._PRIORITY_ORDER.get((priority or "").upper(), 99)

    @staticmethod
    def _normalize(value: str) -> str:
        return "".join(ch for ch in value.lower() if ch.isalnum())

    @staticmethod
    def _resolve_path(path_str: str) -> Path:
        path = Path(path_str).expanduser()
        return path if path.is_absolute() else Path.cwd() / path


def resolve_task_skill_names(
    *,
    domain: Optional[str],
    task_id: Optional[str],
    fallback_skill_names: Optional[Sequence[str]] = None,
    skills_library_dir: str = "skills_library",
    mapping_root: Optional[str] = None,
    top_k: Optional[int] = 6,
) -> List[str]:
    cache_key = (mapping_root, skills_library_dir, top_k if top_k and top_k > 0 else None)
    resolver = _RESOLVER_CACHE.get(cache_key)
    if resolver is None:
        resolver = TaskSkillResolver(
            skills_library_dir=skills_library_dir,
            mapping_root=mapping_root,
            top_k=top_k,
        )
        _RESOLVER_CACHE[cache_key] = resolver
    return resolver.resolve_task_skills(
        domain=domain,
        task_id=task_id,
        fallback_skill_names=fallback_skill_names,
    )

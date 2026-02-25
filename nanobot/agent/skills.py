"""Skills loader for agent capabilities."""

import json
import os
import re
import shutil
from pathlib import Path
from typing import Any

import yaml

# Default builtin skills directory (relative to this file)
BUILTIN_SKILLS_DIR = Path(__file__).parent.parent / "skills"


class SkillsLoader:
    """
    Loader for agent skills.
    
    Skills are markdown files (SKILL.md) that teach the agent how to use
    specific tools or perform certain tasks.
    """
    
    def __init__(
        self,
        workspace: Path,
        builtin_skills_dir: Path | None = None,
        global_skills_dir: Path | None = None,
    ):
        self.workspace = workspace
        self.workspace_skills = workspace / "skills"
        if global_skills_dir is None:
            if workspace.parent.name == "agents":
                global_skills_dir = workspace.parent.parent / "skills"
            else:
                global_skills_dir = self.workspace_skills
        self.global_skills = global_skills_dir
        self.builtin_skills = builtin_skills_dir or BUILTIN_SKILLS_DIR
        self._skill_content_cache: dict[str, str | None] = {}
        self._skill_metadata_cache: dict[str, dict | None] = {}

    def clear_cache(self) -> None:
        """Clear per-build caches to avoid stale reads across turns."""
        self._skill_content_cache.clear()
        self._skill_metadata_cache.clear()

    def _skill_roots(self) -> list[tuple[str, Path]]:
        """Get ordered skill roots by priority, de-duplicated by directory path."""
        roots: list[tuple[str, Path | None]] = [
            ("workspace", self.workspace_skills),
            ("global", self.global_skills),
            ("builtin", self.builtin_skills),
        ]
        seen: set[Path] = set()
        result: list[tuple[str, Path]] = []

        for source, path in roots:
            if path is None:
                continue
            key = path.expanduser().resolve(strict=False)
            if key in seen:
                continue
            seen.add(key)
            result.append((source, path))

        return result
    
    @staticmethod
    def _normalize_skill_names(skill_names: list[str] | set[str] | None) -> set[str] | None:
        if skill_names is None:
            return None
        return set(skill_names)

    def list_skills(
        self,
        filter_unavailable: bool = True,
        skill_names: list[str] | set[str] | None = None,
    ) -> list[dict[str, str]]:
        """
        List all available skills.
        
        Args:
            filter_unavailable: If True, filter out skills with unmet requirements.
        
        Returns:
            List of skill info dicts with 'name', 'path', 'source'.
        """
        allowed_names = self._normalize_skill_names(skill_names)
        skills = []
        seen_names: set[str] = set()

        for source, root in self._skill_roots():
            if not root.exists():
                continue
            for skill_dir in sorted(root.iterdir(), key=lambda p: p.name):
                if not skill_dir.is_dir() or skill_dir.name in seen_names:
                    continue
                if allowed_names is not None and skill_dir.name not in allowed_names:
                    continue
                skill_file = skill_dir / "SKILL.md"
                if not skill_file.exists():
                    continue
                skills.append(
                    {"name": skill_dir.name, "path": str(skill_file), "source": source}
                )
                seen_names.add(skill_dir.name)
        
        # Filter by requirements
        if filter_unavailable:
            return [s for s in skills if self._check_requirements(self._get_skill_meta(s["name"]))]
        return skills
    
    def load_skill(self, name: str) -> str | None:
        """
        Load a skill by name.
        
        Args:
            name: Skill name (directory name).
        
        Returns:
            Skill content or None if not found.
        """
        if name in self._skill_content_cache:
            return self._skill_content_cache[name]

        for _, root in self._skill_roots():
            skill_file = root / name / "SKILL.md"
            if skill_file.exists():
                content = skill_file.read_text(encoding="utf-8")
                self._skill_content_cache[name] = content
                return content

        self._skill_content_cache[name] = None
        return None
    
    def load_skills_for_context(self, skill_names: list[str]) -> str:
        """
        Load specific skills for inclusion in agent context.
        
        Args:
            skill_names: List of skill names to load.
        
        Returns:
            Formatted skills content.
        """
        parts = []
        for name in skill_names:
            content = self.load_skill(name)
            if content:
                content = self._strip_frontmatter(content)
                parts.append(f"### Skill: {name}\n\n{content}")
        
        return "\n\n---\n\n".join(parts) if parts else ""
    
    def build_skills_summary(
        self,
        skill_names: list[str] | None = None,
        skills: list[dict[str, str]] | None = None,
        readable_skill_names: set[str] | None = None,
    ) -> str:
        """
        Build a summary of all skills (name, description, path, availability).
        
        This is used for progressive loading - the agent can read the full
        skill content using read_file when needed.
        
        Returns:
            XML-formatted skills summary.
        """
        all_skills = (
            skills
            if skills is not None
            else self.list_skills(filter_unavailable=False, skill_names=skill_names)
        )
        if not all_skills:
            return ""
        
        def escape_xml(s: str) -> str:
            return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        
        lines = ["<skills>"]
        for s in all_skills:
            name = escape_xml(s["name"])
            path = s["path"]
            desc = escape_xml(self._get_skill_description(s["name"]))
            skill_meta = self._get_skill_meta(s["name"])
            available = self._check_requirements(skill_meta)
            
            lines.append(f"  <skill available=\"{str(available).lower()}\">")
            lines.append(f"    <name>{name}</name>")
            lines.append(f"    <description>{desc}</description>")
            if readable_skill_names is None or s["name"] in readable_skill_names:
                lines.append(f"    <location>{path}</location>")
            else:
                lines.append("    <location>inlined</location>")
            
            # Show missing requirements for unavailable skills
            if not available:
                missing = self._get_missing_requirements(skill_meta)
                if missing:
                    lines.append(f"    <requires>{escape_xml(missing)}</requires>")
            
            lines.append(f"  </skill>")
        lines.append("</skills>")
        
        return "\n".join(lines)
    
    def _get_missing_requirements(self, skill_meta: dict) -> str:
        """Get a description of missing requirements."""
        missing = []
        requires = skill_meta.get("requires", {})
        for b in requires.get("bins", []):
            if not shutil.which(b):
                missing.append(f"CLI: {b}")
        for env in requires.get("env", []):
            if not os.environ.get(env):
                missing.append(f"ENV: {env}")
        return ", ".join(missing)
    
    def _get_skill_description(self, name: str) -> str:
        """Get the description of a skill from its frontmatter."""
        meta = self.get_skill_metadata(name)
        if meta and meta.get("description"):
            return meta["description"]
        return name  # Fallback to skill name
    
    def _strip_frontmatter(self, content: str) -> str:
        """Remove YAML frontmatter from markdown content."""
        if content.startswith("---"):
            match = re.match(r"^---\r?\n.*?\r?\n---(?:\r?\n|$)", content, re.DOTALL)
            if match:
                return content[match.end():].strip()
        return content
    
    def _parse_nanobot_metadata(self, raw: Any) -> dict:
        """Parse skill metadata field (supports legacy JSON strings and YAML dicts)."""
        data: Any = raw
        if isinstance(raw, str):
            try:
                data = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                return {}

        if not isinstance(data, dict):
            return {}
        if isinstance(data.get("nanobot"), dict):
            return data["nanobot"]
        if isinstance(data.get("openclaw"), dict):
            return data["openclaw"]
        return data
    
    def _check_requirements(self, skill_meta: dict) -> bool:
        """Check if skill requirements are met (bins, env vars)."""
        requires = skill_meta.get("requires", {})
        for b in requires.get("bins", []):
            if not shutil.which(b):
                return False
        for env in requires.get("env", []):
            if not os.environ.get(env):
                return False
        return True
    
    def _get_skill_meta(self, name: str) -> dict:
        """Get nanobot metadata for a skill (cached in frontmatter)."""
        meta = self.get_skill_metadata(name) or {}
        skill_meta = self._parse_nanobot_metadata(meta.get("metadata", {}))
        for key in ("requires", "always", "os"):
            if key in meta and key not in skill_meta:
                skill_meta[key] = meta[key]
        return skill_meta

    def get_always_skills(self, skill_names: list[str] | None = None) -> list[str]:
        """Get skills marked as always=true that meet requirements."""
        result = []
        for s in self.list_skills(filter_unavailable=True, skill_names=skill_names):
            meta = self.get_skill_metadata(s["name"]) or {}
            skill_meta = self._parse_nanobot_metadata(meta.get("metadata", {}))
            always_meta = skill_meta.get("always")
            always_top = meta.get("always")
            if always_meta is True or always_top is True:
                result.append(s["name"])
        return result

    def _parse_frontmatter(self, content: str) -> dict | None:
        """Parse YAML frontmatter, preserving legacy metadata JSON compatibility."""
        if not content.startswith("---"):
            return None

        match = re.match(r"^---\s*\r?\n(.*?)\r?\n---(?:\s*\r?\n|$)", content, re.DOTALL)
        if not match:
            return None

        raw_frontmatter = match.group(1)
        try:
            parsed = yaml.safe_load(raw_frontmatter) or {}
        except yaml.YAMLError:
            return None

        if not isinstance(parsed, dict):
            return None

        metadata_field = parsed.get("metadata")
        if isinstance(metadata_field, str):
            try:
                legacy_metadata = json.loads(metadata_field)
            except json.JSONDecodeError:
                legacy_metadata = None
            if isinstance(legacy_metadata, dict):
                parsed["metadata"] = legacy_metadata
                for key, value in legacy_metadata.items():
                    parsed.setdefault(key, value)

        return parsed
    
    def get_skill_metadata(self, name: str) -> dict | None:
        """
        Get metadata from a skill's frontmatter.
        
        Args:
            name: Skill name.
        
        Returns:
            Metadata dict or None.
        """
        if name in self._skill_metadata_cache:
            return self._skill_metadata_cache[name]

        content = self.load_skill(name)
        if not content:
            self._skill_metadata_cache[name] = None
            return None

        metadata = self._parse_frontmatter(content)
        self._skill_metadata_cache[name] = metadata
        return metadata

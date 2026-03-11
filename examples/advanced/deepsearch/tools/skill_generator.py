"""Skill generator tool — create complete skill packages.

1:1 port of SkyworkAI's SkillGeneratorTool from src/tool/workflow_tools/skill_generator.py.
Simplified for Orbiter: no SCP registry, generates skill files to disk.
"""

from __future__ import annotations

import asyncio
import ast
import json
import logging
import os
import re
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from orbiter.tool import Tool
from orbiter.types import SystemMessage, UserMessage

from ..llm_utils import call_llm

logger = logging.getLogger("deepagent")


# ======================================================================
# Pydantic models (1:1 from SkyworkAI)
# ======================================================================


class SkillFileSpec(BaseModel):
    filename: str = Field(description="Relative path within the skill directory")
    purpose: str = Field(description="Brief description of what this file does")


class SkillSpecification(BaseModel):
    skill_name: str = Field(description="Name of the skill (kebab-case)")
    description: str = Field(description="Concise description of what the skill does")
    scripts: list[SkillFileSpec] = Field(default_factory=list, description="Scripts under scripts/")
    resources: list[SkillFileSpec] = Field(default_factory=list, description="Resources under resources/")
    include_examples: bool = Field(default=True, description="Whether to generate examples.md")
    include_reference: bool = Field(default=True, description="Whether to generate reference.md")
    implementation_plan: str = Field(description="High-level plan for the skill workflow")


class SkillFileContent(BaseModel):
    filename: str = Field(description="Relative path within the skill directory")
    content: str = Field(description="Full file content")


class SkillEvaluation(BaseModel):
    is_valid: bool = Field(description="Whether the skill passes validation")
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    reasoning: str = Field(description="Brief explanation")


class SkillFilePlan(BaseModel):
    filename: str
    purpose: str
    generator: str
    status: str = "pending"


class SkillBuildPlan(BaseModel):
    skill_name: str
    description: str
    implementation_plan: str
    files: list[SkillFilePlan] = Field(default_factory=list)
    log: list[str] = Field(default_factory=list)
    created_at: str = ""


_FENCE_PATTERN = re.compile(r"^```[a-zA-Z]*\s*\n?", re.MULTILINE)
_FENCE_TAIL_PATTERN = re.compile(r"\n?```\s*$")


def _prefixed(prefix: str, filename: str) -> str:
    return filename if filename.startswith(prefix) else f"{prefix}{filename}"


def _strip_fences(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = _FENCE_PATTERN.sub("", cleaned, count=1)
        cleaned = _FENCE_TAIL_PATTERN.sub("", cleaned)
    return cleaned.strip()


def _parse_file_list(raw: str) -> list[SkillFileContent]:
    cleaned = _strip_fences(raw)
    try:
        data = json.loads(cleaned)
        items = data.get("files", data) if isinstance(data, dict) else data
        if not isinstance(items, list):
            items = [items]
        return [SkillFileContent(**item) for item in items]
    except Exception as e:
        logger.error(f"Failed to parse JSON output: {e}")
        return []


_SKILL_GENERATOR_DESCRIPTION = """Skill generator tool that creates complete skill packages.
This tool will:
1. Analyze task requirements and design a skill specification
2. Create a build plan listing all files to generate
3. Generate all files concurrently
4. Validate the generated skill
5. Write files to disk

Args:
- task (str): Description of what the skill should do.
- skill_name (Optional[str]): Explicit skill name (kebab-case).
- description (Optional[str]): Explicit skill description.

Example: {"name": "skill_generator", "args": {"task": "Create a skill that translates text between languages"}}.
"""


class SkillGeneratorTool(Tool):
    """Generate complete skill packages with scripts, resources, and docs.

    1:1 port of SkyworkAI's SkillGeneratorTool, simplified for Orbiter.
    """

    def __init__(
        self,
        *,
        model: str = "openai:gpt-4o-mini",
        base_dir: str = "deepagent_output/skill_generator",
    ) -> None:
        self.name = "skill_generator"
        self.description = _SKILL_GENERATOR_DESCRIPTION
        self.parameters = {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "Description of what the skill should do.",
                },
                "skill_name": {
                    "type": "string",
                    "description": "Explicit skill name (kebab-case).",
                },
                "description": {
                    "type": "string",
                    "description": "Explicit skill description.",
                },
            },
            "required": ["task"],
        }
        self._model = model
        self._base_dir = base_dir
        os.makedirs(self._base_dir, exist_ok=True)

    async def execute(self, **kwargs: Any) -> str:
        """Execute the skill generation workflow.

        Args:
            task: Description of what the skill should do.
            skill_name: Explicit skill name.
            description: Explicit skill description.

        Returns:
            Generation result or error message.
        """
        task: str = kwargs.get("task", "")
        skill_name: str | None = kwargs.get("skill_name")
        description: str | None = kwargs.get("description")

        if not task:
            return "Error: No task provided."

        try:
            logger.info(f"Starting skill generation for task: {task}")

            # Step 1: Analyze
            spec = await self._analyze_task(task, skill_name, description)
            logger.info(f"Skill specification: {spec.skill_name}")

            # Step 2: Build plan
            plan = self._create_build_plan(spec)
            skill_dir = os.path.join(self._base_dir, spec.skill_name)
            plan_path = self._write_plan(plan, skill_dir)

            # Step 3: Generate files concurrently
            files = await self._build_from_plan(spec, plan, skill_dir)
            logger.info(f"Generated {len(files)} file(s)")

            # Step 4: Validate
            evaluation = self._evaluate_skill(spec, files)
            if not evaluation.is_valid:
                return (
                    f"Skill generation failed evaluation: {evaluation.reasoning}. "
                    f"Errors: {', '.join(evaluation.errors)}"
                )

            # Step 5: Write to disk
            self._write_skill_directory(spec, files)

            file_list = "\n".join(f"  - {f.filename}" for f in files)
            return (
                f"Successfully generated skill '{spec.skill_name}'.\n"
                f"Description: {spec.description}\n"
                f"Directory: {skill_dir}\n"
                f"Plan: {plan_path}\n"
                f"Files:\n{file_list}"
            )

        except Exception as e:
            logger.error(f"Error in skill generation: {e}")
            return f"Error during skill generation: {e}"

    # ------------------------------------------------------------------
    # Step 1: Task analysis
    # ------------------------------------------------------------------

    async def _analyze_task(
        self, task: str, skill_name: str | None, description: str | None
    ) -> SkillSpecification:
        system_prompt = (
            "You are an expert at designing AI agent skills. "
            "Analyze the given task and design a complete skill specification."
        )

        user_prompt = f"""Design a skill specification for the following task:

Task: {task}

{"Skill name (if specified): " + skill_name if skill_name else ""}
{"Skill description (if specified): " + description if description else ""}

Requirements:
1. skill_name: kebab-case (e.g. "text-translator")
2. description: What the skill does
3. scripts: Python utility scripts needed (filename + purpose)
4. resources: Data/config files needed (filename + purpose)
5. include_examples: Whether examples.md would be useful
6. include_reference: Whether reference.md would be useful
7. implementation_plan: High-level plan"""

        messages = [
            SystemMessage(content=system_prompt),
            UserMessage(content=user_prompt),
        ]

        response = await call_llm(
            model=self._model,
            messages=messages,
            response_format=SkillSpecification,
        )

        if response.parsed_model is not None:
            spec = response.parsed_model
        else:
            spec = SkillSpecification(
                skill_name=skill_name or "generated-skill",
                description=description or task,
                implementation_plan="Generate skill based on task requirements",
            )

        if skill_name:
            spec.skill_name = skill_name
        if description:
            spec.description = description
        return spec

    # ------------------------------------------------------------------
    # Step 2: Build plan
    # ------------------------------------------------------------------

    def _create_build_plan(self, spec: SkillSpecification) -> SkillBuildPlan:
        files: list[SkillFilePlan] = []
        files.append(SkillFilePlan(filename="SKILL.md", purpose="Main instruction file", generator="skill_md"))
        for s in spec.scripts:
            files.append(SkillFilePlan(filename=_prefixed("scripts/", s.filename), purpose=s.purpose, generator="scripts"))
        for r in spec.resources:
            files.append(SkillFilePlan(filename=_prefixed("resources/", r.filename), purpose=r.purpose, generator="resources"))
        if spec.include_examples:
            files.append(SkillFilePlan(filename="examples.md", purpose="Usage examples", generator="docs"))
        if spec.include_reference:
            files.append(SkillFilePlan(filename="reference.md", purpose="API reference", generator="docs"))

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        plan = SkillBuildPlan(
            skill_name=spec.skill_name,
            description=spec.description,
            implementation_plan=spec.implementation_plan,
            files=files,
            created_at=now,
        )
        plan.log.append(f"[{now}] Plan created - {len(files)} file(s) to generate")
        return plan

    def _write_plan(self, plan: SkillBuildPlan, skill_dir: str) -> str:
        os.makedirs(skill_dir, exist_ok=True)
        plan_path = os.path.join(skill_dir, "plan.md")
        lines = [f"# Build Plan: {plan.skill_name}", "", f"> {plan.description}", "", "## Files", ""]
        for fp in plan.files:
            checkbox = "x" if fp.status == "completed" else " "
            lines.append(f"- [{checkbox}] `{fp.filename}` - {fp.purpose}")
        lines.extend(["", "## Log", ""])
        for entry in plan.log:
            lines.append(f"- {entry}")
        with open(plan_path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
        return plan_path

    # ------------------------------------------------------------------
    # Step 3: Generate files
    # ------------------------------------------------------------------

    async def _build_from_plan(
        self, spec: SkillSpecification, plan: SkillBuildPlan, skill_dir: str
    ) -> list[SkillFileContent]:
        generators = {
            "skill_md": self._gen_skill_md,
            "scripts": self._gen_scripts,
            "resources": self._gen_resources,
            "docs": self._gen_docs,
        }
        unique_gens = set(fp.generator for fp in plan.files)
        tasks = [generators[g](spec) for g in unique_gens if g in generators]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        all_files: list[SkillFileContent] = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Generator failed: {result}")
                continue
            all_files.extend(result)

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        generated_names = {f.filename for f in all_files}
        for fp in plan.files:
            fp.status = "completed" if fp.filename in generated_names else "failed"
        plan.log.append(f"[{now}] Build finished - {len(all_files)} total file(s)")
        self._write_plan(plan, skill_dir)
        return all_files

    async def _gen_skill_md(self, spec: SkillSpecification) -> list[SkillFileContent]:
        scripts_ref = "\n".join(f"  - {_prefixed('scripts/', s.filename)}: {s.purpose}" for s in spec.scripts) or "  (none)"
        resources_ref = "\n".join(f"  - {_prefixed('resources/', r.filename)}: {r.purpose}" for r in spec.resources) or "  (none)"

        response = await call_llm(
            model=self._model,
            messages=[
                SystemMessage(content="You are an expert at writing SKILL.md files for AI agent skills."),
                UserMessage(content=f"""Generate a SKILL.md for skill "{spec.skill_name}":
Description: {spec.description}
Plan: {spec.implementation_plan}
Scripts: {scripts_ref}
Resources: {resources_ref}
Start with YAML frontmatter (name, description). Return ONLY the markdown content."""),
            ],
        )
        return [SkillFileContent(filename="SKILL.md", content=_strip_fences(response.message.strip()))]

    async def _gen_scripts(self, spec: SkillSpecification) -> list[SkillFileContent]:
        if not spec.scripts:
            return []
        scripts_desc = "\n".join(f"- {s.filename}: {s.purpose}" for s in spec.scripts)
        response = await call_llm(
            model=self._model,
            messages=[
                SystemMessage(content="You are an expert Python developer. Generate complete Python scripts."),
                UserMessage(content=f"""Generate Python scripts for skill "{spec.skill_name}":
Description: {spec.description}
Scripts: {scripts_desc}
Return a JSON array of {{"filename": "scripts/...", "content": "..."}}. Return ONLY JSON."""),
            ],
        )
        return _parse_file_list(response.message.strip())

    async def _gen_resources(self, spec: SkillSpecification) -> list[SkillFileContent]:
        if not spec.resources:
            return []
        resources_desc = "\n".join(f"- {r.filename}: {r.purpose}" for r in spec.resources)
        response = await call_llm(
            model=self._model,
            messages=[
                SystemMessage(content="You are an expert at creating data and config files."),
                UserMessage(content=f"""Generate resource files for skill "{spec.skill_name}":
Description: {spec.description}
Resources: {resources_desc}
Return a JSON array of {{"filename": "resources/...", "content": "..."}}. Return ONLY JSON."""),
            ],
        )
        return _parse_file_list(response.message.strip())

    async def _gen_docs(self, spec: SkillSpecification) -> list[SkillFileContent]:
        docs = []
        if spec.include_examples:
            docs.append("examples.md")
        if spec.include_reference:
            docs.append("reference.md")
        if not docs:
            return []
        response = await call_llm(
            model=self._model,
            messages=[
                SystemMessage(content="You are an expert technical writer."),
                UserMessage(content=f"""Generate documentation for skill "{spec.skill_name}":
Description: {spec.description}
Files to generate: {json.dumps(docs)}
Return a JSON array of {{"filename": "...", "content": "..."}}. Return ONLY JSON."""),
            ],
        )
        return _parse_file_list(response.message.strip())

    # ------------------------------------------------------------------
    # Step 4: Validation
    # ------------------------------------------------------------------

    def _evaluate_skill(self, spec: SkillSpecification, files: list[SkillFileContent]) -> SkillEvaluation:
        errors: list[str] = []
        warnings: list[str] = []
        filenames = {f.filename for f in files}

        if "SKILL.md" not in filenames:
            errors.append("Missing required file: SKILL.md")

        for f in files:
            if f.filename.endswith(".py"):
                try:
                    ast.parse(f.content)
                except SyntaxError as e:
                    errors.append(f"Syntax error in {f.filename}: {e}")

        for f in files:
            if f.filename.endswith(".json"):
                try:
                    json.loads(f.content)
                except json.JSONDecodeError as e:
                    errors.append(f"Invalid JSON in {f.filename}: {e}")

        if errors:
            return SkillEvaluation(is_valid=False, errors=errors, warnings=warnings, reasoning=f"Validation failed with {len(errors)} error(s)")

        return SkillEvaluation(is_valid=True, errors=errors, warnings=warnings, reasoning="Skill passed all validation checks")

    # ------------------------------------------------------------------
    # Step 5: Write to disk
    # ------------------------------------------------------------------

    def _write_skill_directory(self, spec: SkillSpecification, files: list[SkillFileContent]) -> str:
        skill_dir = os.path.join(self._base_dir, spec.skill_name)
        os.makedirs(skill_dir, exist_ok=True)
        for f in files:
            file_path = os.path.join(skill_dir, f.filename)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "w", encoding="utf-8") as fh:
                fh.write(f.content)
        return skill_dir

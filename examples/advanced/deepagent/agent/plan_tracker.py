"""
PlanTracker — Standalone plan extraction and progress tracking.

Extracts step-by-step plans from LLM output, persists them to ``todo.md``,
and injects concise progress snapshots into the agent context.

This module has **zero** framework-specific imports.  External dependencies
(context updates, LLM calls) are injected via callbacks.
"""

import json
import logging
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protocols for dependency injection
# ---------------------------------------------------------------------------

class LLMMessage(Protocol):
    """Minimal protocol for an LLM response message."""

    @property
    def content(self) -> str | None: ...

    @property
    def tool_calls(self) -> list: ...


LLMCallable = Callable[[list[dict[str, Any]], list], Awaitable[Any]]
"""Async callable that takes (messages, tools) and returns an LLMMessage-like object."""

ContextUpdateCallback = Callable[[str, str], None]
"""Callable(summary_text, label) — upserts a system message into the context."""


# ---------------------------------------------------------------------------
# PlanStepState
# ---------------------------------------------------------------------------

@dataclass
class PlanStepState:
    """Represents a single plan step and its progress."""

    index: int
    description: str
    status: str = "pending"
    updates: list[str] = field(default_factory=list)
    next_hint: str | None = None

    def add_update(self, note: str) -> None:
        """Append a progress note to this step."""
        if note:
            self.updates.append(note)

    def mark_completed(self) -> None:
        """Mark the step as completed."""
        self.status = "completed"


# ---------------------------------------------------------------------------
# PlanTracker
# ---------------------------------------------------------------------------

class PlanTracker:
    """
    Extracts step-by-step plans from LLM output, writes them to ``todo.md``,
    and appends concise progress snapshots into the agent context.

    Dependencies are injected at construction time:

    * *on_context_update* — called with ``(summary, label)`` whenever the plan
      context should be refreshed (maps to ``ContextManager.upsert_system_message``).
    * *llm_call* — optional async LLM callable for plan-update interpretation.
    """

    _plan_header_pattern = re.compile(r"#PLAN#(?P<plan>.*)", re.IGNORECASE | re.DOTALL)
    _plan_step_pattern = re.compile(r"#?\s*(\d+)\.\s*(.+)")
    _step_reference_pattern = re.compile(r"\bstep[\s#:\-]*?(\d+)", re.IGNORECASE)
    _todo_status_block = re.compile(r"<TODO_STATUS>(.*?)</TODO_STATUS>", re.IGNORECASE | re.DOTALL)
    _todo_plan_block = re.compile(r"<TODO_PLAN>(.*?)</TODO_PLAN>", re.IGNORECASE | re.DOTALL)
    _completion_pattern_template = r"step\s*{step}\b.*?(?:completed|complete|done|finished|resolved|answered)"
    _advance_pattern = re.compile(
        r"(?:move|moving|proceed|proceeding|next|now)\s+(?:to|onto)\s+step\s*(\d+)",
        re.IGNORECASE,
    )

    def __init__(
        self,
        base_dir: Path,
        on_context_update: ContextUpdateCallback,
        llm_call: LLMCallable | None = None,
    ) -> None:
        self._todo_path = base_dir / "todo.md"
        self._on_context_update = on_context_update
        self._llm_call = llm_call
        self._steps: list[PlanStepState] = []
        self._last_rendered: str = ""
        self._active_step_index: int | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def has_plan(self) -> bool:
        """Return True if a plan has been extracted."""
        return bool(self._steps)

    async def process_llm_output(self, llm_output: LLMMessage) -> None:
        """Handle an LLM message to extract or update plan progress."""
        content = (llm_output.content or "").strip()
        if not content:
            return

        plan_changed = self._apply_todo_plan_block(content)
        status_changed = self._apply_todo_status_block(content)
        if plan_changed or status_changed:
            self._write_and_append_context(force_write=True, force_context=True)

        if not self._steps or "#PLAN#" in content.upper():
            try:
                if self._extract_plan(content):
                    logger.info("Captured plan from LLM output and initialized todo.md")
                    return
            except Exception as exc:
                logger.warning("Plan extraction failed: %s", exc)
                return

        if not self._steps:
            return

        llm_updated = False
        if self._llm_call:
            try:
                llm_updated = await self._llm_update_plan(content)
            except Exception as exc:
                logger.warning("LLM plan updater failed: %s", exc)
        if llm_updated:
            self._write_and_append_context(force_write=True, force_context=True)
            return

        try:
            self._update_progress(content, bool(llm_output.tool_calls))
            self._write_and_append_context(force_write=True)
        except Exception as exc:
            logger.warning("Plan tracking update failed: %s", exc)

    def finalize(self, final_note: str, mark_remaining_complete: bool) -> None:
        """Persist final plan state once the task is complete or aborted.

        Args:
            final_note: Final agent summary or message to attach.
            mark_remaining_complete: If True, mark any pending steps as completed.
        """
        if not self._steps:
            return

        if mark_remaining_complete:
            for step in self._steps:
                if step.status != "completed":
                    step.mark_completed()
                    self._capture_completion_hint(step)

        target = None
        if self._active_step_index is not None:
            target = next((s for s in self._steps if s.index == self._active_step_index), None)
        if target is None and self._steps:
            target = self._steps[-1]
        if target:
            target.add_update(f"Step {target.index} completed. Final summary provided separately.")

        self._write_and_append_context()

    def get_active_or_next_step(self) -> PlanStepState | None:
        """Return the active step if set, else the next pending step."""
        if not self._steps:
            return None
        if self._active_step_index is not None:
            step = self._get_step_by_index(self._active_step_index)
            if step:
                return step
        for step in self._steps:
            if step.status != "completed":
                return step
        return self._steps[-1]

    # ------------------------------------------------------------------
    # Plan extraction
    # ------------------------------------------------------------------

    def _extract_plan(self, content: str) -> bool:
        """Parse a ``#PLAN#`` block and reset plan state."""
        match = self._plan_header_pattern.search(content)
        plan_body = (match.group("plan") or "") if match else content

        steps: list[PlanStepState] = []
        for line in plan_body.splitlines():
            parsed = self._plan_step_pattern.match(line.strip())
            if parsed:
                index = int(parsed.group(1))
                desc = parsed.group(2).strip()
                if desc:
                    steps.append(PlanStepState(index=index, description=desc))

        if not steps:
            return False

        steps = self._dedupe_and_sort_steps(steps)
        self._steps = steps
        self._active_step_index = self._steps[0].index if self._steps else None
        self._write_and_append_context()
        return True

    # ------------------------------------------------------------------
    # Progress updates
    # ------------------------------------------------------------------

    def _update_progress(self, message: str, has_tool_calls: bool) -> None:
        """Update plan using explicit step references or fall back to active step."""
        step_refs = sorted(self._extract_step_references(message))
        if not step_refs:
            fallback_step = self._get_step_by_index(self._active_step_index)
            if not fallback_step:
                fallback_step = next((s for s in self._steps if s.status != "completed"), None)

            if fallback_step:
                note = self._clean_note(message)
                if note and (not fallback_step.updates or fallback_step.updates[-1] != note):
                    fallback_step.add_update(note)
                if fallback_step.status == "pending":
                    fallback_step.status = "in_progress"
                self._active_step_index = fallback_step.index
                logger.info(
                    "No explicit step reference found; recorded progress under Step %d.",
                    fallback_step.index,
                )
            else:
                logger.info(
                    "No explicit step reference found and no plan steps available to update; persisting todo.md."
                )

            self._write_and_append_context(force_write=True)
            return

        note = self._clean_note(message)
        step_specific_notes = self._extract_step_notes(message)
        last_seen_index: int | None = None

        for idx in step_refs:
            step = self._get_step_by_index(idx)
            if not step:
                logger.warning("Assistant referenced Step %d, which is not in the current plan.", idx)
                continue

            last_seen_index = idx
            note_for_step = step_specific_notes.get(idx, note)
            if note_for_step and (not step.updates or step.updates[-1] != note_for_step):
                step.add_update(note_for_step)

            if self._should_mark_completed(message, idx, has_tool_calls):
                step.mark_completed()
                self._capture_completion_hint(step)
                self._advance_active_step(idx)
            else:
                self._active_step_index = idx

        if last_seen_index is not None and self._active_step_index is None:
            self._active_step_index = last_seen_index

        self._write_and_append_context()

    async def _llm_update_plan(self, message: str) -> bool:
        """Use an LLM to interpret the latest message and propose plan updates."""
        if not self._llm_call or not message or not self._steps:
            return False

        plan_lines = []
        for step in self._steps:
            latest = step.updates[-1] if step.updates else ""
            plan_lines.append(
                f"{step.index}. {step.description} | status={step.status or 'pending'} | latest_note={latest}"
            )
        plan_text = "\n".join(plan_lines)

        system_prompt = (
            "You update a numbered plan. Given the current plan and the latest assistant message, "
            "return JSON only with keys:\n"
            '- plan_updates: list of {"step": int, "status": one of [pending,in_progress,completed,failed], "note": string (optional)}\n'
            "- active_step: int (optional)\n"
            "Do not renumber steps or invent new ones. Keep responses terse JSON only."
        )
        user_prompt = (
            f"Current plan:\n{plan_text}\n\n"
            f"Latest assistant/tool message:\n{message}\n\n"
            "Return the JSON object now."
        )

        resp = await self._llm_call(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            [],
        )
        raw = (resp.content or "").strip()
        if not raw:
            return False
        if not raw.startswith("{"):
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                raw = match.group(0)
        try:
            data = json.loads(raw)
        except Exception as exc:
            logger.debug("LLM plan updater returned non-JSON content: %s (%s)", raw, exc)
            return False

        updates = data.get("plan_updates") or []
        active_step = data.get("active_step")
        status_map = {
            "pending": "pending",
            "in_progress": "in_progress",
            "completed": "completed",
            "failed": "failed",
        }

        changed = False
        for upd in updates:
            try:
                idx = int(upd.get("step"))
            except (TypeError, ValueError):
                continue
            step = self._get_step_by_index(idx)
            if not step:
                continue
            status_val = status_map.get(str(upd.get("status", "")).lower())
            note = upd.get("note")
            if status_val and step.status != status_val:
                step.status = status_val
                changed = True
                if status_val == "completed":
                    self._capture_completion_hint(step)
                    self._advance_active_step(idx)
            if note:
                cleaned = self._clean_note(str(note))
                if cleaned and (not step.updates or step.updates[-1] != cleaned):
                    step.add_update(cleaned)
                    changed = True

        if active_step is not None:
            try:
                active_idx = int(active_step)
                if self._get_step_by_index(active_idx):
                    self._active_step_index = active_idx
                    changed = True
            except (TypeError, ValueError):
                pass

        return changed

    # ------------------------------------------------------------------
    # TODO_PLAN / TODO_STATUS block parsing
    # ------------------------------------------------------------------

    def _apply_todo_plan_block(self, message: str) -> bool:
        """Replace the current plan with steps defined inside ``<TODO_PLAN>...</TODO_PLAN>``."""
        match = self._todo_plan_block.search(message or "")
        if not match:
            return False

        body = match.group(1) or ""
        steps: list[PlanStepState] = []

        for raw_line in body.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            num_match = re.match(r"^\s*(\d+)[\.\)]\s+(.*)", line)
            dash_match = re.match(r"^\s*-\s+(.*)", line)
            desc = None
            if num_match:
                desc = num_match.group(2).strip()
            elif dash_match:
                desc = dash_match.group(1).strip()
            if desc:
                steps.append(PlanStepState(index=len(steps) + 1, description=desc))

        if not steps:
            return False

        steps = self._dedupe_and_sort_steps(steps)
        self._steps = steps
        self._active_step_index = self._steps[0].index if self._steps else None
        return True

    def _apply_todo_status_block(self, message: str) -> bool:
        """Apply status updates defined in a ``<TODO_STATUS>...</TODO_STATUS>`` block."""
        match = self._todo_status_block.search(message or "")
        if not match:
            return False

        block = match.group(1) or ""
        status_pattern = re.compile(r"^\s*(\d+)[\.\)]\s*([A-Z_ ]+)\s*(?:[-|:]\s*(.*))?$", re.IGNORECASE)
        status_map = {
            "DONE": "completed",
            "COMPLETED": "completed",
            "SUCCESS": "completed",
            "FAILED": "failed",
            "FAIL": "failed",
            "BLOCKED": "failed",
            "IN_PROGRESS": "in_progress",
            "WORKING": "in_progress",
            "PENDING": "pending",
            "TODO": "pending",
        }

        changed = False
        for raw_line in block.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            m = status_pattern.match(line)
            if not m:
                continue
            try:
                idx = int(m.group(1))
            except (TypeError, ValueError):
                continue
            status_key = (m.group(2) or "").upper().replace(" ", "_")
            status_val = status_map.get(status_key)
            if not status_val:
                continue
            note = (m.group(3) or "").strip()
            step = self._get_step_by_index(idx)
            if not step:
                continue
            if step.status != status_val:
                step.status = status_val
                changed = True
            if note and (not step.updates or step.updates[-1] != note):
                step.add_update(note)
                changed = True

            if status_val == "completed":
                self._capture_completion_hint(step)
                self._advance_active_step(idx)
            elif status_val == "in_progress":
                self._active_step_index = idx

        return changed

    # ------------------------------------------------------------------
    # Step reference helpers
    # ------------------------------------------------------------------

    def _extract_step_references(self, message: str) -> set[int]:
        """Return all step indices explicitly mentioned in the message."""
        refs: set[int] = set()
        for match in self._step_reference_pattern.finditer(message):
            try:
                refs.add(int(match.group(1)))
            except (ValueError, TypeError):
                continue
        return refs

    def _extract_step_notes(self, message: str) -> dict[int, str]:
        """Return a mapping of step index to the note text for that step section."""
        notes: dict[int, str] = {}
        matches = list(re.finditer(self._step_reference_pattern, message))
        if not matches:
            return notes

        for i, match in enumerate(matches):
            try:
                idx = int(match.group(1))
            except (ValueError, TypeError):
                continue
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(message)
            segment = message[start:end].strip()
            cleaned = self._clean_note(segment)
            if cleaned:
                notes[idx] = cleaned
        return notes

    def _get_step_by_index(self, index: int | None) -> PlanStepState | None:
        """Look up a step by its index number."""
        if index is None:
            return None
        for step in self._steps:
            if step.index == index:
                return step
        return None

    def _dedupe_and_sort_steps(self, steps: list[PlanStepState]) -> list[PlanStepState]:
        """Ensure one entry per step index (last occurrence wins) and sort by index."""
        deduped: dict[int, PlanStepState] = {}
        for step in steps:
            deduped[step.index] = step
        return [deduped[idx] for idx in sorted(deduped)]

    # ------------------------------------------------------------------
    # Completion and advancement
    # ------------------------------------------------------------------

    def _advance_active_step(self, completed_index: int) -> None:
        """Move the active pointer to the next pending step after a completion."""
        for step in self._steps:
            if step.index > completed_index and step.status != "completed":
                self._active_step_index = step.index
                return
        self._active_step_index = completed_index

    def _should_mark_completed(self, message: str, step_index: int, has_tool_calls: bool) -> bool:
        """Determine if a step should be marked completed based on the latest message."""
        completion_pattern = re.compile(
            self._completion_pattern_template.format(step=step_index),
            re.IGNORECASE,
        )
        if completion_pattern.search(message):
            return True

        advance = self._advance_pattern.search(message)
        if advance:
            try:
                next_step = int(advance.group(1))
                if next_step - 1 == step_index:
                    return True
            except ValueError:
                pass

        lower_msg = message.lower()
        if any(phrase in lower_msg for phrase in ["next step", "move on to the next step"]) and any(
            step.index > step_index and step.status != "completed" for step in self._steps
        ):
            return True

        success_terms = {
            "created", "written", "saved", "downloaded", "completed", "done",
            "success", "successful", "successfully", "finished", "verified",
            "exists", "ready",
        }
        step_obj = self._get_step_by_index(step_index)
        step_tokens = set(re.findall(r"[a-zA-Z]{3,}", step_obj.description.lower() if step_obj else ""))
        msg_tokens = set(re.findall(r"[a-zA-Z]{3,}", lower_msg))
        return bool(success_terms & msg_tokens and step_tokens and (step_tokens & msg_tokens))

    def _capture_completion_hint(self, step: PlanStepState) -> None:
        """Persist the 'Next' hint when a step is completed."""
        if step.next_hint:
            return
        hint = self._next_action_hint(step)
        if hint:
            step.next_hint = self._clean_note(hint, 200)

    # ------------------------------------------------------------------
    # Rendering and persistence
    # ------------------------------------------------------------------

    def _clean_note(self, note: str, limit: int = 500) -> str:
        """Compact whitespace and truncate a note to *limit* characters."""
        if not note:
            return ""
        compact = " ".join(note.split())
        if len(compact) > limit:
            return compact[:limit] + "..."
        return compact

    def _write_and_append_context(self, force_write: bool = False, force_context: bool = False) -> None:
        """Persist ``todo.md`` and push a concise snapshot into the agent context."""
        rendered = self._render_markdown()
        changed = rendered != self._last_rendered

        if changed or force_write:
            self._todo_path.parent.mkdir(parents=True, exist_ok=True)
            self._todo_path.write_text(rendered, encoding="utf-8")
            self._last_rendered = rendered

        if changed or force_context:
            summary = self._render_context_summary()
            try:
                self._on_context_update(
                    summary,
                    "This is the plan list and what has been accomplished so far",
                )
            except Exception as exc:
                logger.warning("Failed to append plan summary to context: %s", exc)

    def _render_markdown(self) -> str:
        """Render the plan as a Markdown checklist for ``todo.md``."""
        lines = ["# Agent Plan", ""]
        for step in self._steps:
            status_box = self._status_box(step.status)
            lines.append(f"- {status_box} Step {step.index}: {step.description}")
            latest_update = step.updates[-1] if step.updates else "Pending"
            lines.append(f"  - Update: {self._clean_note(latest_update, 500)}")
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"

    def _render_context_summary(self) -> str:
        """Render a compact context summary for the agent's system prompt."""
        lines = [
            "This is the plan list and what has been accomplished so far",
            "Plan progress snapshot (todo.md):",
        ]
        for step in self._steps:
            status = step.status
            latest = step.updates[-1] if step.updates else "No updates yet."
            lines.append(
                f"[{status}] Step {step.index}: {step.description} | Last update: {self._clean_note(latest, 160)}"
            )
        return "\n".join(lines)

    def _status_box(self, status: str) -> str:
        """Return an emoji for the given status string."""
        status_lower = (status or "").lower()
        if status_lower == "completed":
            return "\u2705"
        if status_lower == "failed":
            return "\u274c"
        if status_lower == "in_progress":
            return "\U0001f504"
        return "\u270b"

    def _next_action_hint(self, step: PlanStepState) -> str:
        """Return a short hint about what to do next relative to *step*."""
        if (step.status or "").lower() == "completed":
            pending = next(
                (s for s in self._steps if s.index > step.index and s.status != "completed"),
                None,
            )
            return pending.description if pending else "All steps completed."
        if (step.status or "").lower() == "in_progress":
            return "Continue this step."
        return "Work on this step."

"""InstructionOptimizer — LLM prompt optimization via textual gradients."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from orbiter.train.evolution import (  # pyright: ignore[reportMissingImports]
    EvolutionStrategy,
)
from orbiter.train.operator.llm_call import (  # pyright: ignore[reportMissingImports]
    LLMCallOperator,
)


class InstructionOptimizer(EvolutionStrategy):
    """Optimizer that improves system prompts via textual gradients.

    Analyzes evaluation cases (especially failures) to generate natural-
    language "gradients" describing how each operator's prompts should
    change, then applies those gradients to produce improved prompts.

    Implements :class:`EvolutionStrategy` so it can be plugged directly
    into an :class:`EvolutionPipeline`.

    Args:
        model: Model identifier (e.g., ``"gpt-4"``).
        operators: LLM call operators whose prompts will be optimized.
        llm_fn: Async callable with signature
            ``(*, system_prompt: str, user_prompt: str, **kw) -> str``
            used for meta-LLM calls (gradient generation and prompt
            rewriting).  If *None*, the optimizer produces placeholder
            gradients and does not rewrite prompts.
    """

    __slots__ = ("_gradients", "_llm_fn", "_model", "_operators")

    def __init__(
        self,
        model: str,
        operators: list[LLMCallOperator],
        *,
        llm_fn: Any = None,
    ) -> None:
        self._model = model
        self._operators = list(operators)
        self._llm_fn = llm_fn
        self._gradients: list[str] = []

    @property
    def model(self) -> str:
        """Model identifier used for meta-LLM calls."""
        return self._model

    @property
    def operators(self) -> list[LLMCallOperator]:
        """Operators whose prompts are being optimized."""
        return list(self._operators)

    @property
    def gradients(self) -> list[str]:
        """Most recently computed textual gradients."""
        return list(self._gradients)

    # ------------------------------------------------------------------
    # Core optimization API
    # ------------------------------------------------------------------

    async def backward(
        self, evaluated_cases: Sequence[dict[str, Any]]
    ) -> list[str]:
        """Generate textual gradients from evaluated cases.

        Analyses failures and low-scoring cases to produce natural-language
        descriptions of how each operator's prompts should change.

        Args:
            evaluated_cases: Dicts with at least ``input``, ``output``,
                and optionally ``expected``, ``score``, ``status``.

        Returns:
            One gradient string per operator.
        """
        if not evaluated_cases or not self._operators:
            self._gradients = []
            return []

        # Identify weak cases (score < 1.0 or status == "failed").
        weak = [
            c
            for c in evaluated_cases
            if c.get("score", 1.0) < 1.0 or c.get("status") == "failed"
        ]
        if not weak:
            # No clear failures — use all cases for analysis.
            weak = list(evaluated_cases)

        gradients: list[str] = []
        for op in self._operators:
            state = op.get_state()
            prompt_text = self._build_backward_prompt(state, weak)

            if self._llm_fn is not None:
                raw = await self._llm_fn(
                    system_prompt=(
                        "You are an expert prompt engineer. Analyse the "
                        "following evaluation failures and describe concisely "
                        "how the system prompt should be improved."
                    ),
                    user_prompt=prompt_text,
                    model=self._model,
                )
                gradients.append(str(raw))
            else:
                gradients.append(
                    f"Improve prompt for '{op.name}': "
                    f"{len(weak)} weak case(s) detected"
                )

        self._gradients = gradients
        return list(gradients)

    async def step(self) -> dict[str, str]:
        """Apply accumulated gradients to produce improved prompts.

        Uses the meta-LLM to rewrite each operator's ``system_prompt``
        based on the textual gradients from :meth:`backward`.  Template
        variables (``{{...}}``) are preserved across rewrites.

        Returns:
            Mapping of operator name → new system prompt.
        """
        if not self._gradients:
            return {}

        improved: dict[str, str] = {}
        for op, gradient in zip(self._operators, self._gradients):
            state = op.get_state()
            current = state.get("system_prompt", "")

            if self._llm_fn is not None:
                rewrite_prompt = self._build_step_prompt(current, gradient)
                raw = await self._llm_fn(
                    system_prompt=(
                        "You are an expert prompt engineer. Rewrite the "
                        "system prompt incorporating the suggested "
                        "improvements. Keep it concise and preserve all "
                        "template variables exactly as they appear."
                    ),
                    user_prompt=rewrite_prompt,
                    model=self._model,
                )
                new_prompt = str(raw)
            else:
                new_prompt = current

            # Guarantee template variables survive the rewrite.
            new_prompt = self._preserve_templates(current, new_prompt)

            # Apply the updated prompt to the operator.
            op.load_state({**state, "system_prompt": new_prompt})
            improved[op.name] = new_prompt

        self._gradients = []
        return improved

    # ------------------------------------------------------------------
    # EvolutionStrategy interface
    # ------------------------------------------------------------------

    async def synthesise(
        self,
        agent: Any,
        data: Sequence[dict[str, Any]],
        epoch: int,
    ) -> list[dict[str, Any]]:
        """Pass-through — InstructionOptimizer does not synthesise data."""
        return list(data)

    async def train(
        self,
        agent: Any,
        data: Sequence[dict[str, Any]],
        epoch: int,
    ) -> float:
        """Run backward + step as the 'training' phase."""
        gradients = await self.backward(list(data))
        if gradients:
            await self.step()
        return 0.0

    async def evaluate(
        self,
        agent: Any,
        data: Sequence[dict[str, Any]],
        epoch: int,
    ) -> float:
        """No-op — evaluation is handled externally."""
        return 0.0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_backward_prompt(
        state: dict[str, Any],
        weak_cases: Sequence[dict[str, Any]],
    ) -> str:
        """Build the analysis prompt for gradient generation."""
        lines = [
            f"Current system_prompt: {state.get('system_prompt', '')}",
            f"Current user_prompt: {state.get('user_prompt', '')}",
            "",
            "Failure cases:",
        ]
        for i, case in enumerate(weak_cases[:10]):  # cap at 10
            lines.append(
                f"  {i + 1}. input={case.get('input', '')!r} "
                f"output={case.get('output', '')!r} "
                f"expected={case.get('expected', '')!r} "
                f"score={case.get('score', 'N/A')}"
            )
        lines.append("")
        lines.append("Describe how the system prompt should be changed.")
        return "\n".join(lines)

    @staticmethod
    def _build_step_prompt(current_prompt: str, gradient: str) -> str:
        """Build the rewrite prompt for prompt improvement."""
        return (
            f"Current system prompt:\n{current_prompt}\n\n"
            f"Suggested improvements:\n{gradient}\n\n"
            "Rewrite the system prompt with these improvements applied. "
            "Preserve all template variables (e.g., {{query}}) exactly."
        )

    @staticmethod
    def _preserve_templates(original: str, rewritten: str) -> str:
        """Ensure all ``{{...}}`` template variables from *original* appear in *rewritten*."""
        templates = set(re.findall(r"\{\{[^}]+\}\}", original))
        for tmpl in templates:
            if tmpl not in rewritten:
                rewritten = f"{rewritten} {tmpl}"
        return rewritten

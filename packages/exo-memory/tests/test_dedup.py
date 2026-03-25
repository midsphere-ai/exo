"""Tests for LLM-based memory deduplication."""

from __future__ import annotations

import json

import pytest

from exo.memory.base import MemoryItem  # pyright: ignore[reportMissingImports]
from exo.memory.dedup import (  # pyright: ignore[reportMissingImports]
    MemUpdateChecker,
    MergeResult,
    UpdateDecision,
    _parse_llm_response,
)
from exo.memory.long_term import (  # pyright: ignore[reportMissingImports]
    LongTermMemory,
)

# ---------------------------------------------------------------------------
# Mock LLM checker
# ---------------------------------------------------------------------------


class MockChecker:
    """Mock LLM checker that returns pre-configured JSON responses."""

    def __init__(self, response: dict) -> None:
        self._response = json.dumps(response)
        self.call_count = 0
        self.last_prompt: str | None = None

    async def __call__(self, prompt: str) -> str:
        self.call_count += 1
        self.last_prompt = prompt
        return self._response


class FailingChecker:
    """Mock checker that raises."""

    async def __call__(self, prompt: str) -> str:
        msg = "LLM unavailable"
        raise RuntimeError(msg)


# ===========================================================================
# UpdateDecision
# ===========================================================================


class TestUpdateDecision:
    def test_values(self) -> None:
        assert UpdateDecision.ADD == "add"
        assert UpdateDecision.DELETE == "delete"
        assert UpdateDecision.MERGE == "merge"
        assert UpdateDecision.SKIP == "skip"

    def test_all_members(self) -> None:
        assert len(UpdateDecision) == 4


# ===========================================================================
# MergeResult
# ===========================================================================


class TestMergeResult:
    def test_defaults(self) -> None:
        result = MergeResult(decision=UpdateDecision.ADD)
        assert result.decision == UpdateDecision.ADD
        assert result.merged_content is None
        assert result.delete_ids == []

    def test_with_merge_content(self) -> None:
        result = MergeResult(
            decision=UpdateDecision.MERGE,
            merged_content="combined facts",
            delete_ids=["id1", "id2"],
        )
        assert result.merged_content == "combined facts"
        assert result.delete_ids == ["id1", "id2"]

    def test_frozen(self) -> None:
        result = MergeResult(decision=UpdateDecision.ADD)
        with pytest.raises(AttributeError):
            result.decision = UpdateDecision.SKIP  # type: ignore[misc]


# ===========================================================================
# MemUpdateChecker — exact match fallback (no LLM)
# ===========================================================================


class TestExactMatchFallback:
    async def test_add_when_no_existing(self) -> None:
        checker = MemUpdateChecker()
        item = MemoryItem(content="new fact", memory_type="facts")
        result = await checker.check(item, [])
        assert result.decision == UpdateDecision.ADD

    async def test_skip_on_exact_duplicate(self) -> None:
        checker = MemUpdateChecker()
        existing = MemoryItem(content="sky is blue", memory_type="facts")
        new_item = MemoryItem(content="sky is blue", memory_type="facts")
        result = await checker.check(new_item, [existing])
        assert result.decision == UpdateDecision.SKIP

    async def test_add_when_different_content(self) -> None:
        checker = MemUpdateChecker()
        existing = MemoryItem(content="sky is blue", memory_type="facts")
        new_item = MemoryItem(content="grass is green", memory_type="facts")
        result = await checker.check(new_item, [existing])
        assert result.decision == UpdateDecision.ADD

    async def test_add_when_different_type(self) -> None:
        checker = MemUpdateChecker()
        existing = MemoryItem(content="same content", memory_type="facts")
        new_item = MemoryItem(content="same content", memory_type="user_profile")
        result = await checker.check(new_item, [existing])
        assert result.decision == UpdateDecision.ADD

    async def test_repr_exact(self) -> None:
        checker = MemUpdateChecker()
        assert "exact" in repr(checker)


# ===========================================================================
# MemUpdateChecker — LLM-based dedup
# ===========================================================================


class TestLLMDedup:
    async def test_llm_add(self) -> None:
        mock = MockChecker({"decision": "add"})
        checker = MemUpdateChecker(checker=mock)
        existing = MemoryItem(id="e1", content="sky is blue", memory_type="facts")
        new_item = MemoryItem(content="grass is green", memory_type="facts")
        result = await checker.check(new_item, [existing])
        assert result.decision == UpdateDecision.ADD
        assert mock.call_count == 1

    async def test_llm_skip(self) -> None:
        mock = MockChecker({"decision": "skip"})
        checker = MemUpdateChecker(checker=mock)
        existing = MemoryItem(id="e1", content="sky is blue", memory_type="facts")
        new_item = MemoryItem(content="the sky is blue", memory_type="facts")
        result = await checker.check(new_item, [existing])
        assert result.decision == UpdateDecision.SKIP

    async def test_llm_merge(self) -> None:
        mock = MockChecker(
            {
                "decision": "merge",
                "merged_content": "sky is blue and vast",
                "delete_ids": ["e1"],
            }
        )
        checker = MemUpdateChecker(checker=mock)
        existing = MemoryItem(id="e1", content="sky is blue", memory_type="facts")
        new_item = MemoryItem(content="sky is vast", memory_type="facts")
        result = await checker.check(new_item, [existing])
        assert result.decision == UpdateDecision.MERGE
        assert result.merged_content == "sky is blue and vast"
        assert result.delete_ids == ["e1"]

    async def test_llm_delete(self) -> None:
        mock = MockChecker({"decision": "delete", "delete_ids": ["e1"]})
        checker = MemUpdateChecker(checker=mock)
        existing = MemoryItem(id="e1", content="old info", memory_type="facts")
        new_item = MemoryItem(content="updated info", memory_type="facts")
        result = await checker.check(new_item, [existing])
        assert result.decision == UpdateDecision.DELETE
        assert result.delete_ids == ["e1"]

    async def test_top_k_limits_candidates(self) -> None:
        mock = MockChecker({"decision": "add"})
        checker = MemUpdateChecker(checker=mock, top_k=2)
        existing = [
            MemoryItem(id=f"e{i}", content=f"fact {i}", memory_type="facts") for i in range(5)
        ]
        new_item = MemoryItem(content="new fact", memory_type="facts")
        await checker.check(new_item, existing)
        # Prompt should only contain top 2 existing items
        assert mock.last_prompt is not None
        assert "e0" in mock.last_prompt
        assert "e1" in mock.last_prompt
        assert "e2" not in mock.last_prompt

    async def test_prompt_includes_content(self) -> None:
        mock = MockChecker({"decision": "add"})
        checker = MemUpdateChecker(checker=mock)
        existing = MemoryItem(id="e1", content="existing fact", memory_type="facts")
        new_item = MemoryItem(content="new fact", memory_type="facts")
        await checker.check(new_item, [existing])
        assert mock.last_prompt is not None
        assert "new fact" in mock.last_prompt
        assert "existing fact" in mock.last_prompt

    async def test_repr_llm(self) -> None:
        mock = MockChecker({"decision": "add"})
        checker = MemUpdateChecker(checker=mock)
        assert "llm" in repr(checker)


# ===========================================================================
# _parse_llm_response
# ===========================================================================


class TestParseLLMResponse:
    def test_valid_json(self) -> None:
        result = _parse_llm_response('{"decision": "skip"}')
        assert result.decision == UpdateDecision.SKIP

    def test_with_merge_content(self) -> None:
        resp = json.dumps(
            {
                "decision": "merge",
                "merged_content": "combined",
                "delete_ids": ["a", "b"],
            }
        )
        result = _parse_llm_response(resp)
        assert result.decision == UpdateDecision.MERGE
        assert result.merged_content == "combined"
        assert result.delete_ids == ["a", "b"]

    def test_markdown_fenced_json(self) -> None:
        resp = '```json\n{"decision": "add"}\n```'
        result = _parse_llm_response(resp)
        assert result.decision == UpdateDecision.ADD

    def test_invalid_json_defaults_to_add(self) -> None:
        result = _parse_llm_response("not json at all")
        assert result.decision == UpdateDecision.ADD

    def test_unknown_decision_defaults_to_add(self) -> None:
        result = _parse_llm_response('{"decision": "unknown_action"}')
        assert result.decision == UpdateDecision.ADD

    def test_missing_decision_defaults_to_add(self) -> None:
        result = _parse_llm_response('{"merged_content": "hello"}')
        assert result.decision == UpdateDecision.ADD

    def test_delete_ids_not_list_defaults_to_empty(self) -> None:
        result = _parse_llm_response('{"decision": "delete", "delete_ids": "not-a-list"}')
        assert result.delete_ids == []

    def test_uppercase_decision(self) -> None:
        result = _parse_llm_response('{"decision": "SKIP"}')
        assert result.decision == UpdateDecision.SKIP


# ===========================================================================
# LongTermMemory integration with MemUpdateChecker
# ===========================================================================


class TestLongTermMemoryWithChecker:
    async def test_no_checker_preserves_original_behavior(self) -> None:
        store = LongTermMemory()
        await store.add(MemoryItem(content="fact A", memory_type="facts"))
        await store.add(MemoryItem(content="fact A", memory_type="facts"))
        assert len(store) == 1  # exact dedup

    async def test_checker_add(self) -> None:
        mock = MockChecker({"decision": "add"})
        checker = MemUpdateChecker(checker=mock)
        store = LongTermMemory(update_checker=checker)
        await store.add(MemoryItem(content="fact A", memory_type="facts"))
        await store.add(MemoryItem(content="fact B", memory_type="facts"))
        assert len(store) == 2

    async def test_checker_skip(self) -> None:
        mock = MockChecker({"decision": "skip"})
        checker = MemUpdateChecker(checker=mock)
        store = LongTermMemory(update_checker=checker)
        await store.add(MemoryItem(content="fact A", memory_type="facts"))
        # Second add: checker says skip
        await store.add(MemoryItem(content="similar A", memory_type="facts"))
        assert len(store) == 1

    async def test_checker_delete(self) -> None:
        checker_no_llm = MemUpdateChecker()  # exact match for first add
        store = LongTermMemory(update_checker=checker_no_llm)

        item1 = MemoryItem(id="old1", content="old info", memory_type="facts")
        await store.add(item1)
        assert len(store) == 1

        # Now switch to LLM checker that says DELETE
        mock = MockChecker({"decision": "delete", "delete_ids": ["old1"]})
        llm_checker = MemUpdateChecker(checker=mock)
        store._update_checker = llm_checker

        await store.add(MemoryItem(content="new info", memory_type="facts"))
        # old1 deleted, new item added
        assert len(store) == 1
        assert await store.get("old1") is None

    async def test_checker_merge(self) -> None:
        checker_no_llm = MemUpdateChecker()
        store = LongTermMemory(update_checker=checker_no_llm)

        item1 = MemoryItem(id="m1", content="sky is blue", memory_type="facts")
        await store.add(item1)

        # Switch to LLM checker that says MERGE
        mock = MockChecker(
            {
                "decision": "merge",
                "merged_content": "sky is blue and vast",
                "delete_ids": ["m1"],
            }
        )
        llm_checker = MemUpdateChecker(checker=mock)
        store._update_checker = llm_checker

        new_item = MemoryItem(content="sky is vast", memory_type="facts")
        await store.add(new_item)

        # m1 deleted, new merged item stored
        assert len(store) == 1
        assert await store.get("m1") is None
        results = await store.search(query="sky")
        assert len(results) == 1
        assert results[0].content == "sky is blue and vast"

    async def test_checker_merge_without_content_uses_original(self) -> None:
        mock = MockChecker(
            {
                "decision": "merge",
                "merged_content": None,
                "delete_ids": [],
            }
        )
        checker = MemUpdateChecker(checker=mock)
        store = LongTermMemory(update_checker=checker)

        item = MemoryItem(content="original content", memory_type="facts")
        await store.add(item)
        # Content unchanged when merged_content is None
        results = await store.search()
        assert results[0].content == "original content"

    async def test_first_add_no_existing_items(self) -> None:
        """First add with checker should go through (no existing to compare)."""
        mock = MockChecker({"decision": "skip"})  # Would skip if called
        checker = MemUpdateChecker(checker=mock)
        store = LongTermMemory(update_checker=checker)

        await store.add(MemoryItem(content="first item", memory_type="facts"))
        assert len(store) == 1
        # Checker's check() returns ADD when no existing items, so LLM never called
        assert mock.call_count == 0

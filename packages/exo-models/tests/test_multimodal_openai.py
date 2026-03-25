"""Tests for OpenAI multimodal content conversion helpers."""

from __future__ import annotations

import logging

import pytest

from exo.models.openai import _content_blocks_to_openai, _to_openai_messages
from exo.types import (
    AudioBlock,
    DocumentBlock,
    ImageDataBlock,
    ImageURLBlock,
    TextBlock,
    ToolResult,
    UserMessage,
    VideoBlock,
)


class TestContentBlocksToOpenAI:
    def test_text_block(self) -> None:
        parts = _content_blocks_to_openai([TextBlock(text="hello")])
        assert parts == [{"type": "text", "text": "hello"}]

    def test_image_url_block(self) -> None:
        parts = _content_blocks_to_openai(
            [ImageURLBlock(url="https://example.com/img.png", detail="high")]
        )
        assert parts == [
            {
                "type": "image_url",
                "image_url": {"url": "https://example.com/img.png", "detail": "high"},
            }
        ]

    def test_image_data_block(self) -> None:
        parts = _content_blocks_to_openai([ImageDataBlock(data="abc123", media_type="image/png")])
        assert len(parts) == 1
        assert parts[0]["type"] == "image_url"
        assert parts[0]["image_url"]["url"] == "data:image/png;base64,abc123"

    def test_audio_block(self) -> None:
        parts = _content_blocks_to_openai([AudioBlock(data="audiodata", format="wav")])
        assert parts == [
            {"type": "input_audio", "input_audio": {"data": "audiodata", "format": "wav"}}
        ]

    def test_video_block_skipped(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING):
            parts = _content_blocks_to_openai([VideoBlock(url="gs://bucket/video.mp4")])
        assert parts == []
        assert "video" in caplog.text.lower()

    def test_document_block_skipped(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING):
            parts = _content_blocks_to_openai([DocumentBlock(data="pdfdata")])
        assert parts == []
        assert "document" in caplog.text.lower()

    def test_multiple_blocks(self) -> None:
        parts = _content_blocks_to_openai(
            [
                TextBlock(text="describe"),
                ImageURLBlock(url="https://x.com/img.jpg"),
            ]
        )
        assert len(parts) == 2
        assert parts[0]["type"] == "text"
        assert parts[1]["type"] == "image_url"


class TestToOpenAIMessagesMultimodal:
    def test_user_message_string_unchanged(self) -> None:
        from exo.types import UserMessage

        msgs = _to_openai_messages([UserMessage(content="hello")])
        assert msgs == [{"role": "user", "content": "hello"}]

    def test_user_message_with_blocks(self) -> None:
        blocks = [TextBlock(text="hi"), ImageURLBlock(url="https://x.com/a.png")]
        msgs = _to_openai_messages([UserMessage(content=blocks)])
        assert msgs[0]["role"] == "user"
        assert isinstance(msgs[0]["content"], list)
        assert len(msgs[0]["content"]) == 2

    def test_tool_result_string_content(self) -> None:
        tr = ToolResult(tool_call_id="id1", tool_name="tool", content="result")
        msgs = _to_openai_messages([tr])
        assert msgs == [{"role": "tool", "tool_call_id": "id1", "content": "result"}]

    def test_tool_result_error_content(self) -> None:
        tr = ToolResult(tool_call_id="id1", tool_name="tool", error="boom")
        msgs = _to_openai_messages([tr])
        assert msgs[0]["content"] == "boom"

    def test_tool_result_with_media_injects_synthetic_user_message(self) -> None:
        blocks = [ImageURLBlock(url="https://x.com/generated.png")]
        tr = ToolResult(tool_call_id="id1", tool_name="dalle", content=blocks)
        msgs = _to_openai_messages([tr])
        # Should have tool message with placeholder + synthetic user message with media
        tool_msgs = [m for m in msgs if m["role"] == "tool"]
        user_msgs = [m for m in msgs if m["role"] == "user"]
        assert len(tool_msgs) == 1
        assert tool_msgs[0]["content"] == "[media result]"
        assert len(user_msgs) == 1
        assert isinstance(user_msgs[0]["content"], list)
        assert user_msgs[0]["content"][0]["type"] == "image_url"

    def test_tool_result_with_empty_media_no_synthetic_message(self) -> None:
        # Media blocks that all get skipped (video, document) produce no synthetic message
        blocks = [VideoBlock(url="gs://bucket/v.mp4")]
        tr = ToolResult(tool_call_id="id1", tool_name="tool", content=blocks)
        msgs = _to_openai_messages([tr])
        # Tool message present but no synthetic user message (video was skipped)
        assert any(m["role"] == "tool" for m in msgs)
        user_msgs = [m for m in msgs if m["role"] == "user"]
        assert len(user_msgs) == 0

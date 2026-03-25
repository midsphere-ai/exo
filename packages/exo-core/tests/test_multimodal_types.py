"""Tests for multimodal content block types in exo.types."""

import pytest
from pydantic import ValidationError

from exo._internal.message_builder import _content_is_empty
from exo.types import (
    AssistantMessage,
    AudioBlock,
    ContentBlock,
    DocumentBlock,
    ImageDataBlock,
    ImageURLBlock,
    TextBlock,
    ToolResult,
    UserMessage,
    VideoBlock,
)

# ---------------------------------------------------------------------------
# Individual block construction
# ---------------------------------------------------------------------------


class TestTextBlock:
    def test_create(self) -> None:
        b = TextBlock(text="hello")
        assert b.type == "text"
        assert b.text == "hello"

    def test_frozen(self) -> None:
        b = TextBlock(text="hi")
        with pytest.raises(ValidationError):
            b.text = "bye"  # type: ignore[misc]


class TestImageURLBlock:
    def test_defaults(self) -> None:
        b = ImageURLBlock(url="https://example.com/img.png")
        assert b.type == "image_url"
        assert b.url == "https://example.com/img.png"
        assert b.detail == "auto"

    def test_custom_detail(self) -> None:
        b = ImageURLBlock(url="https://x.com/a.jpg", detail="high")
        assert b.detail == "high"

    def test_data_uri(self) -> None:
        b = ImageURLBlock(url="data:image/png;base64,abc123")
        assert b.url.startswith("data:")


class TestImageDataBlock:
    def test_defaults(self) -> None:
        b = ImageDataBlock(data="base64data")
        assert b.type == "image_data"
        assert b.media_type == "image/jpeg"

    def test_custom_media_type(self) -> None:
        b = ImageDataBlock(data="abc", media_type="image/png")
        assert b.media_type == "image/png"


class TestAudioBlock:
    def test_defaults(self) -> None:
        b = AudioBlock(data="audiodata")
        assert b.type == "audio"
        assert b.format == "mp3"

    def test_wav_format(self) -> None:
        b = AudioBlock(data="wavdata", format="wav")
        assert b.format == "wav"


class TestVideoBlock:
    def test_url_variant(self) -> None:
        b = VideoBlock(url="gs://bucket/video.mp4")
        assert b.type == "video"
        assert b.url == "gs://bucket/video.mp4"
        assert b.data is None

    def test_data_variant(self) -> None:
        b = VideoBlock(data="base64videodata")
        assert b.data == "base64videodata"
        assert b.url is None

    def test_default_media_type(self) -> None:
        b = VideoBlock(data="x")
        assert b.media_type == "video/mp4"


class TestDocumentBlock:
    def test_create(self) -> None:
        b = DocumentBlock(data="pdfdata")
        assert b.type == "document"
        assert b.media_type == "application/pdf"
        assert b.title is None

    def test_with_title(self) -> None:
        b = DocumentBlock(data="pdfdata", title="Report Q1")
        assert b.title == "Report Q1"


# ---------------------------------------------------------------------------
# Discriminated union
# ---------------------------------------------------------------------------


class TestContentBlockUnion:
    def test_text_block_via_dict(self) -> None:
        from pydantic import TypeAdapter

        ta: TypeAdapter[ContentBlock] = TypeAdapter(ContentBlock)
        block = ta.validate_python({"type": "text", "text": "hello"})
        assert isinstance(block, TextBlock)

    def test_image_url_via_dict(self) -> None:
        from pydantic import TypeAdapter

        ta: TypeAdapter[ContentBlock] = TypeAdapter(ContentBlock)
        block = ta.validate_python({"type": "image_url", "url": "https://x.com/a.png"})
        assert isinstance(block, ImageURLBlock)

    def test_image_data_via_dict(self) -> None:
        from pydantic import TypeAdapter

        ta: TypeAdapter[ContentBlock] = TypeAdapter(ContentBlock)
        block = ta.validate_python({"type": "image_data", "data": "abc"})
        assert isinstance(block, ImageDataBlock)

    def test_audio_via_dict(self) -> None:
        from pydantic import TypeAdapter

        ta: TypeAdapter[ContentBlock] = TypeAdapter(ContentBlock)
        block = ta.validate_python({"type": "audio", "data": "abc"})
        assert isinstance(block, AudioBlock)

    def test_video_via_dict(self) -> None:
        from pydantic import TypeAdapter

        ta: TypeAdapter[ContentBlock] = TypeAdapter(ContentBlock)
        block = ta.validate_python({"type": "video", "url": "gs://bucket/v.mp4"})
        assert isinstance(block, VideoBlock)

    def test_document_via_dict(self) -> None:
        from pydantic import TypeAdapter

        ta: TypeAdapter[ContentBlock] = TypeAdapter(ContentBlock)
        block = ta.validate_python({"type": "document", "data": "pdfdata"})
        assert isinstance(block, DocumentBlock)


# ---------------------------------------------------------------------------
# Updated message types
# ---------------------------------------------------------------------------


class TestUserMessageMultimodal:
    def test_string_content(self) -> None:
        msg = UserMessage(content="hello")
        assert msg.content == "hello"

    def test_list_content(self) -> None:
        blocks: list[ContentBlock] = [
            TextBlock(text="describe this"),
            ImageURLBlock(url="https://example.com/cat.jpg"),
        ]
        msg = UserMessage(content=blocks)
        assert isinstance(msg.content, list)
        assert len(msg.content) == 2

    def test_mixed_blocks(self) -> None:
        blocks: list[ContentBlock] = [
            TextBlock(text="listen to this"),
            AudioBlock(data="audiodata", format="mp3"),
        ]
        msg = UserMessage(content=blocks)
        assert msg.content[1].type == "audio"  # type: ignore[index]


class TestAssistantMessageMultimodal:
    def test_default_is_empty_string(self) -> None:
        msg = AssistantMessage()
        assert msg.content == ""

    def test_string_content(self) -> None:
        msg = AssistantMessage(content="I see an image")
        assert msg.content == "I see an image"


class TestToolResultMultimodal:
    def test_string_content(self) -> None:
        tr = ToolResult(tool_call_id="id1", tool_name="search", content="result text")
        assert tr.content == "result text"

    def test_list_content(self) -> None:
        blocks: list[ContentBlock] = [
            ImageDataBlock(data="base64data", media_type="image/png"),
        ]
        tr = ToolResult(tool_call_id="id1", tool_name="dalle", content=blocks)
        assert isinstance(tr.content, list)
        assert tr.content[0].type == "image_data"  # type: ignore[index]

    def test_empty_default(self) -> None:
        tr = ToolResult(tool_call_id="x", tool_name="y")
        assert tr.content == ""


# ---------------------------------------------------------------------------
# _content_is_empty helper
# ---------------------------------------------------------------------------


class TestContentIsEmpty:
    def test_empty_string(self) -> None:
        assert _content_is_empty("") is True

    def test_non_empty_string(self) -> None:
        assert _content_is_empty("hello") is False

    def test_empty_list(self) -> None:
        assert _content_is_empty([]) is True

    def test_non_empty_list(self) -> None:
        blocks: list[ContentBlock] = [TextBlock(text="hi")]
        assert _content_is_empty(blocks) is False

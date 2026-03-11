"""Reporter — manages markdown reports with citation deduplication.

1:1 port of SkyworkAI's Report class from reporter.py.
Manages incremental report building with add_item() and complete().
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from orbiter.types import SystemMessage, UserMessage

from .llm_utils import call_llm

logger = logging.getLogger("deepagent")


# ---------------------------------------------------------------------------
# Pydantic models (1:1 from SkyworkAI)
# ---------------------------------------------------------------------------


class ContentItem(BaseModel):
    """Content item extracted from report input."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    content: str = Field(description="The content of the item")
    summary: str = Field(description="The summary of the item")
    reference_ids: list[int] = Field(description="The reference IDs of the item")


class ReferenceItem(BaseModel):
    """Reference item for citations."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    id: int = Field(description="The ID of the reference")
    description: str = Field(description="The brief description of the reference")
    url: str | None = Field(default=None, description="The URL of the reference")


class ReportItem(BaseModel):
    """A single report item with content and references."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    content: ContentItem = Field(description="The content of the item")
    references: list[ReferenceItem] = Field(description="The references of the item")


# ---------------------------------------------------------------------------
# Report class (1:1 port of SkyworkAI Report)
# ---------------------------------------------------------------------------


class Report(BaseModel):
    """Report builder with citation deduplication and markdown generation.

    Port of SkyworkAI's Report class. Supports incremental add_item() calls
    and a final complete() that merges, deduplicates, and generates markdown.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    title: str = Field(description="The title of the report")
    items: list[ReportItem] = Field(default_factory=list, description="Report items")
    model_name: str = Field(
        default="openai:gpt-4o-mini",
        description="Model for extraction and report generation",
    )
    report_file_path: str | None = Field(
        default=None,
        description="File path for saving the report",
    )

    def __init__(
        self,
        model_name: str | None = None,
        report_file_path: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if model_name is not None:
            self.model_name = model_name
        if report_file_path is not None:
            self.report_file_path = report_file_path

    async def add_item(
        self,
        file_path: str | None = None,
        content: str | dict[str, Any] | None = None,
    ) -> ReportItem:
        """Add a new item to the report by extracting ReportItem from content.

        1:1 port of SkyworkAI Report.add_item().

        Args:
            file_path: Optional path to a file whose content will be read.
            content: Input content as string or dictionary.

        Returns:
            The extracted and added ReportItem.
        """
        # Read file content if file_path is provided
        file_content = ""
        if file_path and os.path.exists(file_path):
            try:
                with open(file_path, encoding="utf-8") as f:
                    file_content = f.read()
            except Exception as e:
                file_content = f"[Note: Failed to read file {file_path}: {e}]"

        # Prepare input text
        if isinstance(content, dict):
            input_text = json.dumps(content, indent=4, ensure_ascii=False)
        else:
            input_text = str(content) if content else ""

        # Combine content and file content
        combined_content = input_text
        if file_content:
            if combined_content:
                combined_content = f"{combined_content}\n\n--- File Content from {file_path} ---\n\n{file_content}"
            else:
                combined_content = f"--- File Content from {file_path} ---\n\n{file_content}"

        # Build extraction prompt (1:1 from SkyworkAI)
        prompt = (
            "Extract and structure the following content into a report item with content, summary, and references.\n\n"
            f"Input Content:\n```json\n{combined_content}\n```\n\n"
            "Please extract:\n"
            "1. **Content**: The main content text (preserve the original content exactly, "
            "including all citations in markdown link format [1](url), [2](url), [3](url), etc.)\n"
            "2. **Summary**: A concise 2-3 sentence summary of the content\n"
            "3. **Reference IDs**: List of integer IDs that reference sources mentioned in the content\n"
            "4. **References**: List of reference items, each with:\n"
            "   - id: Integer ID matching the reference IDs found in the content\n"
            "   - description: Brief description of the reference source\n"
            "   - url: URL for the reference (extract from content or file_path)\n\n"
            "IMPORTANT REQUIREMENTS:\n"
            "- **Citation Format**: Citations MUST be in markdown link format: [1](url), [2](url), etc.\n"
            "- **Preserve Citations**: The content field MUST include all citation markers exactly as they appear\n"
            "- **Extract Reference IDs**: Parse all citation numbers from the content\n"
            "- **Match References**: Each reference_id must have a corresponding ReferenceItem\n\n"
            "Please Only Return the ReportItem object, no other text or explanation."
        )

        messages = [
            SystemMessage(
                content="You are an expert at extracting structured information from content. Extract content, summaries, and references accurately."
            ),
            UserMessage(content=prompt),
        ]

        response = await call_llm(
            model=self.model_name,
            messages=messages,
            response_format=ReportItem,
        )

        if response.parsed_model is not None:
            report_item = response.parsed_model
        else:
            # Fallback: create a basic ReportItem from the raw content
            report_item = ReportItem(
                content=ContentItem(
                    content=combined_content,
                    summary=combined_content[:200] + "..." if len(combined_content) > 200 else combined_content,
                    reference_ids=[],
                ),
                references=[],
            )

        self.items.append(report_item)
        return report_item

    async def complete(self) -> str:
        """Complete the report — merge, deduplicate, generate final markdown.

        1:1 port of SkyworkAI Report.complete().

        Returns:
            Final markdown report content.

        Raises:
            ValueError: If report has no items or no file path set.
        """
        if not self.items:
            raise ValueError("Cannot complete report: no items found")
        if not self.report_file_path:
            raise ValueError("Cannot complete report: report_file_path is not set")

        # Step 1: Collect and deduplicate references
        all_references_dict: dict[str, ReferenceItem] = {}
        reference_key_to_id: dict[str, int] = {}

        def normalize_reference_key(ref: ReferenceItem) -> str:
            desc = ref.description.strip().lower() if ref.description else ""
            url = ref.url.strip().lower() if ref.url else ""
            if url:
                return f"url:{url.rstrip('/')}"
            if desc.startswith(("http://", "https://", "file://")):
                return f"url:{desc.rstrip('/')}"
            return f"desc:{desc}"

        for item in self.items:
            for ref in item.references:
                key = normalize_reference_key(ref)
                if key in all_references_dict:
                    existing = all_references_dict[key]
                    if ref.url and not existing.url:
                        existing.url = ref.url
                    if ref.description and len(ref.description) > len(existing.description):
                        existing.description = ref.description
                else:
                    all_references_dict[key] = ref
                    reference_key_to_id[key] = ref.id

        # Step 2: Create reference mapping (old_id -> new_id)
        unique_references = list(all_references_dict.values())
        reference_mapping: dict[int, int] = {}

        for new_id, (key, _ref) in enumerate(all_references_dict.items(), start=1):
            for item in self.items:
                for old_ref in item.references:
                    if normalize_reference_key(old_ref) == key:
                        reference_mapping[old_ref.id] = new_id

        # Step 3: Build URL mapping
        reference_urls: dict[int, str] = {}
        for new_id, ref in enumerate(unique_references, start=1):
            description = ref.description
            if ref.url:
                reference_urls[new_id] = ref.url
            elif description.startswith(("http://", "https://")):
                reference_urls[new_id] = description
            elif os.path.exists(description) or "/" in description or "\\" in description:
                abs_path = os.path.abspath(description) if not os.path.isabs(description) else description
                reference_urls[new_id] = f"file://{abs_path}"
            else:
                url_match = re.search(r"(https?://[^\s]+)", description)
                reference_urls[new_id] = url_match.group(1) if url_match else description

        # Step 4: Update citations in content
        updated_contents = []
        for item in self.items:
            content = item.content.content
            reference_ids = item.content.reference_ids

            def replace_citation(match: re.Match[str]) -> str:
                old_id_str = match.group(1)
                try:
                    old_id = int(old_id_str)
                    new_id = reference_mapping.get(old_id)
                    if new_id is not None:
                        url = reference_urls.get(new_id, f"#ref{new_id}")
                        return f"[{new_id}]({url})"
                    return match.group(0)
                except ValueError:
                    return match.group(0)

            updated_content = re.sub(r"\[(\d+)\]?(?:\([^)]+\))?", replace_citation, content)
            updated_reference_ids = sorted(set(reference_mapping.get(rid, rid) for rid in reference_ids))

            updated_contents.append({
                "content": updated_content,
                "summary": item.content.summary,
                "reference_ids": updated_reference_ids,
            })

        # Step 5: Build renumbered references
        renumbered_references = []
        for new_id, ref in enumerate(unique_references, start=1):
            url = ref.url if ref.url else reference_urls.get(new_id, ref.description)
            renumbered_references.append({
                "id": new_id,
                "description": ref.description,
                "url": url,
            })

        # Step 6: Build LLM prompt for final report
        items_text = "\n\n".join([
            f"## Item {i + 1}\n\n**Summary:** {item['summary']}\n\n**Content:**\n{item['content']}\n\n**Reference IDs:** {item['reference_ids']}"
            for i, item in enumerate(updated_contents)
        ])

        references_text = "\n".join([
            f"[{ref['id']}]({ref['url']}) {ref['description']}"
            for ref in renumbered_references
        ])
        if references_text:
            references_text = "\n" + references_text.replace("\n", "\n\n") + "\n"
            references_text = "\n" + references_text.replace("\n", "\n\n") + "\n"

        prompt = (
            "Generate a complete, well-structured markdown report based on the following report items and references.\n\n"
            f"Report Title: {self.title}\n\n"
            f"Report Items:\n{items_text}\n\n"
            f"References:\n{references_text}\n\n"
            f"Please generate a comprehensive markdown report that:\n"
            f"1. **Starts with the title** as a main heading (# {self.title})\n"
            f"2. **Organizes content logically** - Group related items into sections\n"
            f"3. **Preserves all citations** - Keep [number](url) format exactly\n"
            f"4. **Integrates summaries** - Use item summaries for smooth transitions\n"
            f"5. **Includes References section** at the end\n\n"
            f"IMPORTANT: Preserve all citations and facts. Use proper markdown formatting.\n\n"
            f"Return ONLY the complete markdown report content."
        )

        messages = [
            SystemMessage(
                content="You are an expert report writer specializing in creating comprehensive, well-structured reports with proper citations and references."
            ),
            UserMessage(content=prompt),
        ]

        # Step 7: Generate report via LLM
        response = await call_llm(model=self.model_name, messages=messages)
        if not response.success:
            raise ValueError(f"Failed to generate report: {response.message}")

        report_content = response.message.strip()

        # Step 8: Ensure References section exists
        if "## References" not in report_content and "References" not in report_content:
            report_content += f"\n\n## References\n\n{references_text}\n"
        else:
            report_content = re.sub(
                r"## References.*?(?=\n##|\Z)",
                f"## References\n\n{references_text}\n",
                report_content,
                flags=re.DOTALL,
            )

        # Step 9: Ensure all citations have URLs
        def add_url_to_citation(match: re.Match[str]) -> str:
            citation_num = match.group(1)
            if match.group(0).count("(") == 0:
                citation_id = int(citation_num)
                url = reference_urls.get(citation_id, f"#ref{citation_id}")
                return f"[{citation_num}]({url})"
            return match.group(0)

        report_content = re.sub(r"\[(\d+)\](?!\()", add_url_to_citation, report_content)

        # Step 10: Write to file
        os.makedirs(os.path.dirname(self.report_file_path), exist_ok=True)
        with open(self.report_file_path, "w", encoding="utf-8") as f:
            f.write(report_content)

        return report_content

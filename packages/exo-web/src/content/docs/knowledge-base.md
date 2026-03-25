---
title: Knowledge Base
description: Document ingestion and RAG retrieval
section: guides
order: 4
---

# Knowledge Base

The Knowledge Base gives agents access to your documents through Retrieval-Augmented Generation (RAG). Upload documents, and agents can search and cite them during conversations.

## Creating a Knowledge Base

1. Navigate to **Knowledge Base** in the sidebar
2. Click **New Knowledge Base**
3. Name it and configure the embedding model
4. Upload documents (PDF, TXT, Markdown, CSV)

## Supported Formats

| Format | Description |
|--------|-------------|
| PDF | Automatically extracted and chunked |
| TXT | Plain text, split by paragraphs |
| Markdown | Parsed with heading-based chunking |
| CSV | Row-based ingestion with header mapping |

## Connecting to Agents

1. Open your agent's edit page
2. In the **Knowledge** section, select one or more knowledge bases
3. The agent will automatically search relevant documents during conversations

## Chunk Settings

Fine-tune how documents are split:

- **Chunk size** — Target size in tokens (default: 512)
- **Chunk overlap** — Overlap between chunks for context continuity
- **Separator** — Custom text separator for splitting

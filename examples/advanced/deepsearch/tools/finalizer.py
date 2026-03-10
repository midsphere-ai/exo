"""Answer finalization and reference building."""
from __future__ import annotations

import logging
import re
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from ..types import KnowledgeItem, Reference, WebContent

from .embeddings import cosine_similarity, jaccard_rank, EmbeddingProvider
from ..utils.text_tools import get_knowledge_str, chunk_text
from ..utils.url_tools import normalize_hostname

logger = logging.getLogger("deepsearch")


# ---------------------------------------------------------------------------
# finalize_answer
# ---------------------------------------------------------------------------


async def finalize_answer(
    md_content: str,
    knowledge: list["KnowledgeItem"],
    generate_text_fn,  # async def(system, prompt) -> str
    schema_gen,
) -> str:
    """Polish the final answer with accumulated knowledge.

    Uses an LLM call to revise the draft answer produced during research,
    applying editorial rules for structure, language, and formatting.

    If the polished result is less than 85 % the length of the original the
    original is returned instead (guard against over-condensation).
    """
    try:
        knowledge_str = get_knowledge_str(knowledge)
        language_style = getattr(schema_gen, "language_style", "formal English")

        system_prompt = (
            "You are a senior editor with multiple best-selling books and columns "
            "published in top magazines. You break conventional thinking, establish "
            "unique cross-disciplinary connections, and bring new perspectives to the "
            "user.\n"
            "\n"
            "Your task is to revise the provided markdown content (written by your "
            "junior intern) while preserving its original vibe, delivering a polished "
            "and professional version.\n"
            "\n"
            "<structure>\n"
            "- Begin with fact-driven statement of the main question or issue you'll "
            "address\n"
            "- Develop your argument using a logical progression of ideas while allowing "
            "for occasional contemplative digressions that enrich the reader's "
            "understanding\n"
            "- Organize paragraphs with clear topic sentences but vary paragraph length "
            "to create rhythm and emphasis, do not use bullet points or numbered lists.\n"
            "- Write section headers as single phrases without colons (##, ###) to "
            "organize long content. Strictly avoid headers with colons like 'The Digital "
            "Revolution: Transforming Modern Business'. Instead use 'Digital "
            "Transformation in Business'.\n"
            "- Present facts, quotes and data points with minimal hedging\n"
            "- Conclude with both a definitive statement of your position and a "
            "thought-provoking reflection that leaves readers pondering deeper "
            "implications.\n"
            "- Remove all disclaimer and copyright notices at the end of the content.\n"
            "</structure>\n"
            "\n"
            "\n"
            "<language-style>\n"
            "- Balance factual precision with vivid, evocative descriptions\n"
            "- Incorporate humor and wit to illuminate serious intellectual points\n"
            "- Use accessible language that combines clarity with elegant expressiveness\n"
            "- Include relevant cultural references from both Eastern and Western "
            "traditions while maintaining analytical rigor\n"
            "- Brutal intellectual honesty that doesn't pander to feelings\n"
            "</language-style>\n"
            "\n"
            "<content-approach>\n"
            "- Address a contemporary issue through both rational analysis and humanistic "
            "reflection\n"
            "- Support arguments with empirical evidence alongside illustrative anecdotes "
            "and metaphors\n"
            "- Consider practical implications while exploring philosophical dimensions\n"
            "- Maintain intellectual honesty and critical thinking with an appreciation "
            "for life's paradoxes and ironies\n"
            "- Dismiss regulatory barriers and status quo defenders as obstacles to "
            "progress\n"
            "- Present technological development as a moral imperative\n"
            "</content-approach>\n"
            "\n"
            "<rules>\n"
            "1. Avoid any bullet points or numbered lists, use natural language instead.\n"
            "2. Extend the content with 5W1H strategy and add more details to make it "
            "more informative and engaging. Use available knowledge to ground facts and "
            "fill in missing information.\n"
            "3. Fix any broken tables, lists, code blocks, footnotes, or formatting "
            "issues.\n"
            "4. Tables are good! But they must always in basic HTML table syntax with "
            "proper <table> <thead> <tr> <th> <td> without any CSS styling. STRICTLY "
            "AVOID any markdown table syntax. HTML Table should NEVER BE fenced with "
            "(```html) triple backticks.\n"
            "5. Replace any obvious placeholders or Lorem Ipsum values such as "
            '"example.com" with the actual content derived from the knowledge.\n'
            "6. Latex are good! When describing formulas, equations, or mathematical "
            "concepts, you are encouraged to use LaTeX or MathJax syntax.\n"
            "7. Your output language must be the same as user input language.\n"
            "</rules>\n"
            "\n"
            "\n"
            "The following knowledge items are provided for your reference. Note that "
            "some of them may not be directly related to the content user provided, but "
            "may give some subtle hints and insights:\n"
            + "\n\n".join(knowledge_str)
            + "\n\n"
            'IMPORTANT: Do not begin your response with phrases like "Sure", "Here is", '
            '"Below is", or any other introduction. Directly output your revised content '
            f"in {language_style} that is ready to be published. Preserving HTML tables "
            "if exist, never use tripple backticks html to wrap html table."
        )

        result = await generate_text_fn(system_prompt, md_content)

        logger.debug(
            "finalized answer before/after: %d -> %d", len(md_content), len(result)
        )

        if len(result) < len(md_content) * 0.85:
            logger.warning(
                "finalized answer length %d is significantly shorter than original "
                "content %d, returning original content instead.",
                len(result),
                len(md_content),
            )
            return md_content

        return result

    except Exception:
        logger.exception("finalizer error")
        return md_content


# ---------------------------------------------------------------------------
# build_references
# ---------------------------------------------------------------------------


async def build_references(
    answer: str,
    web_contents: dict[str, "WebContent"],
    embedding_provider: EmbeddingProvider | None,
    min_chunk_length: int = 80,
    max_ref: int = 10,
    min_rel_score: float = 0.7,
    only_hostnames: list[str] | None = None,
) -> tuple[str, list[dict]]:
    """Build citation references using semantic matching.

    Returns ``(modified_answer_with_markers, list_of_reference_dicts)`` where
    each reference dict has keys ``exact_quote``, ``url``, ``title``,
    ``relevance_score``, ``answer_chunk``, and ``answer_chunk_position``.
    """
    logger.debug(
        "[buildReferences] Starting with max_ref=%d, min_chunk_length=%d, "
        "min_rel_score=%.2f",
        max_ref,
        min_chunk_length,
        min_rel_score,
    )
    logger.debug(
        "[buildReferences] Answer length: %d chars, Web content sources: %d",
        len(answer),
        len(web_contents),
    )

    # ------------------------------------------------------------------
    # Step 1: Chunk the answer
    # ------------------------------------------------------------------
    logger.debug("[buildReferences] Step 1: Chunking answer text")
    chunk_result = chunk_text(answer)
    answer_chunks: list[str] = chunk_result["chunks"]
    answer_chunk_positions: list[list[int]] = chunk_result["chunk_positions"]
    logger.debug(
        "[buildReferences] Answer segmented into %d chunks", len(answer_chunks)
    )

    # ------------------------------------------------------------------
    # Step 2: Prepare all web content chunks, filter by min length
    # ------------------------------------------------------------------
    logger.debug(
        "[buildReferences] Step 2: Preparing web content chunks and filtering by "
        "minimum length (%d chars)",
        min_chunk_length,
    )
    all_web_content_chunks: list[str] = []
    chunk_to_source_map: dict[int, dict[str, Any]] = {}
    valid_web_chunk_indices: set[int] = set()

    chunk_index = 0
    for url, content in web_contents.items():
        if not content.chunks:
            continue
        if only_hostnames and normalize_hostname(url) not in only_hostnames:
            continue

        for i, chunk in enumerate(content.chunks):
            all_web_content_chunks.append(chunk)
            chunk_to_source_map[chunk_index] = {
                "url": url,
                "title": content.title or url,
                "text": chunk,
            }
            if chunk and len(chunk) >= min_chunk_length:
                valid_web_chunk_indices.add(chunk_index)
            chunk_index += 1

    logger.debug(
        "[buildReferences] Collected %d web chunks, %d above minimum length",
        len(all_web_content_chunks),
        len(valid_web_chunk_indices),
    )

    if not all_web_content_chunks:
        logger.debug(
            "[buildReferences] No web content chunks available, returning without "
            "references"
        )
        return answer, []

    # ------------------------------------------------------------------
    # Step 3: Filter answer chunks by minimum length
    # ------------------------------------------------------------------
    logger.debug("[buildReferences] Step 3: Filtering answer chunks by minimum length")
    valid_answer_chunks: list[str] = []
    valid_answer_chunk_indices: list[int] = []
    valid_answer_chunk_positions: list[list[int]] = []

    for i, ac in enumerate(answer_chunks):
        if not ac.strip() or len(ac) < min_chunk_length:
            continue
        valid_answer_chunks.append(ac)
        valid_answer_chunk_indices.append(i)
        valid_answer_chunk_positions.append(answer_chunk_positions[i])

    logger.debug(
        "[buildReferences] Found %d/%d valid answer chunks above minimum length",
        len(valid_answer_chunks),
        len(answer_chunks),
    )

    if not valid_answer_chunks:
        logger.debug(
            "[buildReferences] No valid answer chunks, returning without references"
        )
        return answer, []

    # ------------------------------------------------------------------
    # Step 4: Get embeddings for all chunks in a single request
    # ------------------------------------------------------------------
    logger.debug(
        "[buildReferences] Step 4: Getting embeddings for all chunks in a single "
        "request (only including web chunks above min length)"
    )

    # Build a combined list and a mapping to recover which embedding
    # belongs to which original collection (answer vs web).
    chunk_index_map: dict[int, dict[str, Any]] = {}
    all_chunks: list[str] = []

    # Add answer chunks first
    for idx, chunk in enumerate(valid_answer_chunks):
        all_chunks.append(chunk)
        chunk_index_map[len(all_chunks) - 1] = {
            "type": "answer",
            "original_index": idx,
        }

    # Then add web chunks that meet minimum length requirement
    for i in range(len(all_web_content_chunks)):
        if i in valid_web_chunk_indices:
            all_chunks.append(all_web_content_chunks[i])
            chunk_index_map[len(all_chunks) - 1] = {
                "type": "web",
                "original_index": i,
            }

    logger.debug(
        "[buildReferences] Requesting embeddings for %d total chunks "
        "(%d answer + %d web)",
        len(all_chunks),
        len(valid_answer_chunks),
        len(valid_web_chunk_indices),
    )

    try:
        if embedding_provider is None:
            raise RuntimeError("No embedding provider configured")

        # Get embeddings for all chunks in one request
        all_embeddings = await embedding_provider.embed(all_chunks)

        if not all_embeddings:
            raise RuntimeError("Empty embedding result")

        # Separate embeddings back into answer and web collections
        answer_embeddings: dict[int, list[float]] = {}
        web_embedding_map: dict[int, list[float]] = {}

        for i, embedding in enumerate(all_embeddings):
            mapping = chunk_index_map.get(i)
            if mapping is None:
                continue
            if mapping["type"] == "answer":
                answer_embeddings[mapping["original_index"]] = embedding
            else:
                web_embedding_map[mapping["original_index"]] = embedding

        logger.debug(
            "[buildReferences] Successfully generated and separated embeddings: "
            "%d answer, %d web",
            len(answer_embeddings),
            len(web_embedding_map),
        )

        # --------------------------------------------------------------
        # Step 5: Compute pairwise cosine similarity
        # --------------------------------------------------------------
        logger.debug(
            "[buildReferences] Step 5: Computing pairwise cosine similarity between "
            "answer and web chunks"
        )
        all_matches: list[dict[str, Any]] = []

        for i in range(len(valid_answer_chunks)):
            answer_chunk_index = valid_answer_chunk_indices[i]
            answer_chunk = valid_answer_chunks[i]
            answer_chunk_position = valid_answer_chunk_positions[i]
            answer_embedding = answer_embeddings.get(i)

            if answer_embedding is None:
                continue

            matches_for_chunk: list[dict[str, Any]] = []

            for web_chunk_index in valid_web_chunk_indices:
                web_embedding = web_embedding_map.get(web_chunk_index)
                if web_embedding is not None:
                    score = cosine_similarity(answer_embedding, web_embedding)
                    matches_for_chunk.append(
                        {
                            "web_chunk_index": web_chunk_index,
                            "relevance_score": score,
                        }
                    )

            matches_for_chunk.sort(
                key=lambda m: m["relevance_score"], reverse=True
            )

            for match in matches_for_chunk:
                all_matches.append(
                    {
                        "web_chunk_index": match["web_chunk_index"],
                        "answer_chunk_index": answer_chunk_index,
                        "relevance_score": match["relevance_score"],
                        "answer_chunk": answer_chunk,
                        "answer_chunk_position": answer_chunk_position,
                    }
                )

            if matches_for_chunk:
                logger.debug(
                    "[buildReferences] Processed answer chunk %d/%d, top score: %.4f",
                    i + 1,
                    len(valid_answer_chunks),
                    matches_for_chunk[0]["relevance_score"],
                )

        # Log relevance-score statistics
        if all_matches:
            scores = [m["relevance_score"] for m in all_matches]
            logger.debug(
                "Reference relevance statistics: min=%.4f max=%.4f mean=%.4f count=%d",
                min(scores),
                max(scores),
                sum(scores) / len(scores),
                len(scores),
            )

        # --------------------------------------------------------------
        # Step 6: Sort all matches by relevance
        # --------------------------------------------------------------
        all_matches.sort(key=lambda m: m["relevance_score"], reverse=True)
        logger.debug(
            "[buildReferences] Step 6: Sorted %d potential matches by relevance score",
            len(all_matches),
        )

        # --------------------------------------------------------------
        # Step 7: Filter for uniqueness and threshold
        # --------------------------------------------------------------
        logger.debug(
            "[buildReferences] Step 7: Filtering matches to ensure uniqueness and "
            "threshold (min: %.2f)",
            min_rel_score,
        )
        used_web_chunks: set[int] = set()
        used_answer_chunks: set[int] = set()
        filtered_matches: list[dict[str, Any]] = []

        for match in all_matches:
            if match["relevance_score"] < min_rel_score:
                continue
            wci = match["web_chunk_index"]
            aci = match["answer_chunk_index"]
            if wci not in used_web_chunks and aci not in used_answer_chunks:
                filtered_matches.append(match)
                used_web_chunks.add(wci)
                used_answer_chunks.add(aci)
                if len(filtered_matches) >= max_ref:
                    break

        logger.debug(
            "[buildReferences] Selected %d/%d references after filtering",
            len(filtered_matches),
            len(all_matches),
        )
        return _build_final_result(answer, filtered_matches, chunk_to_source_map)

    except Exception as exc:
        logger.error("Embedding failed, falling back to Jaccard similarity: %s", exc)

        # ------------------------------------------------------------------
        # Jaccard fallback
        # ------------------------------------------------------------------
        all_matches_fb: list[dict[str, Any]] = []

        for i in range(len(valid_answer_chunks)):
            answer_chunk = valid_answer_chunks[i]
            answer_chunk_index = valid_answer_chunk_indices[i]
            answer_chunk_position = valid_answer_chunk_positions[i]

            logger.debug(
                "[buildReferences] Processing answer chunk %d/%d with Jaccard "
                "similarity",
                i + 1,
                len(valid_answer_chunks),
            )

            fallback_results = jaccard_rank(answer_chunk, all_web_content_chunks)

            for match in fallback_results:
                if match["index"] in valid_web_chunk_indices:
                    all_matches_fb.append(
                        {
                            "web_chunk_index": match["index"],
                            "answer_chunk_index": answer_chunk_index,
                            "relevance_score": match["relevance_score"],
                            "answer_chunk": answer_chunk,
                            "answer_chunk_position": answer_chunk_position,
                        }
                    )

        all_matches_fb.sort(key=lambda m: m["relevance_score"], reverse=True)
        logger.debug(
            "[buildReferences] Fallback complete. Found %d potential matches",
            len(all_matches_fb),
        )

        used_web_chunks_fb: set[int] = set()
        used_answer_chunks_fb: set[int] = set()
        filtered_matches_fb: list[dict[str, Any]] = []

        for match in all_matches_fb:
            if match["relevance_score"] < min_rel_score:
                continue
            wci = match["web_chunk_index"]
            aci = match["answer_chunk_index"]
            if wci not in used_web_chunks_fb and aci not in used_answer_chunks_fb:
                filtered_matches_fb.append(match)
                used_web_chunks_fb.add(wci)
                used_answer_chunks_fb.add(aci)
                if len(filtered_matches_fb) >= max_ref:
                    break

        logger.debug(
            "[buildReferences] Selected %d references using fallback method",
            len(filtered_matches_fb),
        )
        return _build_final_result(answer, filtered_matches_fb, chunk_to_source_map)


# ---------------------------------------------------------------------------
# _build_final_result  (helper)
# ---------------------------------------------------------------------------


def _build_final_result(
    answer: str,
    filtered_matches: list[dict[str, Any]],
    chunk_to_source_map: dict[int, dict[str, Any]],
) -> tuple[str, list[dict]]:
    """Build reference dicts and inject ``[^N]`` markers into the answer."""
    logger.debug(
        "[buildFinalResult] Building final result with %d references",
        len(filtered_matches),
    )

    # Build reference objects
    references: list[dict[str, Any]] = []
    for match in filtered_matches:
        source = chunk_to_source_map[match["web_chunk_index"]]
        if not source.get("text") or not source.get("url") or not source.get("title"):
            continue
        references.append(
            {
                "exact_quote": source["text"],
                "url": source["url"],
                "title": source["title"],
                "date_time": source.get("date_time"),
                "relevance_score": match["relevance_score"],
                "answer_chunk": match["answer_chunk"],
                "answer_chunk_position": match["answer_chunk_position"],
            }
        )

    # Sort references by position in the answer (to insert markers in order)
    references_by_position = sorted(
        references, key=lambda r: r["answer_chunk_position"][0]
    )

    logger.debug("[buildFinalResult] Injecting reference markers into answer")

    # Insert markers from beginning to end, tracking offset
    modified_answer = answer
    offset = 0

    for i, ref in enumerate(references_by_position):
        marker = f"[^{i + 1}]"
        insert_position = ref["answer_chunk_position"][1] + offset

        # Look ahead to check if there's a list item coming next
        text_after_insert = modified_answer[insert_position:]
        next_list_item_match = re.match(r"^\s*\n\s*\*\s+", text_after_insert)

        if next_list_item_match:
            # Move the marker to right after the last content character,
            # but INSIDE any punctuation at the end of the content
            before_text = modified_answer[
                max(0, insert_position - 30) : insert_position
            ]
            last_punctuation = re.search(r"[！。？!.?]$", before_text)
            if last_punctuation:
                insert_position -= 1
        else:
            # Handle newlines and table pipes at end of chunk
            chunk_end_text = modified_answer[
                max(0, insert_position - 5) : insert_position
            ]
            newline_match = re.search(r"\n+$", chunk_end_text)
            table_end_match = re.search(r"\s*\|\s*$", chunk_end_text)

            if newline_match:
                insert_position -= len(newline_match.group(0))
            elif table_end_match:
                insert_position -= len(table_end_match.group(0))

        # Insert the marker
        modified_answer = (
            modified_answer[:insert_position]
            + marker
            + modified_answer[insert_position:]
        )

        # Update offset for subsequent insertions
        offset += len(marker)

    logger.debug(
        "[buildFinalResult] Complete. Generated %d references", len(references)
    )
    return modified_answer, references

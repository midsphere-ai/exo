# Benchmark Results

**Query:** "How does SpaceX's Starship compare to NASA's SLS in terms of cost per launch, payload capacity, and development timeline, and which one is more likely to be used for the Artemis program going forward?"

**Date:** 2026-03-12
**Model:** gemini:gemini-3.1-pro-preview
**Fast Model:** gemini:gemini-3-flash-preview

## Timing

| Mode | Time | Sources Found | Sources Cited |
|------|------|--------------|--------------|
| Speed | 22.7s | 10 | 8 |
| Balanced | 77.9s | 30 | 14 |
| Quality | 164.6s | 26 | 22 |

## Pipeline Stages

### Speed (~23s)
- Classifier + direct SearXNG search (parallel)
- Writer (fast model, no enrichment)

### Balanced (~78s)
- Classifier
- Hybrid research (2 adaptive rounds + parallel researchers, 15s timeout)
- Rerank by embeddings
- Enrich top 5 via Jina Reader
- Writer

### Quality (~165s)
- Classifier
- Hybrid research (4 adaptive rounds + parallel researchers, 60s timeout)
- Rerank by embeddings
- Enrich top 10 via Jina Reader
- Writer

## Answer Quality Assessment

### Speed
- Covered all parts of the question
- Good structure with cost, payload, and Artemis roles
- Missing specific dollar figures for Starship cost per launch
- No mention of Feb 2026 Artemis overhaul (data was in snippets only)

### Balanced
- Specific cost data: SLS $4B/launch, Starship ~$100M
- Feb 2026 Artemis overhaul covered (Block 1B cancellation)
- Starship 50% performance shortfall mentioned
- 2028 delay for Artemis III included
- Good depth for the time investment

### Quality
- Most comprehensive: detailed cost-per-kg analysis ($43K/kg SLS vs $50/kg Starship target)
- Full development timeline coverage (SLS since 2011, Starship Flight 10 Aug 2025, V3 April 2026)
- Deep space payload capabilities (Jupiter, Neptune, interstellar)
- Feb 2026 Artemis overhaul with Block 1B cancellation
- Forward-looking analysis on Starship V3 and industry trends
- Summary section with clean takeaways

## Output Files
- `speed.txt` — Full speed mode output
- `balanced.txt` — Full balanced mode output
- `quality.txt` — Full quality mode output

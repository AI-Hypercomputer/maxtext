# Verifier-Craft Notebook

## 2026-06-11: run-20260611-114500 (tmem deep-dive, HTML)

**Verdict**: PASS

**Patterns observed**:
- "Key insight" used twice in the same document (Section 2 callout, Section 6.3 opening). Mild AI tell, not severe in technical writing but worth watching for in future drafts.
- Dangling arithmetic in bullet lists: author caught a numerical error mid-sentence and patched with "...actually" instead of rewriting. This is a drafting artifact, not an AI tell per se, but it reads sloppy.
- Contradictory numbers in two places (Section 9.2 vs. Section 12 table): the text claims one optimization accounts for 7-9 of 9.5 GB total savings, but the table distributes savings across six optimizations summing to more than 9.5. Flagged as a structural/coherence issue (facts verifier should validate the numbers themselves).
- No conclusion/closing section -- piece ends on an appendix. For a deep-dive intended as talk material, this leaves the reader stranded.
- Otherwise very clean: no voice drift, no marketing register, no classic AI prose patterns, good sentence variation in most sections.

## 2026-06-11: run-20260611-114500 v2 (tmem deep-dive, HTML)

**Verdict**: PASS

**What changed from v1**: The contradictory-numbers problem was fixed by replacing per-optimization numerical estimates with qualitative labels ("largest single contributor", "moderate", "small") and adding a clear explanation of why standalone impacts are non-additive. This is a valid fix but trades specificity for consistency. The "key insight" duplication persists (Section 2 callout + Section 6.3 opening).

**Recurring patterns to watch**:
- "Key insight" as a callout opener: twice in a single doc is a pattern. Flag it early in future drafts.
- Pieces that end on appendices without a closing summary: this is the second time. For talk-material-intended deliverables, always flag when there is no landing paragraph before the appendices.
- When numbers are contradictory, the writer's instinct is to remove numbers rather than fix them. Result is defensible but less useful. Better to push for corrected numbers when approximate measurements exist.
- Code snippets with mismatched signatures (function def vs. body referencing undeclared variables): caught `stream_buf_idx` in `_update_state_io`. Formally a facts/accuracy issue, but it disrupts reading flow, so craft-verifier should note it as a reading-experience issue.

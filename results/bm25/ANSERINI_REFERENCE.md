# Anserini reference for the BM25 comparison

Strata's BM25 results in this directory are compared against Anserini's published BEIR
regression for SciFact. Machine-readable copy (used by `bench/validate_bm25_beir.py`):
[`anserini_reference.json`](anserini_reference.json).

| Field | Value |
|---|---|
| System | Anserini **2.3.0** (release tag `anserini-2.3.0`, commit `578143efbb73d842748e6a3c379f40f459d03895`), Lucene 10.5.0 |
| Configuration | **`beir-v1.0.0-scifact.flat`** (flat, *not* multifield): `title` + `"\n"` + `text` in one `contents` field |
| Analyzer | `DefaultEnglishAnalyzer`: StandardTokenizer → EnglishPossessiveFilter → LowerCaseFilter → StopFilter (Lucene English, 33 words) → PorterStemFilter |
| Retrieval | `-bm25` (k1 = 0.9, b = 0.4), `-removeQuery`, `-hits 1000` |
| Evaluation | `trec_eval -c`, `ndcg_cut.10`, `recall.100`, `recall.1000`, qrels `beir-v1.0.0-scifact.test` (300 queries) |
| Published | nDCG@10 **0.6789**, R@100 **0.9253**, R@1000 **0.9767** |
| Index stats | 5,183 documents, 838,128 total terms |

Sources, pinned to the release:
- Published numbers: <https://github.com/castorini/anserini/blob/anserini-2.3.0/docs/reproduce/from-document-collection/beir-v1.0.0-scifact.flat.md>
- Regression config (commands, index stats): <https://github.com/castorini/anserini/blob/anserini-2.3.0/src/main/resources/reproduce/from-document-collection/configs/beir-v1.0.0-scifact.flat.yaml>

The same values appear on `master` at `6caeb33a27385d6acc726d81874e34d3a567f46f` (2026-08-16), the
last commit that touched the config. Retrieved 2026-09-25.

Anserini's "multifield" SciFact configuration (`beir-v1.0.0-scifact.multifield`) indexes title and
text as separate fields and publishes different numbers; it is not the comparison used here.

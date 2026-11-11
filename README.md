# kb-retrieval-benchmark

**Which retrieval method and which model should a personal markdown knowledge base use at 300, 900 and 3,000 documents?**

3 models × 11 retrieval setups × 3 knowledge-base sizes — 31,713 answers, all graded.

- Report: [`report.en.html`](report.en.html) (English) · [`report.html`](report.html) (中文) — open locally in a browser
- Method, data sources and question construction: appendix of the report; full working notes in [`docs/PLAN.md`](docs/PLAN.md) (Chinese)

## Results (3,000 documents, clean MuSiQue set, after the 2026-11-11 question audit)

| Setup | Accuracy | Says "not in the KB" when it isn't | p50 latency | Cost / question |
|---|--:|--:|--:|--:|
| **DeepSeek V4.1 Flash + agent with grep** | **88%** | 90% | 61 s | $0.0043 |
| Claude Sonnet 5.5 + agent with grep | 84% | 93% | 21 s | $0.031 |
| Claude Haiku 5.5 + agent with grep | 67% | 97% | 42 s | $0.0032 |
| DeepSeek V4.1 Flash, single-shot hybrid retrieval | 56% | 95% | 8 s | $0.0012 |

Exact paired McNemar tests on the same 115 questions, Holm-adjusted over 21 comparisons (table in the report
appendix): **DeepSeek vs Sonnet with grep is not significant** (88% vs 84%, p = 0.39; the earlier p = 0.015 rested
largely on mislabeled questions). DeepSeek vs Haiku, and agent + grep vs single-shot retrieval, remain significant.
At 300 documents nothing is significant (38 paired questions, accuracies near the ceiling).

**Question audit (2026-11-11).** A deliberately harsh review of the benchmark found that MuSiQue gold labels had
never been checked and that the leak check for unanswerable questions could not see multi-hop leaks. Fixed:
a blind gold audit by Claude Opus 5.5 (judged against the KB, never shown model answers) removed 56 ill-posed or
mislabeled questions and added aliases to 28 (their answers re-graded); a second leak check (retrieval by the
removed last hop, untruncated BRIGHT pages, hand review) removed more unanswerable questions per KB size; runs
without a response now count as failures; significance is computed in code (`harness/stats.py`) and tested
(`tests/`). Not fixed, because it would need reruns: the 300 / 900 KBs contain almost none of the questions' own
confusable pages (gold documents are 72% / 65% / 20% of the KB), 15 of 20 NoLiMa questions have several valid
answers, the DeepSeek / Haiku closed-book filter was added after the fact, and reasoning effort differs (Claude
medium, DeepSeek high). Before / after numbers: [`report/audit_impact.md`](report/audit_impact.md).

What the numbers say:

1. **Single-shot retrieval is bounded by the retriever** (BM25 → hybrid → QMD-style enhanced: 25–30% → 43–56% → 64–72% accuracy across the three models).
2. **With an agent, Haiku trails by ~20 points; DeepSeek and Sonnet are within noise of each other.** For the same model, the retrieval backend moves accuracy by about 8–16 points.
3. **More recall ≠ more accuracy**: semantic search tools surface more evidence than grep, but agents answer no better with them (Sonnet: grep 84% vs hybrid 72%, significant; DeepSeek: 88% vs 83%, not significant). At ≤ 3,000 documents, grep + read is enough.
4. **"Full evidence found" is not a measure of search effort**: it counts grep hit lines and grows with the number of calls. DeepSeek makes ~3× Sonnet's tool calls and 27% of its agent runs hit the 12-step cap (Claude: 1%), so results are sensitive to the cap.
5. **Haiku refuses more**: with agent + grep it says "not in the KB" on 27% of answerable questions (DeepSeek 12%, Sonnet 19%).
6. An LLM-wiki digest (summary + entity pages) does not lower accuracy and helps hybrid search most for Haiku, but the Haiku digest drops ~23% of answer facts.

## Data

The knowledge base mixes three public datasets (nested 300 ⊂ 900 ⊂ 3000 documents; ~55% Wikipedia, ~35% web pages, ~10% fiction):

| Dataset | Used for | License |
|---|---|---|
| [MuSiQue](https://github.com/StonyBrookNLP/musique) (dev, full) — Trivedi et al., TACL 2022 | Wikipedia pages; 2–4-hop questions; unanswerable variants built by removing one evidence page | CC BY 4.0 |
| [BRIGHT](https://huggingface.co/datasets/xlangai/BRIGHT) — Su et al., 2024 | 7 StackExchange domains: real questions and the long web pages they cite | CC BY 4.0 |
| [NoLiMa](https://huggingface.co/datasets/amodaresi/NoLiMa) — Modarressi et al., ICML 2025 | Fiction chunks with inserted "needle" sentences (latent-association retrieval) | Adobe Research License |

**What is in this repository:** derived question sets and graded results for MuSiQue and BRIGHT (`questions_v2/`, `data/results/p2_judged.tar.gz`), and the Haiku LLM-wiki digests of their documents (`data/digest/kb2-haiku/`). **NoLiMa-derived content is not redistributed** (20 questions, fiction chunks and their digests) — rebuild it from the original dataset with the scripts below. The reports contain aggregate metrics only.

## Reproduce

Requires [uv](https://docs.astral.sh/uv/), and API keys for the parts you rerun: `ANTHROPIC_API_KEY`, `DEEPSEEK_API_KEY`, `OPENROUTER_API_KEY` (embeddings).

Rebuild the report from the published graded results (no API calls; NoLiMa rows will be empty):

```bash
mkdir -p results && tar -xzf data/results/p2_judged.tar.gz
uv run python -I harness/build_html.py        # -> report.html, report.en.html
```

Full pipeline (downloads data into `data/raw/`; see `docs/PLAN.md` for details and costs):

```bash
uv run python -I scripts/check_datasets.py          # discriminative check of candidate datasets
uv run python -I scripts/kb2_closed_book.py 600     # closed-book filter for MuSiQue candidates
uv run python -I scripts/build_kb2.py               # corpora S300/S900/S3000 + question set
uv run python -I scripts/translate_questions.py questions_v2/questions.jsonl
uv run python -I scripts/kb2_validate.py && uv run python -I scripts/kb2_validate_bright.py   # leak check 1
uv run python -I scripts/kb2_leak_check2.py && uv run python -I scripts/kb2_leak_check2_bright.py  # leak check 2
uv run python -I scripts/kb2_gold_audit.py          # gold audit; decisions reviewed into questions_v2/gold_decisions.json
uv run python -I scripts/kb2_finalize.py            # apply exclusions and aliases
uv run python -I scripts/build_indexes.py kb2/S3000 kb2/S900 kb2/S300
uv run python -I scripts/digest_wiki.py map kb2/S3000 --model haiku && uv run python -I scripts/digest_wiki.py reduce kb2/S300 kb2/S900 kb2/S3000 --model haiku
uv run python -I harness/run.py --model haiku --configs A1:R1_grep,A0:R3_hybrid --scales 300,900,3000 --workers 8 --budget 20 --out results/p2_haiku.jsonl
uv run python -I harness/judge.py results/p2_haiku.jsonl --batch
uv run python -I harness/build_html.py
```

Tests for the scoring and statistics code (no API calls): `uv run pytest`.

## Layout

| Path | What |
|---|---|
| `harness/` | model adapters (Anthropic, DeepSeek, OpenRouter), KB tools, runner, Opus grader, report builders, significance tests (`stats.py`) |
| `retrievers/` | chunking, BM25, dense (API embeddings + numpy), hybrid, QMD-style enhanced hybrid |
| `scripts/` | dataset checks, corpus / question construction, validation, LLM-wiki digest; v1 scripts (RepliQA) kept for the record |
| `questions_v2/` | question set (MuSiQue + BRIGHT) with translations, leak checks, gold audit and closed-book results |
| `report/` | dataset discriminative check, closed-book checks, digest comparison, audit before / after |
| `tests/` | unit tests for scoring rules, recommendation, exclusions and statistics |
| `docs/PLAN.md` | the full plan and log, including why the first dataset choice (RepliQA) was dropped |

Total spend for the published run: about $823 on Claude (models, digests, grading, $32 of it for the audit) and $61 on DeepSeek.

## License

Code: [MIT](LICENSE). Derived data keeps the license of its source dataset (CC BY 4.0 for MuSiQue and BRIGHT — please cite the original papers).

---

### 中文摘要

在 300 / 900 / 3000 篇的个人 md 知识库上，比较 3 个模型 × 11 种检索方式（单次 / 两轮 / 智能体 × grep、BM25、混合、QMD 式增强混合、LLM-wiki）。结论（2026-11-11 题库审计后）：**DeepSeek V4.1 Flash + 智能体 grep** 在 900 / 3000 篇最好（3000 篇 88%），但与 Sonnet + grep（84%）的差距不显著；单次检索的瓶颈在检索器，智能体明显优于单次检索；语义检索工具召回更高但答得并不更好，3000 篇以内 grep + 读文件就够。审计删除了 56 道金标有问题的 MuSiQue 题、按体量排除了更多泄漏的不可回答题，前后对比见 `report/audit_impact.md`。完整报告见 `report.html`，方法与数据来源见报告附录和 `docs/PLAN.md`。

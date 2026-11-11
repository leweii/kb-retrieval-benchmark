# kb-retrieval-benchmark

**Which retrieval method and which model should a personal markdown knowledge base use at 300, 900 and 3,000 documents?**

3 models × 11 retrieval setups × 3 knowledge-base sizes — 31,713 answers, all graded.

- Report: [`report.en.html`](report.en.html) (English) · [`report.html`](report.html) (中文) — open locally in a browser
- Method, data sources and question construction: appendix of the report; full working notes in [`docs/PLAN.md`](docs/PLAN.md) (Chinese)

## Results (3,000 documents, memory-clean MuSiQue set)

| Setup | Accuracy | Says "not in the KB" when it isn't | p50 latency | Cost / question |
|---|--:|--:|--:|--:|
| **DeepSeek V4.1 Flash + agent with grep** | **70%** | 78% | 61 s | $0.0044 |
| Claude Sonnet 5.5 + agent with grep | 62% | 79% | 21 s | $0.032 |
| Claude Haiku 5.5 + agent with grep | 46% | 86% | 41 s | $0.0032 |
| DeepSeek V4.1 Flash, single-shot hybrid retrieval | 39% | 83% | 8 s | $0.0012 |

Paired McNemar tests on the same 168 questions: DeepSeek vs Sonnet p = 0.015; agent vs single-shot p < 0.001.

What the numbers say:

1. **Single-shot retrieval is bounded by the retriever** (BM25 → hybrid → QMD-style enhanced: 15–19% → 29–39% → 43–55% accuracy across the three models); the model matters much less.
2. **With an agent, the model matters more than the retrieval backend**: ~20–25 points between models, ~10 points between six retrieval backends for the same model.
3. **More recall ≠ more accuracy**: semantic search tools surface more evidence than grep but agents answer worse with them — likely because each search dumps many long, partly confusable passages into context. At ≤ 3,000 documents, grep + read is enough.
4. **DeepSeek's lead comes from searching harder** (2.2× the tool calls, +14 points full-evidence recall), not from reading better: given identical passages, Sonnet and DeepSeek answer about equally well. Caveat: Claude ran at effort *medium*, DeepSeek at its default *high*.
5. **Haiku finds evidence but doesn't use it** — it refuses 32–43% of the time even when all evidence is in front of it.
6. An LLM-wiki digest (summary + entity pages) does not lower accuracy and helps hybrid search, but the Haiku digest drops ~23% of answer facts.

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
uv run python -I scripts/kb2_validate.py && uv run python -I scripts/kb2_validate_bright.py && uv run python -I scripts/kb2_finalize.py
uv run python -I scripts/build_indexes.py kb2/S3000 kb2/S900 kb2/S300
uv run python -I scripts/digest_wiki.py map kb2/S3000 --model haiku && uv run python -I scripts/digest_wiki.py reduce kb2/S300 kb2/S900 kb2/S3000 --model haiku
uv run python -I harness/run.py --model haiku --configs A1:R1_grep,A0:R3_hybrid --scales 300,900,3000 --workers 8 --budget 20 --out results/p2_haiku.jsonl
uv run python -I harness/judge.py results/p2_haiku.jsonl --batch
uv run python -I harness/build_html.py
```

## Layout

| Path | What |
|---|---|
| `harness/` | model adapters (Anthropic, DeepSeek, OpenRouter), KB tools, runner, Opus grader, report builders |
| `retrievers/` | chunking, BM25, dense (API embeddings + numpy), hybrid, QMD-style enhanced hybrid |
| `scripts/` | dataset checks, corpus / question construction, validation, LLM-wiki digest; v1 scripts (RepliQA) kept for the record |
| `questions_v2/` | question set (MuSiQue + BRIGHT) with translations, validation and closed-book results |
| `report/` | dataset discriminative check, closed-book checks, digest comparison |
| `docs/PLAN.md` | the full plan and log, including why the first dataset choice (RepliQA) was dropped |

Total spend for the published run: about $790 on Claude (models, digests, grading) and $61 on DeepSeek.

## License

Code: [MIT](LICENSE). Derived data keeps the license of its source dataset (CC BY 4.0 for MuSiQue and BRIGHT — please cite the original papers).

---

### 中文摘要

在 300 / 900 / 3000 篇的个人 md 知识库上，比较 3 个模型 × 11 种检索方式（单次 / 两轮 / 智能体 × grep、BM25、混合、QMD 式增强混合、LLM-wiki）。结论：**DeepSeek V4.1 Flash + 智能体 grep** 在三档都最好；单次检索的瓶颈在检索器，智能体模式下模型比检索方式更重要；语义检索工具召回更高但会引入噪声，3000 篇以内 grep + 读文件就够。完整报告见 `report.html`，方法与数据来源见报告附录和 `docs/PLAN.md`。

---
title: Text To SQL Analytics Assistant
emoji: 📊
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
license: mit
short_description: English to safe SQL + chart + insight over DuckDB.
---

# Text-to-SQL Analytics Assistant

Ask business questions in English. Get a grounded SQL query, a chart, and a
short insight — all without a human analyst in the loop.

Built as a portfolio project for GenAI / Data Scientist roles. The emphasis is
on **engineering around the LLM**, not just prompting it: schema-aware
retrieval, SQL safety validation with a real parser, and a measurable
execution-accuracy benchmark.

> **Status:** working end-to-end on the Chinook music-store dataset (DuckDB).
> Reproducible eval report lives at [`evals/last_run.md`](evals/last_run.md).

## Eval results on a 20-question benchmark

| Metric | Value |
|---|---|
| Execution accuracy | **95%** (19/20) |
| SQL validity rate  | 95% |
| Easy / Medium / Hard | 5/5 · 7/7 · 7/8 |
| Latency (median)   | 11.6 s |
| Latency (p95)      | 15.7 s |

Model: `qwen/qwen3.8-27b` on Groq free tier. The single failure is Q20
(nested-subquery alias scoping) — a real model slip, not a comparator artifact.
Rerun with `python evals/run_eval.py`.

---

## Why this project exists

Most "text-to-SQL" demos on GitHub stuff the entire schema into the prompt,
return raw strings with no safety layer, and never report accuracy. Hiring
managers see through that.

This repo demonstrates:

- **Schema retrieval**, not schema dumping — scales to 200-table warehouses
- **SQL safety validator** using `sqlglot`, not regex
- **Execution-accuracy evaluation** on a 20-question benchmark
- **A real Gradio app on HF Spaces**, not just a notebook

---

## Architecture

```
┌──────────────┐    ┌──────────────────┐    ┌──────────────┐    ┌─────────────┐
│  User asks a │───▶│ Schema retriever │───▶│ LLM          │───▶│  sqlglot    │
│  question in │    │ (sentence-trans- │    │ generates    │    │  validator  │
│  English     │    │  formers + top-k)│    │ DuckDB SQL   │    │ (SELECT     │
└──────────────┘    └──────────────────┘    └──────────────┘    │  only +     │
                                                                 │  auto-LIMIT)│
                                                                 └─────┬───────┘
                                                                       │
                            ┌─────────────┐    ┌──────────────┐        │
                            │  Narrative  │◀───│ DuckDB (read-│◀───────┘
                            │  (LLM, 2-3  │    │ only exec,   │
                            │  sentences, │    │ 10s timeout) │
                            │  grounded)  │    └──────┬───────┘
                            └─────────────┘           │
                                   │                  ▼
                                   ▼            ┌───────────┐
                            ┌─────────────┐     │ Plotly    │
                            │ Gradio UI   │◀────│ auto-pick │
                            │ (HF Spaces) │     │ chart     │
                            └─────────────┘     └───────────┘
```

### Pipeline details

| Step | Component | Why |
|---|---|---|
| 1 | `SchemaRetriever` embeds every table profile with `all-MiniLM-L6-v2`; top-k by cosine | Keeps the prompt short and lets the system scale beyond 10-table toys |
| 2 | Groq `qwen/qwen3.8-27b` + 5 few-shot examples | Free, fast, 95% execution accuracy on the eval |
| 3 | `sqlglot` parses the SQL, rejects anything that isn't a single `SELECT`/`WITH`, auto-adds `LIMIT 500` | Blocks `DROP`/`DELETE`/`UPDATE`/`INSERT`/`ALTER` even if the LLM is tricked |
| 4 | DuckDB executes read-only against the Chinook database | Zero-infrastructure analytical SQL |
| 5 | Narrator LLM call grounded on the returned rows (first 20) | Small, factual, cheap |
| 6 | Chart heuristic: time-series → line, categorical → bar, numeric pair → scatter | No chart when it would mislead |

---

## Setup

```bash
# 1. Clone & install
git clone https://github.com/Siddharth-Golla6/text-to-sql.git
cd text-to-sql
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate
pip install -r requirements.txt

# 2. Download Chinook + build DuckDB
python data/setup_db.py

# 3. Configure Groq
cp .env.example .env     # Windows: copy .env.example .env
# Edit .env and set GROQ_API_KEY (free key: https://console.groq.com/keys)

# 4. Run the Gradio app
python app.py
```

Open http://localhost:8501.

---

## Running the eval

```bash
python evals/run_eval.py
# or run just the first 5 questions
python evals/run_eval.py --limit 5
```

The runner:

1. Loads `evals/questions.jsonl` (20 natural-language questions + their gold SQL)
2. Asks the agent for each question
3. Executes **both** the model's SQL and the gold SQL in DuckDB
4. Compares the result sets semantically (row-order independent for aggregate
   queries, order-preserving when the question asks for a ranking)
5. Writes `evals/last_run.json` and a human-readable `evals/last_run.md`

### Why execution accuracy and not exact SQL match

Two SQL strings can look very different and still return the exact same result
(`JOIN ... ON` vs `USING`, column aliases, equivalent predicates). What matters
is the business answer — so we compare *the data*.

### Metrics reported

- `execution_accuracy` — fraction of questions whose model-SQL result matches the gold-SQL result
- `sql_validity_rate` — fraction of questions where the model produced a safe, parseable, executable `SELECT`
- `latency_mean_ms / median / p95` — end-to-end latency per question (includes retrieve + generate + validate + execute)
- Per-difficulty breakdown (easy / medium / hard)

---

## Repository structure

```
text-to-sql/
├── gradio_app.py                 # UI (Gradio Blocks)
├── agents/
│   ├── sql_agent.py              # orchestration + rate-limit retry
│   ├── schema_retriever.py       # table profiles + embeddings
│   ├── safety_validator.py       # sqlglot-based SQL guard
│   └── narrator.py               # grounded insight generator
├── prompts/
│   ├── sql_generation.txt        # system prompt + 5 few-shots
│   └── narration.txt
├── evals/
│   ├── questions.jsonl           # 20 benchmark questions
│   ├── run_eval.py               # execution-accuracy harness
│   ├── last_run.json             # eval output
│   └── last_run.md               # readable report (95% accuracy)
├── data/
│   ├── setup_db.py               # downloads Chinook + builds DuckDB
│   └── chinook.duckdb            # generated locally; built on HF Spaces first run
├── app.py                        # HF Spaces entry point (bootstraps DB)
├── deploy_to_hf.py               # one-command Space deployment
├── requirements.txt
├── .env.example
└── README.md
```

---

## Design decisions

1. **Groq free tier (`qwen/qwen3.8-27b`)** over OpenAI/Claude.
   Reason: zero cost, interactive-fast, best execution accuracy on our eval
   among the open-weight models Groq offers.
2. **DuckDB** over Postgres/MySQL.
   Reason: single-file, no server, reads Parquet/SQLite directly, massively
   faster for the demo. Dialect is close enough to Postgres that the SQL ports
   cleanly.
3. **`sqlglot` as the safety layer** instead of regex.
   Regex-based guards get fooled by comments, string literals, and clever
   formatting. A proper parser never does.
4. **Schema retrieval instead of schema dumping.**
   Even 11 tables benefit from focus; a 200-table warehouse breaks the "dump
   it in the prompt" approach entirely.
5. **Few-shot examples hard-coded, not retrieved.**
   For this schema size the four examples cover the main query patterns. Scaling
   to larger warehouses would add example retrieval, but that's premature here.

---

## Limitations (interview-ready honesty)

- **No column-level retrieval.** The retriever picks tables; the LLM has to
  pick columns. For wide tables (100+ columns) you would also embed columns.
- **No cross-column constraints learned from data.** The LLM assumes
  `BillingCountry` is in `Invoice`; nothing enforces that if the schema
  changes.
- **LLM-as-judge NOT used for narratives.** The narrative is factual *because*
  it's grounded on the actual result rows, but it is not independently graded.
- **No row-level security.** This is a read-only demo. Production would need
  per-user policies (see the "scaling" notes below).
- **No multi-turn memory.** Each question is independent. A follow-up like
  "and for 2011?" won't work — that's the Level 2 project in the roadmap.

---

## Scaling notes (interview talking points)

- 200-table schema: add column-level embeddings; retrieve top-k tables, then
  top-k columns within those.
- Row-level security: inject a `WHERE` filter server-side after the LLM returns
  the SQL — rewrite the AST with `sqlglot` to attach `tenant_id = :current_user_tenant`.
- Cost: 20-question eval runs in roughly 60–90s on Groq free tier and
  consumes ~15k tokens total.
- Latency: median end-to-end is dominated by LLM generation (~500–1200 ms on
  Groq). Retrieve + execute together are typically under 50 ms.

---

## Resume bullet

> Built a text-to-SQL analytics assistant over an 11-table Chinook warehouse in
> DuckDB; achieved **95% execution accuracy** on a 20-question benchmark using
> schema-aware retrieval (sentence-transformers), `sqlglot`-based SQL safety
> validation, and a grounded insight narrator. Deployed on Hugging Face Spaces
> (Streamlit) with reproducible eval harness.

---

## License

MIT.

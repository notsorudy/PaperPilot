<p align="center">
  <h1 align="center">📄 PaperPilot — Autonomous Deep Research Agent</h1>
  <p align="center">
    An AI-powered literature review pipeline that autonomously breaks down complex research questions, queries academic databases (ArXiv), extracts full-text PDFs, indexes content into a persistent <strong>Qdrant</strong> vector database, performs <strong>Cross-Encoder neural re-ranking</strong>, and synthesizes fully cited, fact-checked academic reviews with <strong>Human-in-the-Loop (HITL)</strong> oversight.
  </p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.10+"/>
  <img src="https://img.shields.io/badge/LangGraph-StateGraph_DAG-FF6F00?style=for-the-badge&logo=langchain&logoColor=white" alt="LangGraph"/>
  <img src="https://img.shields.io/badge/Qdrant-Vector_Database-DC382D?style=for-the-badge&logo=qdrant&logoColor=white" alt="Qdrant"/>
  <img src="https://img.shields.io/badge/ArXiv-API-B31B1B?style=for-the-badge&logo=arxiv&logoColor=white" alt="ArXiv"/>
  <img src="https://img.shields.io/badge/HuggingFace-Models-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" alt="HuggingFace"/>
</p>

---

## 📖 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [🛠️ Detailed Technical Deep Dive (Architecture Manual)](file:///c:/Users/anshu/Desktop/papers/TECHNICAL_DEEP_DIVE.md)
- [📈 Benchmarks & Evaluation](#-benchmarks--evaluation)
- [System Architecture](#-system-architecture)
- [Human-in-the-Loop (HITL) Oversight](#-human-in-the-loop-hitl-oversight)
- [Pipeline Deep Dive](#-pipeline-deep-dive)
  - [Node 1 — Query Planning (`generate_search_queries`)](#node-1--query-planning-generate_search_queries)
  - [Node 2 — Query Optimization (`optimize_query`)](#node-2--query-optimization-optimize_query)
  - [Node 3 — Paper Search (`search_papers`)](#node-3--paper-search-search_papers)
  - [Node 4 — Query Broadening / Error Recovery (`broaden_query`)](#node-4--query-broadening--error-recovery-broaden_query)
  - [Node 5 — Paper Ranking (`rank_papers`)](#node-5--paper-ranking-rank_papers)
  - [Node 6 — Full-Text Extraction (`extract_text`)](#node-6--full-text-extraction-extract_text)
  - [Node 7 — Persistent Qdrant RAG & Re-ranking (`retrieve_chunks`)](#node-7--persistent-qdrant-rag--re-ranking-retrieve_chunks)
  - [Node 8 — Structured Synthesis (`generate_answer`)](#node-8--structured-synthesis-generate_answer)
  - [Node 9 — Fact-Checking Self-Check (`verify_answer`)](#node-9--fact-checking-self-check-verify_answer)
  - [Node 10 — BibTeX Citations (`generate_citations`)](#node-10--bibtex-citations-generate_citations)
- [Data Flow & State Management](#-data-flow--state-management)
- [Models & Embeddings](#-models--embeddings)
- [Schemas & Structured Outputs](#-schemas--structured-outputs)
- [Prompt Engineering](#-prompt-engineering)
- [External Data Sources](#-external-data-sources)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Usage Example](#-usage-example)
- [Evaluation & Benchmark Suite](#-evaluation--benchmark-suite)
- [Configuration & Tuning](#-configuration--tuning)
- [Tech Stack](#-tech-stack)
- [Limitations & Future Roadmap](#-limitations--future-roadmap)
- [License](#-license)

---

## 🧭 Overview

**PaperPilot** is an autonomous research system designed to synthesize high-quality, grounded literature reviews from complex natural language queries. 

Given an academic question — for example:
> *"Can you find the official Self-RAG paper and explain its architecture, how the critique tokens work, and compare its performance to standard RAG models?"*

PaperPilot executes a stateful graph that:
1. **Deconstructs** the query into intent, reasoning, and academic sub-queries.
2. **Rewrites** search queries into concise keyword targets optimized for academic search engines.
3. **Retrieves & Deduplicates** candidate papers from ArXiv (with built-in query-broadening retry loops).
4. **Ranks** candidate papers using a cross-encoder neural model (`ms-marco-MiniLM-L-6-v2`).
5. **Downloads & Parses** full-text PDFs concurrently into semantic text chunks.
6. **Embeds & Stores** chunks into a persistent on-disk **Qdrant** vector database (`.qdrant_data/`).
7. **Performs Two-Stage RAG**: dense cosine similarity retrieval (top 30) followed by cross-encoder re-ranking (top 15).
8. **Synthesizes** a structured literature review with explicit flags for any aspects lacking evidence (`unanswered_aspects`).
9. **Audits Grounding** through an automated LLM-as-judge + Cross-Encoder verification pass to compute a **faithfulness score**.
10. **Generates** formal BibTeX citations ready for LaTeX publication.

Throughout execution, **Human-in-the-Loop (HITL)** gates allow researchers to inspect and approve the generated search plan and retrieved paper list before expensive PDF ingestion and synthesis occur.

---

## ✨ Key Features

| Capability | Technical Implementation |
|---|---|
| **🧠 Autonomous Planning** | LLM decomposes natural language questions into structured Pydantic search plans with intent classification and rationale. |
| **🔧 ArXiv Query Optimization** | LLM-driven query distillation producing concise 2–3 keyword searches tailored for strict academic indexers. |
| **🛡️ Self-Correcting Retry Loop** | LangGraph conditional edges detect empty or low-yield searches (< 3 papers or < 5 chunks) and dynamically route to a `broaden_query` expansion node. |
| **🧑‍🔬 Human-in-the-Loop (HITL)** | Built-in `MemorySaver` checkpointer with `interrupt_before` review gates at Phase 1 (Plan) and Phase 2 (Ranked Papers). |
| **💾 Persistent Qdrant Vector Store** | Ingested paper chunks are indexed and persisted on disk (`.qdrant_data/`) keyed by collection hash for zero re-indexing latency on repeated queries. |
| **⚡ Asynchronous I/O Concurrency** | Fully async pipeline using `asyncio.gather()` and thread pooling for parallel ArXiv searches and PDF downloads. |
| **📊 Two-Stage Neural Re-Ranking** | Bi-encoder dense retrieval (`all-MiniLM-L6-v2`) + Cross-Encoder re-ranking (`ms-marco-MiniLM-L-6-v2`) on both paper metadata and text excerpts. |
| **⚠️ Insufficient Evidence Handling** | Structured `SynthesisOutput` explicitly isolates and flags ungrounded user questions (`unanswered_aspects`) rather than hallucinating. |
| **🔍 Neural Claim Verification Pass** | Automated LLM claim extraction and cross-encoder score thresholding to calculate empirical **faithfulness accuracy**. |
| **📚 Publication-Ready Citations** | Node formats all retrieved paper metadata into standard BibTeX `@article` citation blocks. |

---

## 📈 Benchmarks & Empirical Evaluation

PaperPilot was benchmarked on live academic research queries, measuring I/O concurrency speedup, neural citation faithfulness, PDF chunk indexing scale, and full pipeline latency.

### 📊 Benchmark Summary Table

| Metric | Measured Value | Benchmark Configuration / Methodology |
|---|---|---|
| **⚡ I/O Concurrency Speedup** | **62.5% reduction in wall-clock time** (2.67x faster) | Parallel `asyncio.gather()` vs. sequential baseline (3.36s vs. 8.96s) across 3 query sets |
| **🛡️ Citation & Claim Faithfulness** | **92.9% – 98.9%** grounded | Cross-Encoder scoring of 28 atomic factual claims against retrieved paper excerpts |
| **📄 PDF Pages & Text Ingestion** | **150 – 315 pages / topic** | Real-time concurrent downloading and parsing of full-text arXiv publications |
| **🧩 Chunks Indexed per Collection** | **744.6 chunks avg.** (up to 1,152 chunks) | Recursive character chunking (1000 chars, 200 overlap) into persistent Qdrant collections |
| **📚 High-Relevance Papers Retained** | **9.0 – 10.0 papers** | Filtered from candidate pool via `ms-marco-MiniLM-L-6-v2` cross-encoder scoring |
| **⏱️ End-to-End Autonomous Latency** | **146.35s avg.** (89.46s – 244.22s) | Complete pipeline: decomposition → search → PDF download → Qdrant RAG → synthesis → claim verification |

---

### 🔬 Detailed Benchmark Breakdown

#### 1. Concurrency Acceleration (`benchmark_async_vs_sequential.py`)
Measures the wall-clock time required to execute multi-query arXiv searches and retrieve full-text PDF documents:
- **Sequential Execution (One-by-one):** `8.96s` average
- **Concurrent Execution (`asyncio.gather`):** `3.36s` average
- **Empirical Speedup:** **62.5% faster (2.67x throughput multiplier)**

#### 2. Citation & Claim Faithfulness Audit (`benchmark_faithfulness.py`)
Audits generated literature reviews by extracting atomic factual claims and scoring them against source chunks using the `ms-marco-MiniLM-L-6-v2` neural cross-encoder:
- **Audit Case 1 (Self-RAG Architecture & Tokens):** 9/9 claims grounded (**100.0%**)
- **Audit Case 2 (LoRA & QLoRA Quantization):** 10/10 claims grounded (**100.0%**)
- **Audit Case 3 (Corrective RAG / CRAG Evaluator):** 7/9 claims grounded (**77.8%**)
- **Overall Faithfulness Score:** **92.9% (26 of 28 atomic claims empirically grounded)**

#### 3. 10-Query Full Pipeline Stress Test (`benchmark.py`)
Autonomous end-to-end evaluation across 10 diverse computer science topics (Self-RAG, LoRA/QLoRA, FlashAttention, Mixture of Experts, DPO vs. PPO, Speculative Decoding, Graph RAG, Agentic Workflows, etc.):
- **Average Chunks Embedded into Qdrant:** `744.6 chunks` per research query
- **Peak Collection Size:** `1,152 chunks` (315 pages parsed)
- **Mean Verification Score:** **98.9%** claim grounding rate

> Run the test harnesses locally via `uv run benchmark.py`, `uv run benchmark_async_vs_sequential.py`, `uv run benchmark_faithfulness.py`, or `uv run eval/run_eval.py`.

---

## 🏗️ System Architecture

PaperPilot is structured as a 10-node Directed Acyclic Graph (DAG) with state checkpointing and conditional retry loops:

```
                            User Research Query
                                     │
                                     ▼
                      ┌─────────────────────────────┐
                      │  generate_search_queries    │  [Phase 1: Planning]
                      └──────────────┬──────────────┘
                                     │
                                     ▼
                      ┌─────────────────────────────┐
                      │       optimize_query        │  [2-3 keyword distillation]
                      └──────────────┬──────────────┘
                                     │
                    [HITL Gate 1: Plan Approval] ⏸️
                                     │
                                     ▼
                      ┌─────────────────────────────┐◄─────────────────┐
                      │        search_papers        │                  │
                      └──────────────┬──────────────┘                  │
                                     │                                 │
                   (Yield < 3 papers?) ───[YES]──►┌────────────────┐   │
                                     │            │ broaden_query  ├───┘ (Retry loop 1)
                                   [NO]           └────────────────┘
                                     ▼
                      ┌─────────────────────────────┐
                      │         rank_papers         │  [Cross-Encoder Top-10]
                      └──────────────┬──────────────┘
                                     │
                  [HITL Gate 2: Paper Approval] ⏸️
                                     │
                                     ▼
                      ┌─────────────────────────────┐◄─────────────────┐
                      │        extract_text         │                  │
                      └──────────────┬──────────────┘                  │
                                     │                                 │
                   (Yield < 5 chunks?) ───[YES]────────────────────────┘ (Retry loop 2)
                                     │
                                   [NO]
                                     ▼
                      ┌─────────────────────────────┐
                      │       retrieve_chunks       │  [Qdrant Dense + Re-rank]
                      └──────────────┬──────────────┘
                                     │
                                     ▼
                      ┌─────────────────────────────┐
                      │       generate_answer       │  [Synthesis with Evidence Gaps]
                      └──────────────┬──────────────┘
                                     │
                                     ▼
                      ┌─────────────────────────────┐
                      │        verify_answer        │  [Neural Claim Audit & Scoring]
                      └──────────────┬──────────────┘
                                     │
                                     ▼
                      ┌─────────────────────────────┐
                      │      generate_citations     │  [BibTeX Formatting]
                      └──────────────┬──────────────┘
                                     │
                                     ▼
                                    END
```

---

## 🧑‍🔬 Human-in-the-Loop (HITL) Oversight

Research synthesis requires trust and control. Rather than running as an opaque black box, PaperPilot exposes **two interactive review gates**:

```
[Agent Starts]
   │
   ├─► Generates Plan & Optimized Queries
   │
   ⏸️ [HITL Gate 1] Researcher reviews Intent, Reasoning, and arXiv Queries.
   │                Option: Approve (Y), Modify, or Abort.
   │
   ├─► Searches arXiv & Scores Papers with Cross-Encoder
   │
   ⏸️ [HITL Gate 2] Researcher inspects Ranked Paper Titles, Authors, and URLs.
   │                Option: Approve PDF ingestion for top candidates or Abort.
   │
   └─► Executes PDF extraction, Qdrant RAG, Synthesis, Audit, and Citations.
```

- **Thread Checkpointing:** Uses LangGraph's `MemorySaver` to checkpoint state at interrupt boundaries.
- **Autonomous Mode:** For evaluation pipelines or API usage, pass `PlannerAgent(enable_hitl=False)` to execute end-to-end without terminal prompts.

---

## 🔬 Pipeline Deep Dive

### Node 1 — Query Planning (`generate_search_queries`)
- **File:** `nodes/planner.py`
- **Model:** Cerebras (`gpt-oss-120b`) with Groq fallback (`llama-3.1-8b-instant`)
- **Schema:** `SearchPlan` (Pydantic)
- **Role:** Decomposes the user's inquiry into intent classification, reasoning strategy, expected academic sources, and 3–6 initial search targets.

### Node 2 — Query Optimization (`optimize_query`)
- **File:** `nodes/optimize_query.py`
- **Model:** Cerebras / Groq fallback
- **Schema:** `OptimizeQuerySchema`
- **Role:** Rewrites raw query phrases into dense 2–3 keyword queries suitable for arXiv title/abstract indexing (e.g., `"How does critique work in Self-RAG?"` → `"Self-RAG critique"`).

### Node 3 — Paper Search (`search_papers`)
- **File:** `nodes/search_arxiv_node.py`
- **Engine:** ArXiv Web API (`arxiv` client)
- **Role:** Executes all queries concurrently via `asyncio.gather()`. Deduplicates results across searches by normalized title.

### Node 4 — Query Broadening / Error Recovery (`broaden_query`)
- **File:** `nodes/broaden_query_node.py`
- **Model:** Cerebras / Groq fallback
- **Role:** Activated automatically when search yields < 3 candidate papers. Reformulates queries to broader domain terms and retries paper search.

### Node 5 — Paper Ranking (`rank_papers`)
- **File:** `nodes/rank_papers_node.py`
- **Model:** `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **Role:** Forms `(query, title + abstract)` pairs and scores papers jointly. Filters candidate pool down to the top 10 most relevant publications.

### Node 6 — Full-Text Extraction (`extract_text`)
- **File:** `nodes/extract_text_node.py`
- **Engine:** `requests`, `PyPDFLoader`, `RecursiveCharacterTextSplitter`
- **Role:** Concurrently downloads PDFs, extracts text pages, attaches metadata (`paper_id`, `title`), and chunks into 1000-char segments with 200-char overlap. Features automatic fallback to abstract injection if PDF download fails.

### Node 7 — Persistent Qdrant RAG & Re-ranking (`retrieve_chunks`)
- **File:** `nodes/retrieve_node.py`
- **Vector DB:** Qdrant (on-disk persistence in `.qdrant_data/`)
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` (384-dim)
- **Re-ranker:** `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **Role:** Stores document embeddings in collection `papers_<hash>`. Performs dense cosine vector search for top 30 chunks, then neural Cross-Encoder re-ranking down to the top 15 excerpts.

### Node 8 — Structured Synthesis (`generate_answer`)
- **File:** `nodes/generate_answer_node.py`
- **Model:** Cerebras / Groq fallback
- **Schema:** `SynthesisOutput`
- **Role:** Synthesizes the final literature review grounded strictly in the top 15 excerpts. Populates `unanswered_aspects` for sub-questions unsupported by retrieved chunks.

### Node 9 — Fact-Checking Self-Check (`verify_answer`)
- **File:** `nodes/verify_answer_node.py`
- **Mechanism:** LLM-as-judge claim extraction + Cross-Encoder scoring
- **Role:** Extracts 5–10 atomic factual claims from the literature review and tests neural alignment against source chunks. Computes a quantitative `faithfulness_score` and flags any ungrounded assertions.

### Node 10 — BibTeX Citations (`generate_citations`)
- **File:** `nodes/generate_citations_node.py`
- **Role:** Formats metadata for all source papers into publication-grade BibTeX `@article` entries with keys, authors, year, and direct arXiv identifiers.

---

## 📊 Data Flow & State Management

The entire pipeline operates on a centralized `ResearchState` TypedDict:

```python
class ResearchState(TypedDict, total=False):
    query: str                              # Original user question
    plan: SearchPlan                        # Structured research plan
    optimized_queries: List[OptimizeQuerySchema]  # ArXiv keyword targets
    retry_count: int                        # Broadening retry counter
    papers: List[PaperMetadata]             # Retrieved & ranked papers
    chunks: List[Any]                       # Extracted PDF text chunks
    relevant_chunks: List[Any]              # Top 15 re-ranked excerpts
    final_answer: str                       # Literature review + audit
    unanswered_aspects: List[str]           # Flagged evidence gaps
    faithfulness_score: float               # Claim grounding ratio (0.0 - 1.0)
    citations: str                          # BibTeX bibliography block
```

---

## 🤖 Models & Embeddings

### Cloud LLMs (Fast Inference APIs)
- **Primary:** Cerebras (`gpt-oss-120b`) via `langchain-cerebras`
- **Fallback:** Groq (`llama-3.1-8b-instant`) via `langchain-groq`

### Local Neural Models (Hugging Face)
- **Bi-Encoder Embeddings:** `all-MiniLM-L6-v2` (384 dimensions, cosine distance)
- **Neural Re-Ranker:** `cross-encoder/ms-marco-MiniLM-L-6-v2` (joint query-document relevance scoring)

---

## 📁 Project Structure

```
papers/
├── app/
│   ├── __init__.py
│   ├── main.py                       # Interactive HITL CLI entrypoint
│   ├── llm.py                        # Cerebras & Groq LLM configurations
│   ├── models.py                     # Global Hugging Face model loaders
│   └── vectorstore.py                # Persistent Qdrant client configuration
│
├── graph/
│   ├── __init__.py
│   └── planner_graph.py              # LangGraph StateGraph definition (10 nodes)
│
├── nodes/
│   ├── __init__.py
│   ├── planner.py                    # Node 1: Query decomposition
│   ├── optimize_query.py             # Node 2: ArXiv query distillation
│   ├── search_arxiv_node.py          # Node 3: Concurrent paper retrieval
│   ├── broaden_query_node.py         # Node 4: Query expansion retry node
│   ├── rank_papers_node.py           # Node 5: Cross-Encoder paper ranking
│   ├── extract_text_node.py          # Node 6: Parallel PDF ingestion & chunking
│   ├── retrieve_node.py              # Node 7: Qdrant dense retrieval + re-ranking
│   ├── generate_answer_node.py       # Node 8: Structured synthesis
│   ├── verify_answer_node.py         # Node 9: Fact-checking claim verification
│   └── generate_citations_node.py    # Node 10: BibTeX citation formatter
│
├── prompts/
│   ├── __init__.py
│   ├── planner_prompts.py            # Planning & decomposition prompt
│   ├── optimize_query_prompt.py      # Keyword distillation prompt
│   ├── broaden_query_prompt.py       # Query expansion prompt
│   ├── synthesis_prompts.py          # Grounded synthesis prompt
│   └── verify_prompt.py              # Claim extraction prompt
│
├── schemas/
│   ├── __init__.py
│   ├── planner_schema.py             # SearchPlan Pydantic model
│   ├── optimize_query_schema.py      # OptimizeQuerySchema & BroadenedQueriesPlan
│   ├── paper_schema.py               # PaperMetadata model
│   ├── synthesis_schema.py           # SynthesisOutput structured review model
│   └── verify_schema.py              # ExtractedClaims & VerificationResult models
│
├── state/
│   ├── __init__.py
│   └── research_state.py             # ResearchState TypedDict definition
│
├── tools/
│   ├── arxiv.py                      # ArXiv API search tool
│   └── openalex.py                   # OpenAlex adapter (scaffolded)
│
├── eval/
│   ├── golden_set.json               # 12-case gold evaluation set
│   └── run_eval.py                   # Automated evaluation harness
│
├── benchmark.py                      # 10-query full pipeline benchmark
├── benchmark_async_vs_sequential.py  # Concurrency timing comparison script
├── benchmark_faithfulness.py         # Citation faithfulness audit script
├── pyproject.toml
├── requirements.txt
└── .env                              # API keys (GROQ_API_KEY, CEREBRAS_API_KEY)
```

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10+
- `uv` (Fast package manager) or standard `pip`
- Cerebras API Key and/or Groq API Key

### 2. Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/your-username/papers.git
cd papers

# Create virtual environment and install dependencies
uv venv
uv pip install -r requirements.txt
```

### 3. Environment Configuration

Create a `.env` file in the root directory:

```env
# Required (at least one LLM key)
GROQ_API_KEY=your_groq_api_key_here
CEREBRAS_API_KEY=your_cerebras_api_key_here
```

### 4. Running the Agent (Interactive HITL)

```bash
uv run app/main.py
```

Follow the on-screen prompts to review and approve the generated search queries and retrieved candidate papers.

---

## 🧪 Evaluation & Benchmark Suite

Run any of the automated evaluation and benchmarking scripts:

```bash
# 1. Full 10-Query Autonomous Benchmark (saves benchmark_results.json)
uv run benchmark.py

# 2. Measure Concurrency Speedup (Async vs. Sequential)
uv run benchmark_async_vs_sequential.py

# 3. Citation Faithfulness & Neural Claim Audit (saves faithfulness_results.json)
uv run benchmark_faithfulness.py

# 4. Gold Standard Concept Recall Evaluation (saves eval/eval_results.json)
uv run eval/run_eval.py
```

---

## 💡 Usage Example

### Interactive CLI Session

```
=========================================================
   📄 PaperPilot — Autonomous Literature Review Agent   
=========================================================

Research Query: "Can you find the official Self-RAG paper and explain its architecture, how the critique tokens work, and compare its performance to standard RAG models?"

Executing Phase 1: Planning and Query Optimization...

============================================================
🔍 [HITL Gate 1] RESEARCH PLAN REVIEW
============================================================
  • Intent:    paper_comparison
  • Reasoning: The user is asking about the specific Self-RAG architecture...
  • Expected Sources: arXiv, ACL Anthology

  • Optimized Search Queries for arXiv:
    1. "Self-RAG architecture"  (keywords: Self-RAG, architecture)
    2. "critique reflection tokens"  (keywords: critique, reflection, tokens)
    3. "Self-RAG RAG comparison"  (keywords: Self-RAG, RAG, comparison)
============================================================

[HITL] Approve research plan and proceed to search arXiv? (Y/n): Y

Executing Phase 2: Searching arXiv & Neural Paper Ranking...
Ranking 12 candidate papers using Hugging Face CrossEncoder...

============================================================
📚 [HITL Gate 2] RETRIEVED PAPERS REVIEW
============================================================
Total Candidate Papers Ranked: 10

  1. [arXiv API] Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection
     Authors: Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, Hannaneh Hajishirzi
     PDF URL: https://arxiv.org/pdf/2310.11511v1

  2. [arXiv API] Corrective Retrieval Augmented Generation
     Authors: Shi-Qi Yan, Jia-Chen Gu, Yun Zhu, Zhen-Hua Ling
     PDF URL: https://arxiv.org/pdf/2401.15884v1
============================================================

[HITL] Approve downloading full PDFs and building Qdrant RAG index for top 10 papers? (Y/n): Y

Executing Phase 3: Text Extraction, Qdrant Retrieval, Synthesis & Self-Check...
Downloading: Self-RAG: Learning to Retrieve, Generate...
Extracted 42 pages. Created 184 text chunks.
Persisted 184 chunks into Qdrant collection 'papers_7f8a12b4e910'.
Searching Qdrant for top 30 most relevant chunks...
Re-ranking retrieved chunks using Hugging Face CrossEncoder...
Selected top 15 chunks after CrossEncoder re-ranking.
Feeding Top 15 chunks to LLM for final synthesis...

🔍 Running Step 8: Neural Claim-Verification Audit (Self-Check)...
Extracted 7 atomic claims to audit against 15 source chunks.
Verification Audit Result: 7/7 claims grounded (Faithfulness Score: 100.0%)

=================================================================
           📑 FINAL SYNTHESIZED LITERATURE REVIEW
=================================================================

# Self-RAG: Architecture, Critique Mechanism, and Performance Comparison

### 1. Architectural Overview
Self-RAG (Self-Reflective Retrieval-Augmented Generation) introduces an adaptive framework...

### 2. Reflection & Critique Tokens
Unlike conventional RAG models that retrieve passages unconditionally...
- `[Retrieve]`: Predicts whether external passage retrieval is necessary.
- `[IsRel]`: Evaluates whether the retrieved passage contains relevant information.
- `[IsSup]`: Checks if the generated response is supported by the passage.
- `[IsUse]`: Rates the overall usefulness of the generation.

### 3. Empirical Performance Comparison
On open-domain QA benchmarks (PopQA, TriviaQA), Self-RAG (7B/13B) outperforms standard RAG...

---
### 🛡️ Fact-Check & Verification Audit (Faithfulness: 100.0%)
**Audit Summary**: 7 of 7 atomic factual claims verified against source literature chunks.
✅ All audited claims have direct neural grounding in the retrieved paper excerpts.

=================================================================
                 📚 BIBTEX CITATIONS
=================================================================

@article{asai2023self,
  title     = {Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection},
  author    = {Akari Asai and Zeqiu Wu and Yizhong Wang and Avirup Sil and Hannaneh Hajishirzi},
  year      = {2023},
  journal   = {arXiv API},
  url       = {https://arxiv.org/pdf/2310.11511v1},
  eprint    = {2310.11511v1}
}
```

---

## ⚙️ Configuration & Tuning

| Parameter | Location | Default | Description |
|---|---|---|---|
| `chunk_size` | `nodes/extract_text_node.py` | `1000` | Target character count per text chunk |
| `chunk_overlap` | `nodes/extract_text_node.py` | `200` | Character overlap between consecutive chunks |
| `k` (Stage 1 Vector Search) | `nodes/retrieve_node.py` | `30` | Top dense candidates retrieved from Qdrant |
| `top_n` (Stage 2 Re-ranking) | `nodes/retrieve_node.py` | `15` | Top excerpts retained after Cross-Encoder scoring |
| `max_results` (arXiv) | `tools/arxiv.py` | `5` | Candidate papers fetched per search query |
| `SUPPORT_THRESHOLD` | `nodes/verify_answer_node.py` | `-0.5` | Cross-Encoder logit threshold for claim verification |

---

## 🛠️ Tech Stack

- **Orchestration:** [LangGraph](https://github.com/langchain-ai/langgraph) `StateGraph` with `MemorySaver`
- **Vector Database:** [Qdrant](https://qdrant.tech/) via `langchain-qdrant` (local on-disk persistence)
- **Embeddings:** [Sentence Transformers](https://sbert.net/) `all-MiniLM-L6-v2` (384-dim)
- **Neural Re-Ranker:** `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **LLM Providers:** [Cerebras](https://cerebras.ai/) (`gpt-oss-120b`) & [Groq](https://groq.com/) (`llama-3.1-8b-instant`)
- **PDF Extraction:** `pypdf` via LangChain `PyPDFLoader`
- **Validation:** Pydantic v2 structured schemas

---

## ⚠️ Limitations & Future Roadmap

- **Academic Index Scope:** Currently queries ArXiv as the primary production source. The repository includes an OpenAlex adapter (`tools/openalex.py`) ready for future multi-database routing.
- **Frontend UI:** Currently runs as an interactive CLI; web UI (Streamlit / Gradio) planned.
- **Token Streaming:** Output is synthesized in single structured blocks; token-by-token streaming is scheduled for upcoming releases.

---

## 📄 License

This project is licensed under the MIT License for academic and research use.

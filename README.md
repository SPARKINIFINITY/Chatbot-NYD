# Chatbot-NYD
<div align="center">

Document QA with Hybrid Retrieval (FAISS + BM25 + Cross‑Encoder) and FLAN‑T5

<br />

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.x-000.svg)](https://flask.palletsprojects.com/)
[![FAISS](https://img.shields.io/badge/FAISS-IP-green.svg)](https://github.com/facebookresearch/faiss)
[![HF Transformers](https://img.shields.io/badge/Transformers-4.x-ff69b4.svg)](https://huggingface.co/transformers)

</div>

AI chatbot system for answering questions over user‑uploaded content. The service ingests files, builds a vector index, performs hybrid retrieval (semantic + BM25) with optional Cross‑Encoder reranking, and generates grounded answers using a seq2seq LLM. Tabular queries on CSV/XLSX are answered via a lightweight table engine.

---

## Table of Contents
- Overview
- Architecture
- Quick Start
- Configuration
- Endpoints
- Models
- Operations
- Security & Privacy
- Troubleshooting
- Roadmap
- License

---

## Overview
- Hybrid retrieval: FAISS dense search + BM25 fusion on persisted chunk text.
- Quality controls: optional Cross‑Encoder reranking and document scoping by file/hash.
- Deterministic generation: FLAN‑T5 with greedy decoding for reproducible results.
- Tabular mode: direct answers from CSV/XLSX snapshots.
- Durable persistence: FAISS index, JSONL metadata, and on‑disk embedding cache.

## Architecture

Client → Flask API → Ingestion → Cleaning → Chunking → Embedding → FAISS Index
                                              │                        │
                                              └─> Metadata JSONL <─────┘

Query → Hybrid Retrieval (FAISS + BM25) → Cross‑Encoder Rerank(ANN) → Context → FLAN‑T5 → Answer
Tabular Queries → Tabular Engine (DataFrames) → Table/Metric


Key components
- utils/parsers.py: File parsing (PDF, DOCX, CSV/XLSX, TXT, code).
- utils/cleaning.py: Text normalization.
- utils/chunking.py: Chunk strategies (text/tabular).
- utils/embeddings.py: Sentence‑Transformers embeddings + cache.
- utils/vectorstore.py: FAISS persistence and search; metadata JSONL.
- utils/retrieval.py: Dense + BM25 fusion and Cross‑Encoder reranking.
- utils/llm.py: Generation pipeline with FLAN‑T5 (fallback to small).
- utils/tabular.py: Lightweight table engine for analytical queries.
- Backend/app.py: Flask routes and wiring.

## Quick Start
1. Prerequisites: Python 3.10+, pip, internet for first model downloads.
2. Install
   bash
   cd chatbot/Backend
   pip install -r requirements.txt
   
3. Run
   bash
   cd chatbot/Backend
   set DEBUG=false
   python app.py
   # http://127.0.0.1:5000
   
4. Use
   - Upload: open http://127.0.0.1:5000/upload.html.
   - Report (PDF export): http://127.0.0.1:5000/report.html → Ctrl+P → Save as PDF → enable Background Graphics.

## Configuration
Configure via environment variables (defaults in utils/config.py):

| Variable | Default | Description |
|---|---|---|
| HOST | 0.0.0.0 | Server host |
| PORT | 5000 | Server port |
| DEBUG | false | Flask debug mode |
| EMBEDDING_MODEL | sentence-transformers/all-MiniLM-L6-v2 | Embedding model name |
| CROSS_ENCODER | cross-encoder/ms-marco-MiniLM-L-6-v2 | Reranker model |
| GEN_MODEL | google/flan-t5-base | Generator model (fallback to flan-t5-small if load fails) |
| FAISS_DIM | 384 | Expected embedding dimension (auto‑rebuild on mismatch) |
| MAX_CONTEXT_CHARS | 8000 | Context length cap for prompts |
| ALLOWED_EXT | csv,xlsx,json,txt,pdf,docx,py,js,ts,java,cpp,md | Upload whitelist |

## Endpoints

GET  /health            → status ok
POST /upload            → form-data: file; returns indexing summary
POST /query             → { query, top_k, use_bm25, use_multiquery, use_rerank, filename?, file_hash? }
POST /ask               → compatibility endpoint; may return table output
GET  /files             → list indexed files
DELETE /files           → remove by filename or file_hash
GET  /                  → home.html
GET  /upload.html       → upload.html
GET  /report.html       → project report (exportable to PDF)


## Models
- Embeddings: sentence-transformers/all-MiniLM-L6-v2
- Cross‑Encoder: cross-encoder/ms-marco-MiniLM-L-6-v2
- Generator: google/flan-t5-base (fallback google/flan-t5-small)

Notes
- Deterministic decoding (do_sample=false, max_new_tokens=256).
- Model names can be changed via env vars; first use will download from Hugging Face.

## Operations
- Index persistence: vectorstore/faiss.index and vectorstore/metadata.jsonl.
- Embedding cache: vectorstore/emb_cache/ (keyed by file hash and chunk count).
- BM25 cache is invalidated on every upload/delete.
- Index rebuilds automatically when embedding dimension changes.

## Troubleshooting
- Models fail to load: ensure network access; try GEN_MODEL=google/flan-t5-small.
- FAISS dimension mismatch: the app rebuilds the index; if issues persist, remove vectorstore/ and re‑index.
- High latency on CPU: lower top_k, disable rerank (use_rerank=false).
- BM25 seems ineffective: ensure documents were uploaded; BM25 relies on persisted chunk text in metadata.

## Roadmap
- LLM‑driven multi‑query expansion and reciprocal rank fusion.
- Inline citations with span highlighting and source grounding.
- Advanced dense retrievers (ColBERTv2, Contriever) and domain tuning.

## License
For hackathon/demo use. Add an explicit license for public distribution.

🧠 Buffett Investing Mindset + AI Investing Agent

Learn & apply Warren Buffett’s investing principles (1977–2024) using AI, contextual embeddings, and RAG.

This project builds a structured knowledge base from all Berkshire Hathaway shareholder letters and uses contextual retrieval to create an AI agent that analyzes companies and explains investing concepts in Buffett’s style.

🚀 Project Goals

Extract investing wisdom from every Berkshire Hathaway shareholder letter (1977–2024)

Turn them into high-quality, context-aware chunks

Build contextual embeddings for accurate retrieval

Create a robust RAG pipeline

Build a Buffett-style investing assistant that can:

Explain Buffett’s thinking

Compare concepts across decades

Analyze companies using Buffett’s framework

📁 Project Structure
Buffett-Investing-Agent/
│
├── data/
│   ├── raw/                 # Raw HTML/PDF letters
│   ├── clean/               # Cleaned text
│   ├── processed/           # Final chunks with metadata + contextual text
│
├── scripts/
│   ├── download_letters.py  # Fetch all letters from the Berkshire site
│   ├── clean_letters.py     # HTML → clean text
│   ├── chunk_letters.py     # Chunking + metadata
│   ├── contextualize.py     # Add contextual summaries + ctx_text
│   ├── embed_chunks.py      # Create contextual embeddings
│   ├── build_index.py       # Build hybrid search index
│
├── agent/
│   ├── rag_pipeline.py      # Retrieval + generation pipeline
│   ├── prompts/             # System + RAG templates
│
├── app/
│   ├── api/                 # FastAPI backend
│   └── web/                 # Streamlit or Next.js UI
│
└── README.md

📚 90-Day Roadmap (High-Level)
Phase 1 — Data Acquisition & Preparation

Download shareholder letters (1977–2024)

Clean HTML → extract narrative + tables

Chunk & add metadata (context-aware)

Section detection

120–200 word chunks

Table chunks

Metadata (year, section, topics…)

Add contextual summaries (LLM-generated)

Build contextualized text (ctx_text)

Output:
chunks_with_ctx.parquet

Phase 2 — Embedding & Retrieval System

Contextual embeddings (embed ctx_text)

Hybrid retrieval index (BM25 + vectors)

Clustering & trend analysis

Float philosophy over time

Inflation commentary evolution

Derivatives (2002–2010)

Capital allocation principles

Phase 3 — RAG Pipeline Construction

Retrieval module (hybrid)

RAG answer generation

Evaluation with curated question sets

Phase 4 — Application Layer

Buffett mentor agent

Investing analysis assistant

Deployment (FastAPI + Streamlit/Next.js)

🧩 Contextual Embeddings (Core Innovation)

Traditional embeddings struggle with:

Decade-spanning topics

Similar concepts across different years

Section-specific nuances (“float” means something different in 1981 vs 2001)

This project uses contextual embeddings:

[Berkshire Hathaway Shareholder Letter — 2008]
Section: Derivatives

Summary:
This chunk describes Buffett’s concerns about derivatives risk during the
Global Financial Crisis, including counterparty exposure and leverage.

Original Text:
...


Embedding this context + text yields:

Higher retrieval accuracy

Better thematic clustering

Stronger grounding for RAG answers

Very clean time-series reasoning

🏗️ Tech Stack
Data & Processing

Python, BeautifulSoup, html5lib, regex

Pandas, PyArrow/Parquet

Chunking & Contextualization

OpenAI / Claude / LLaMA for summarization

Custom chunking logic

Embedding & Indexing

Contextual embeddings (OpenAI, Claude, or ST-based)

Qdrant, Elasticsearch, or ChromaDB

BM25 hybrid search

Reranking

RAG Pipeline

FastAPI backend

Custom prompt templates

Evaluation harness

Frontend

Streamlit or Next.js

Clean UI for investment analysis

🧪 Example Capabilities
Ask Buffett-style questions

“Explain float like Warren Buffett.”

“How did his view on inflation change from 1980 → 2010?”

“Why does he avoid tech stocks historically?”

“What does Buffett say about intrinsic value?”

Analyze modern companies

Use extracted principles to analyze businesses today

Evaluate quality, durability, competitive advantage

Rate management quality

Check alignment with Buffett’s criteria

📈 Future Extensions

Add 10-K and 10-Q parsing

Add economic papers Buffett references

Add Charlie Munger speeches

Build an “Invest Like Buffett” scoring framework

Train a fine-tuned model on your curated chunks

🤝 Contributing

PRs welcome!
Open issues for:

Improvements to chunker

Better contextualization prompts

Retrieval accuracy bugs

Adding new visualization modules

📜 License

MIT License — feel free to fork, modify, and build upon this.
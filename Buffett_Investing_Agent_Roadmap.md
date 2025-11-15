# Buffett Investing Mindset + Investing Agent (90-Day Roadmap)

**Goal:**  
Learn Warren Buffett’s investing mindset from his Letters to Shareholders (1977–2024) **while** building a Buffett-inspired Investing Agent using your skills in Python, ML, SQL, and LLMs.

---

## Phase 1 — Build the “Buffett Corpus” (Days 1–7)

**Goal:** Download, clean, and structure Buffett’s letters into a usable dataset.

### Main Outcomes
- All letters (1977–2024) converted to clean text.
- Text split into meaningful chunks (100–200 words).
- Each chunk labeled with metadata: `year`, `chunk_id`, optional `section`, `topics`.

### Tasks
1. **Collect data**
   - Download letters in HTML/PDF form.
   - Save them under `data/raw/1977_*.pdf`, etc.

2. **Extract & clean text**
   - Use PDF/HTML parsers to get raw text.
   - Remove headers, page numbers, footers, and boilerplate.
   - Normalize whitespace, fix broken lines.

3. **Chunk and add metadata**
   - Split text into chunks (e.g., 100–200 words or 3–6 sentences).
   - Attach metadata:
     - `year`
     - `chunk_id` (e.g., `1988_05`)
     - optional `section` if detectable.

4. **Store corpus**
   - Save as `letters_chunks.parquet` or `letters_chunks.csv`:
     - `year`, `chunk_id`, `text`, `section` (optional).

### Tech Stack (Phase 1)
- **Language:** Python 3.10+
- **Download & parsing:**
  - `requests`, `beautifulsoup4`
  - `pdfplumber` or `pymupdf`
- **Text & data handling:**
  - `re` (regex), `pandas`
  - `nltk` or `spaCy` (for sentence splitting, optional)
- **Storage:**
  - Local files: CSV / Parquet (`pandas.to_parquet`, `to_csv`)

---

## Phase 2 — Learn Buffett Through ML & Analytics (Days 8–21)

**Goal:** Use your data/ML skills to discover patterns in the letters and internalize Buffett’s mindset.

### Main Outcomes
- Topic clusters of Buffett’s writings (moats, inflation, buybacks, etc.).
- Time-series trends of key concepts (e.g., “inflation”, “intrinsic value”).
- A rough “Buffett knowledge graph” linking topics to years and examples.

### Tasks
1. **Embed all chunks**
   - Compute embeddings for each `text` chunk.
   - Store embeddings alongside metadata (e.g., separate file or in vector DB later).

2. **Cluster / topic model**
   - Run KMeans or HDBSCAN on embeddings.
   - Inspect top chunks per cluster.
   - Label clusters manually: e.g., “Inflation & Pricing Power”, “Moats”, “Buybacks & Capital Allocation”.

3. **Analyze concept trends**
   - Simple keyword searches over `text`:
     - “inflation”, “moat”, “intrinsic value”, “Mr. Market”, “buyback”.
   - Group by `year` and count frequencies.
   - Plot: term frequency vs year; cluster prevalence vs decade.

4. **Extract frameworks using LLM**
   - For each cluster / set of years, prompt an LLM to summarize:
     - Buffett’s rules for buying businesses.
     - His view on risk, leverage, and debt.
     - His principles on management, moats, valuation, buybacks.
   - Save summaries as markdown/JSON (this becomes your structured notes).

### Tech Stack (Phase 2)
- **Environment:** Jupyter / VS Code + Python
- **NLP & embeddings:**
  - `sentence-transformers` (e.g., `all-MiniLM-L6-v2`)
  - `torch`
- **Clustering & analysis:**
  - `scikit-learn` (KMeans, MiniBatchKMeans, etc.)
  - `umap-learn` (optional for 2D visualization)
  - `pandas` for groupby and aggregation
- **Visualization:**
  - `matplotlib` or `plotly`
  - Optional `seaborn`

---

## Phase 3 — Build the Buffett Investing Agent MVP (Days 22–60)

**Goal:** Turn the corpus into a working RAG-based Buffett Investing Agent with a simple UI.

### Main Outcomes
- RAG pipeline over the Buffett corpus.
- Backend API that answers questions using the letters.
- Simple UI (Streamlit or web frontend) to query the agent.
- Answers with:
  - Relevant Buffett quotes.
  - Your own explanation.
  - Structured “Buffett-style evaluation” fields.

### Tasks

1. **Set up vector store & ingestion**
   - Choose a vector DB:
     - **Option A (recommended):** Qdrant (Docker + `qdrant-client`)
     - **Option B:** Elasticsearch/OpenSearch with dense vector + BM25.
   - Ingest:
     - `chunk_id`, `year`, `text`, `section`
     - Pre-computed or on-the-fly embeddings.

2. **Implement retrieval**
   - Given a user question:
     - Embed question.
     - Retrieve top-k similar chunks (and optionally hybrid with keyword search).
   - Return both content and metadata.

3. **LLM reasoning layer**
   - Build a prompt template that includes:
     - User question.
     - Retrieved chunks (with year and context).
   - Instruct the LLM to:
     - Answer in plain language.
     - Quote Buffett directly where relevant.
     - Provide a short “Buffett-style principle summary”.
     - Output a machine-readable JSON block for structured evaluation, e.g.:

```json
{
  "business_quality": "High",
  "moat_strength": "Durable",
  "pricing_power": "Strong",
  "valuation_mindset": "Focus on earnings power and long-term ROE",
  "risk_view": "Avoid leverage; prioritize stability",
  "supporting_quotes": [
    "Quote 1...",
    "Quote 2..."
  ]
}
```

4. **Backend API**
   - Build a FastAPI service with an endpoint like `/ask_buffett`:
     - Input: question, optional settings.
     - Output: answer text, quotes, structured JSON, retrieved chunks info.

5. **UI layer**
   - **Option A – Streamlit**
     - Simple text input for the question.
     - Panel showing:
       - Final answer.
       - Key principles.
       - List of relevant Buffett quotes with years.
   - **Option B – Frontend + FastAPI**
     - Small React/Vue SPA calling FastAPI.

### Tech Stack (Phase 3)
- **Vector store:**
  - Qdrant (`qdrant-client`, Docker image `qdrant/qdrant`)
  - or Elasticsearch/OpenSearch (`elasticsearch` Python client)
- **Embeddings:**
  - `sentence-transformers`
  - or an LLM API embedding endpoint (OpenAI, etc.)
- **LLM & orchestration:**
  - OpenAI/Anthropic/etc. SDKs
  - Optional: `langchain`, `llama-index`, or `langgraph`
- **Backend:**
  - `FastAPI` (+ `uvicorn` for dev server)
- **UI:**
  - `streamlit` (fastest), or
  - React/Vue SPA consuming the FastAPI backend

---

## Phase 4 — Master Buffett Using Your Own Agent (Days 60–90)

**Goal:** Use the agent daily to deepen your understanding and refine both the product and your investing mindset.

### Main Outcomes
- A habit of asking the agent questions as your “Buffett mentor”.
- Improved retrieval quality and better prompts.
- Logs of questions/answers you can analyze later.
- Extended features (company comparison, daily principle, etc.).

### Tasks
1. **Daily usage routine**
   - Each day:
     - Ask 3–5 questions about investing problems, companies, or concepts.
     - Save your favorite answers and principles.
   - Example daily prompts:
     - “What does Buffett say about buying cyclical businesses?”
     - “How would Buffett think about a SaaS company with high R&D and low current profits?”

2. **Add logging & basic evaluation**
   - Log:
     - Question, retrieved chunks, final answer, JSON structure, timestamp.
   - Review logs weekly:
     - Are the quotes relevant?
     - Is the reasoning aligned with your understanding of Buffett?

3. **Iterate on retrieval & prompts**
   - Tune:
     - `top_k`, chunk sizes, filter by year/section where needed.
   - Improve prompts:
     - Add explicit instructions like “use at least 2 quotes from different decades”.
     - Ask for more structured outputs where useful.

4. **Add advanced features (optional)**
   - Company evaluator:
     - Input: basic metrics of a company.
     - Output: “Buffett-style evaluation” using your JSON schema.
   - Comparison mode:
     - Compare two companies or two investment ideas.
   - “Daily Buffett Principle”:
     - Randomly sample a chunk, summarize principle, show original quote.

### Tech Stack (Phase 4)
- **Same as Phase 3**, plus:
  - Logging:
    - Simple CSV/Parquet logging, or
    - `sqlite` + `sqlalchemy`
    - or tools like `langsmith`, `langfuse` for traces.
  - Evaluation:
    - `pandas` + `matplotlib/plotly` for analyzing interactions.

---

## Key Tech Stack Summary Table

| Phase | Main Goal                                    | Key Tech Stack                                                                                   |
|-------|----------------------------------------------|--------------------------------------------------------------------------------------------------|
| 1     | Build “Buffett Corpus”                       | Python, `requests`, `beautifulsoup4`, `pdfplumber`/`pymupdf`, `pandas`, `re`, CSV/Parquet       |
| 2     | Analyze patterns & learn mindset             | Python, `pandas`, `sentence-transformers`, `torch`, `scikit-learn`, `umap-learn`, `matplotlib`/`plotly` |
| 3     | Build Buffett Investing Agent MVP (RAG + UI) | Python, Qdrant/Elasticsearch, `sentence-transformers` or API embeddings, FastAPI, Streamlit/React, OpenAI/Anthropic SDKs, optional `langchain`/`llama-index`/`langgraph` |
| 4     | Master Buffett using the agent               | Same as Phase 3 + logging (`csv`/Parquet, `sqlite`, `sqlalchemy`, or `langsmith`/`langfuse`), `pandas` for analysis |

---

## How to Use This Document

- Treat this as your **project README + learning roadmap**.
- Copy it into your repo as `README.md`.
- At the start of each week, pick the tasks for the current phase and turn them into a mini-sprint.
- As you implement, add:
  - links to notebooks,
  - code files,
  - and personal notes under each phase.

This way, by the end of ~90 days, you will have:
1. A deep, data-driven understanding of Buffett’s investing mindset.
2. A working Buffett Investing Agent MVP you can demo, extend, or even productize.

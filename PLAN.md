# Plan: Gemini Multimodal Embeddings + Pinecone

## Model choice
Use **`gemini-embedding-2-preview`** — supports text, images (PNG/JPEG, up to 6/request), and video (MP4/MOV, up to 120s, sampled at 32 frames). Output dimensions: 768 (balanced), 1536, or 3072.

---

## Project structure
```
google_vector_embedding_demo/
├── .env                    # your real keys (gitignored)
├── .env.example
├── .gitignore
├── .venv/                  # virtual environment (gitignored)
├── requirements.txt
├── data/                   # drop your source files here (gitignored)
│   ├── text/               # .txt, .md files
│   ├── images/             # .png, .jpg files
│   ├── video/              # .mp4, .mov files
│   └── pdfs/               # .pdf files
├── ingest/
│   ├── ingest_text.py      # embed & upsert plain text / markdown
│   ├── ingest_images.py    # embed & upsert image files
│   ├── ingest_video.py     # embed & upsert video files
│   └── ingest_pdfs.py      # embed & upsert PDF documents (6-page chunks)
├── search.py               # query the Pinecone index
└── setup_index.py          # create/configure Pinecone index
```

---

## Phase 1 — Setup
1. **Virtual environment** — create and activate before installing anything:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate   # Linux/macOS
   # .venv\Scripts\activate    # Windows
   pip install -r requirements.txt
   ```
   `.venv/` is gitignored. All scripts must be run with this env active.
2. `setup_index.py` — create a Pinecone serverless index with dimension `768` (or 3072 for max fidelity), cosine metric
3. `requirements.txt` — `google-genai`, `pinecone`, `python-dotenv`, `openai` (for OpenRouter)

## Data directory
Drop source files into `data/` before running any ingest script:
- `data/text/` — `.txt` and `.md` files
- `data/images/` — `.png` and `.jpg` files
- `data/video/` — `.mp4` and `.mov` files (max 120s per clip)
- `data/pdfs/` — `.pdf` files (max 6 pages per embed call; multi-page PDFs are split into 6-page chunks)

`data/` is gitignored so large media files are never committed.

## Phase 2 — Ingestion scripts

Each ingestor follows the same pattern:
1. Load file(s) from a local folder
2. Call `client.models.embed_content(model="gemini-embedding-2-preview", contents=...)`
3. Attach metadata (`source`, `type`, `filename`, `timestamp`)
4. Upsert vector + metadata to Pinecone

**Text** (`ingest_text.py`): reads `.txt`/`.md` files, chunks long docs, embeds each chunk.

**Images** (`ingest_images.py`): reads PNG/JPEG, batches up to 6 per embed call, stores filename + path as metadata.

**Video** (`ingest_video.py`): submits video bytes; Gemini samples 32 frames and returns a single embedding representing the clip.

**PDFs** (`ingest_pdfs.py`): reads `.pdf` files; splits into 6-page chunks (API limit), embeds each chunk, stores page range in metadata.

## Phase 3 — Search
`search.py` — takes a text query (or image path), embeds it with the same model, queries Pinecone top-K, returns matches with metadata. Can use OpenRouter + a vision-capable LLM to generate a natural-language answer grounded in the retrieved results (RAG).

---

## Key decisions to confirm before building
1. **Dimension** — 768 (faster/cheaper) or 3072 (higher fidelity)?
2. **Pinecone tier** — serverless (pay-per-query) or pod-based?
3. **Storage location for media files** — local folder, GCS bucket, or other?
4. **RAG response layer** — do you want `search.py` to call OpenRouter to answer questions, or just return raw matches?

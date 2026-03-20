import os
import uuid
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pinecone import Pinecone

load_dotenv()

EMBED_MODEL = "gemini-embedding-2-preview"
EMBED_DIM = 768
CHUNK_SIZE = 1500       # characters per chunk
CHUNK_OVERLAP = 200     # overlap between consecutive chunks
BATCH_SIZE = 20         # texts per embed API call


def chunk_text(text: str) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start : start + CHUNK_SIZE])
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def ingest_text():
    gemini = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index = pc.Index(os.environ["PINECONE_INDEX_NAME"])

    data_dir = Path("data/text")
    files = list(data_dir.glob("*.txt")) + list(data_dir.glob("*.md"))

    if not files:
        print("No .txt or .md files found in data/text/")
        return

    all_chunks = []
    for fpath in files:
        text = fpath.read_text(encoding="utf-8")
        for i, chunk in enumerate(chunk_text(text)):
            all_chunks.append({
                "id": str(uuid.uuid4()),
                "text": chunk,
                "metadata": {
                    "type": "text",
                    "filename": fpath.name,
                    "chunk_index": i,
                    "text": chunk,  # stored for RAG retrieval
                },
            })

    for i in range(0, len(all_chunks), BATCH_SIZE):
        batch = all_chunks[i : i + BATCH_SIZE]
        result = gemini.models.embed_content(
            model=EMBED_MODEL,
            contents=[c["text"] for c in batch],
            config=types.EmbedContentConfig(output_dimensionality=EMBED_DIM),
        )
        vectors = [
            {"id": c["id"], "values": emb.values, "metadata": c["metadata"]}
            for c, emb in zip(batch, result.embeddings)
        ]
        index.upsert(vectors=vectors)
        print(f"Upserted {len(vectors)} chunks (batch {i // BATCH_SIZE + 1})")

    print(f"Done. Total chunks ingested: {len(all_chunks)}")


if __name__ == "__main__":
    ingest_text()

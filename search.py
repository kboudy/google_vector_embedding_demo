import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pinecone import Pinecone
from openai import OpenAI

load_dotenv()

EMBED_MODEL = "gemini-embedding-2-preview"
EMBED_DIM = 768
TOP_K = 5
RAG_MODEL = "google/gemini-2.0-flash-001"

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def embed_query(gemini_client, query: str) -> list[float]:
    path = Path(query)
    if path.exists() and path.suffix.lower() in IMAGE_EXTENSIONS:
        mime_type = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
        part = types.Part.from_bytes(data=path.read_bytes(), mime_type=mime_type)
        content = types.Content(role="user", parts=[part])
        result = gemini_client.models.embed_content(
            model=EMBED_MODEL,
            contents=content,
            config=types.EmbedContentConfig(output_dimensionality=EMBED_DIM),
        )
    else:
        result = gemini_client.models.embed_content(
            model=EMBED_MODEL,
            contents=query,
            config=types.EmbedContentConfig(output_dimensionality=EMBED_DIM),
        )
    return result.embeddings[0].values


def search(query: str, top_k: int = TOP_K) -> list:
    gemini = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index = pc.Index(os.environ["PINECONE_INDEX_NAME"])

    vector = embed_query(gemini, query)
    results = index.query(vector=vector, top_k=top_k, include_metadata=True)
    matches = results.matches

    print(f"\nTop {len(matches)} results for: {query!r}\n")
    for i, match in enumerate(matches, 1):
        meta = match.metadata
        label = meta.get("filename", match.id)
        print(f"  {i}. [{meta.get('type', '?')}] {label}  (score={match.score:.4f})")

    # RAG: synthesize an answer from retrieved text chunks
    text_chunks = [m for m in matches if m.metadata.get("type") == "text"]
    if text_chunks:
        context = "\n\n".join(
            f"[{m.metadata['filename']}]\n{m.metadata.get('text', '')}"
            for m in text_chunks
        )
        or_client = OpenAI(
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url="https://openrouter.ai/api/v1",
        )
        response = or_client.chat.completions.create(
            model=RAG_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "Answer the user's question using only the provided context. "
                               "If the context doesn't contain the answer, say so.",
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {query}",
                },
            ],
        )
        print("\nRAG answer:")
        print(response.choices[0].message.content)

    return matches


if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else input("Query: ")
    search(query)

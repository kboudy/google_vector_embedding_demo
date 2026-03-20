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
PAGES_PER_CHUNK = 6  # Gemini API limit per embed call


def chunk_pdf_bytes(pdf_bytes: bytes, pages_per_chunk: int = PAGES_PER_CHUNK) -> list[tuple[bytes, int, int]]:
    """
    Split a PDF into chunks of up to `pages_per_chunk` pages.
    Returns list of (chunk_bytes, start_page, end_page) tuples.
    Requires pypdf for page splitting.
    """
    try:
        from pypdf import PdfReader, PdfWriter
        import io
    except ImportError:
        raise ImportError("Install pypdf to split large PDFs: pip install pypdf")

    reader = PdfReader(io.BytesIO(pdf_bytes))
    total_pages = len(reader.pages)
    chunks = []

    for start in range(0, total_pages, pages_per_chunk):
        end = min(start + pages_per_chunk, total_pages)
        writer = PdfWriter()
        text_parts = []
        for page_num in range(start, end):
            writer.add_page(reader.pages[page_num])
            text_parts.append(reader.pages[page_num].extract_text() or "")
        buf = io.BytesIO()
        writer.write(buf)
        chunks.append((buf.getvalue(), start + 1, end, "\n\n".join(text_parts)))

    return chunks


def ingest_pdfs():
    gemini = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index = pc.Index(os.environ["PINECONE_INDEX_NAME"])

    data_dir = Path("data/pdfs")
    files = list(data_dir.glob("*.pdf"))

    if not files:
        print("No PDF files found in data/pdfs/")
        return

    total_chunks = 0
    for fpath in files:
        pdf_bytes = fpath.read_bytes()
        chunks = chunk_pdf_bytes(pdf_bytes)

        for chunk_bytes, page_start, page_end, text in chunks:
            part = types.Part.from_bytes(data=chunk_bytes, mime_type="application/pdf")
            content = types.Content(role="user", parts=[part])
            result = gemini.models.embed_content(
                model=EMBED_MODEL,
                contents=content,
                config=types.EmbedContentConfig(output_dimensionality=EMBED_DIM),
            )

            index.upsert(vectors=[{
                "id": str(uuid.uuid4()),
                "values": result.embeddings[0].values,
                "metadata": {
                    "type": "pdf",
                    "filename": fpath.name,
                    "path": str(fpath),
                    "page_start": page_start,
                    "page_end": page_end,
                    "text": text,
                },
            }])
            print(f"Upserted {fpath.name} pages {page_start}–{page_end}")
            total_chunks += 1

    print(f"Done. Total PDF chunks ingested: {total_chunks}")


if __name__ == "__main__":
    ingest_pdfs()

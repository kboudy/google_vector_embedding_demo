import os
import uuid
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pinecone import Pinecone

load_dotenv()

EMBED_MODEL = "gemini-embedding-2-preview"
DESCRIBE_MODEL = "gemini-2.0-flash"
EMBED_DIM = 768

MIME_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
}


def ingest_images():
    gemini = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index = pc.Index(os.environ["PINECONE_INDEX_NAME"])

    data_dir = Path("data/images")
    files = [f for f in data_dir.iterdir() if f.suffix.lower() in MIME_TYPES]

    if not files:
        print("No image files found in data/images/")
        return

    for fpath in files:
        mime_type = MIME_TYPES[fpath.suffix.lower()]
        image_bytes = fpath.read_bytes()

        sidecar = fpath.with_suffix(".txt")
        if sidecar.exists():
            description = sidecar.read_text(encoding="utf-8").strip()
            print(f"  Using sidecar: {sidecar.name}")
        else:
            part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
            desc_content = types.Content(role="user", parts=[
                part,
                types.Part(text="Describe this image in detail: what is visible, who or what is in it, the setting, and any notable details."),
            ])
            description_result = gemini.models.generate_content(
                model=DESCRIBE_MODEL,
                contents=desc_content,
            )
            description = description_result.text
        print(f"  Description: {description[:120]}...")

        base_metadata = {
            "type": "image",
            "filename": fpath.name,
            "path": str(fpath),
            "description": description,
        }

        # Visual vector — media bytes for image similarity search
        visual_result = gemini.models.embed_content(
            model=EMBED_MODEL,
            contents=types.Content(role="user", parts=[
                types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
            ]),
            config=types.EmbedContentConfig(output_dimensionality=EMBED_DIM),
        )

        # Text vector — description for chat/RAG search
        text_result = gemini.models.embed_content(
            model=EMBED_MODEL,
            contents=description,
            config=types.EmbedContentConfig(output_dimensionality=EMBED_DIM),
        )

        index.upsert(vectors=[
            {"id": str(uuid.uuid4()), "values": visual_result.embeddings[0].values,
             "metadata": {**base_metadata, "embed_type": "visual"}},
            {"id": str(uuid.uuid4()), "values": text_result.embeddings[0].values,
             "metadata": {**base_metadata, "embed_type": "text"}},
        ])
        print(f"Upserted image (x2 vectors): {fpath.name}")

    print(f"Done. Total images ingested: {len(files)}")


if __name__ == "__main__":
    ingest_images()

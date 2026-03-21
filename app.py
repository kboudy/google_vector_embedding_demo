import os
from flask import Flask, request, jsonify, render_template, send_from_directory
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pinecone import Pinecone
from openai import OpenAI

load_dotenv()

app = Flask(__name__)

EMBED_MODEL = "gemini-embedding-2-preview"
EMBED_DIM = 768
CHAT_MODEL = "anthropic/claude-sonnet-4-6"
TOP_K = 5
MIN_SCORE = 0.5  # text query threshold
MIN_SCORE_IMAGE = 0.8  # image query threshold (cross-modal scores run lower)
DATA_DIR = os.path.abspath("data")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/media/<path:filepath>")
def serve_media(filepath):
    return send_from_directory(DATA_DIR, filepath)


@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    message = data.get("message", "").strip()
    history = data.get("history", [])

    if not message:
        return jsonify({"error": "Empty message"}), 400

    gemini = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index = pc.Index(os.environ["PINECONE_INDEX_NAME"])
    or_client = OpenAI(
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1",
    )

    # Embed query
    embed_result = gemini.models.embed_content(
        model=EMBED_MODEL,
        contents=message,
        config=types.EmbedContentConfig(output_dimensionality=EMBED_DIM),
    )
    vector = embed_result.embeddings[0].values

    # Search Pinecone
    search_results = index.query(
        vector=vector, top_k=TOP_K, include_metadata=True,
        filter={"embed_type": {"$eq": "text"}},
    )
    matches = [m for m in search_results.matches if m.score >= MIN_SCORE]

    # Build context and source list
    sources = []
    context_parts = []
    for match in matches:
        meta = match.metadata or {}
        raw_path = meta.get("path", "")
        rel = raw_path.replace("\\", "/")
        url = "/media/" + rel[len("data/") :] if rel.startswith("data/") else None
        sources.append(
            {
                "filename": meta.get("filename", "unknown"),
                "type": meta.get("type", "unknown"),
                "score": round(match.score, 3),
                "page_start": meta.get("page_start"),
                "page_end": meta.get("page_end"),
                "url": url,
            }
        )
        media_type = meta.get("type", "unknown")
        filename = meta.get("filename", "unknown")
        if media_type == "text" and meta.get("text"):
            context_parts.append(f"[text: {filename}]\n{meta['text']}")
        elif media_type == "image":
            desc = meta.get("description", "No description available.")
            context_parts.append(f"[image: {filename}]\n{desc}")
        elif media_type == "video":
            desc = meta.get("description", "No description available.")
            context_parts.append(f"[video: {filename}]\n{desc}")
        elif media_type == "pdf":
            pages = (
                f" pages {int(meta['page_start'])}–{int(meta['page_end'])}"
                if meta.get("page_start")
                else ""
            )
            text = meta.get("text", "").strip()
            context_parts.append(
                f"[pdf: {filename}{pages}]\n{text}"
                if text
                else f"[pdf: {filename}{pages}] — no text extracted"
            )

    context = "\n\n---\n\n".join(context_parts)

    system_prompt = (
        "You are a helpful assistant with access to a multimodal knowledge base "
        "containing text, images, video, and PDF documents. "
        "Answer questions using the provided context when relevant. "
        "If the context doesn't contain enough information, say so clearly."
    )
    if context:
        system_prompt += f"\n\nRelevant context from the knowledge base:\n\n{context}"

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history)
    messages.append({"role": "user", "content": message})

    response = or_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
    )

    return jsonify(
        {
            "response": response.choices[0].message.content,
            "sources": sources,
        }
    )


@app.route("/search-image", methods=["POST"])
def search_image():
    file = request.files.get("image")
    if not file:
        return jsonify({"error": "No image provided"}), 400

    mime_type = file.mimetype or "image/jpeg"
    image_bytes = file.read()

    gemini = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index = pc.Index(os.environ["PINECONE_INDEX_NAME"])

    embed_result = gemini.models.embed_content(
        model=EMBED_MODEL,
        contents=types.Content(
            role="user",
            parts=[
                types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
            ],
        ),
        config=types.EmbedContentConfig(output_dimensionality=EMBED_DIM),
    )
    vector = embed_result.embeddings[0].values

    search_results = index.query(
        vector=vector, top_k=TOP_K, include_metadata=True,
        filter={"embed_type": {"$eq": "visual"}},
    )
    matches = [m for m in search_results.matches if m.score >= MIN_SCORE_IMAGE]

    sources = []
    for match in matches:
        meta = match.metadata or {}
        raw_path = meta.get("path", "")
        rel = raw_path.replace("\\", "/")
        url = "/media/" + rel[len("data/") :] if rel.startswith("data/") else None
        sources.append(
            {
                "filename": meta.get("filename", "unknown"),
                "type": meta.get("type", "unknown"),
                "score": round(match.score, 3),
                "page_start": meta.get("page_start"),
                "page_end": meta.get("page_end"),
                "url": url,
            }
        )

    return jsonify({"sources": sources})


if __name__ == "__main__":
    app.run(debug=True, port=5000)

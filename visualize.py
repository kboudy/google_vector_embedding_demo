import os
import webbrowser
import numpy as np
from dotenv import load_dotenv
from pinecone import Pinecone
from sklearn.decomposition import PCA
import plotly.graph_objects as go

load_dotenv()

COLORS = {
    "text":  "#4a9eff",
    "image": "#e8b84b",
    "video": "#a78bfa",
    "pdf":   "#f87171",
}


def visualize():
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index = pc.Index(os.environ["PINECONE_INDEX_NAME"])

    # List all vector IDs
    all_ids = []
    for id_batch in index.list():
        all_ids.extend(id_batch)

    if not all_ids:
        print("No vectors in index.")
        return

    print(f"Fetching {len(all_ids)} vectors...")

    # Fetch in batches of 100 (API limit)
    vectors = {}
    for i in range(0, len(all_ids), 100):
        result = index.fetch(ids=all_ids[i:i + 100])
        vectors.update(result.vectors)

    ids = list(vectors.keys())
    embeddings = np.array([vectors[vid].values for vid in ids])
    metadatas = [vectors[vid].metadata or {} for vid in ids]

    if len(ids) < 3:
        print("Need at least 3 vectors to visualize.")
        return

    # Reduce to 2D with PCA
    print("Reducing dimensions...")
    reduced = PCA(n_components=2).fit_transform(embeddings)

    # Group points by type
    groups: dict[str, dict] = {}
    for i, meta in enumerate(metadatas):
        t = meta.get("type", "unknown")
        if t not in groups:
            groups[t] = {"x": [], "y": [], "labels": []}

        filename = meta.get("filename", ids[i][:8])
        suffix = ""
        if meta.get("chunk_index") is not None:
            suffix = f" (chunk {int(meta['chunk_index']) + 1})"
        elif meta.get("page_start"):
            suffix = f" p.{int(meta['page_start'])}–{int(meta['page_end'])}"

        groups[t]["x"].append(reduced[i, 0])
        groups[t]["y"].append(reduced[i, 1])
        groups[t]["labels"].append(f"{filename}{suffix}")

    traces = []
    for t, data in groups.items():
        color = COLORS.get(t, "#94a3b8")
        traces.append(go.Scatter(
            x=data["x"],
            y=data["y"],
            mode="markers+text",
            name=t,
            text=data["labels"],
            textposition="top center",
            textfont=dict(size=9, color=color),
            marker=dict(
                size=11,
                color=color,
                line=dict(width=1, color="rgba(255,255,255,0.15)"),
            ),
            hovertemplate="<b>%{text}</b><br>" + t + "<extra></extra>",
        ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=dict(
            text=f"Vector Space  ·  {len(ids)} embeddings  ·  PCA 2D projection",
            font=dict(size=13, color="#c6c3bb"),
            x=0.02,
        ),
        paper_bgcolor="#07070a",
        plot_bgcolor="#0e0e12",
        font=dict(family="monospace", color="#c6c3bb", size=11),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            bordercolor="#1a1a20",
            borderwidth=1,
            title=dict(text="type", font=dict(size=10)),
        ),
        xaxis=dict(title="PC1", gridcolor="#1a1a20", zerolinecolor="#2a2a30"),
        yaxis=dict(title="PC2", gridcolor="#1a1a20", zerolinecolor="#2a2a30"),
        hoverlabel=dict(bgcolor="#0e0e12", bordercolor="#1a1a20", font_color="#c6c3bb"),
    )

    out = "visualize.html"
    fig.write_html(out)
    print(f"Saved → {out}")
    webbrowser.open(out)


if __name__ == "__main__":
    visualize()

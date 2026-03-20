import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

DIMENSION = 768
METRIC = "cosine"


def setup():
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index_name = os.environ["PINECONE_INDEX_NAME"]

    existing = [idx.name for idx in pc.list_indexes()]
    if index_name in existing:
        print(f"Index '{index_name}' already exists.")
        return

    pc.create_index(
        name=index_name,
        dimension=DIMENSION,
        metric=METRIC,
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    print(f"Created index '{index_name}' ({DIMENSION}d, {METRIC})")


if __name__ == "__main__":
    setup()

import os
from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()


def clear_index():
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index_name = os.environ["PINECONE_INDEX_NAME"]
    index = pc.Index(index_name)

    stats = index.describe_index_stats()
    total = stats.total_vector_count

    if total == 0:
        print("Index is already empty.")
        return

    confirm = input(f"Delete all {total} vectors from '{index_name}'? [y/N] ")
    if confirm.strip().lower() != "y":
        print("Aborted.")
        return

    index.delete(delete_all=True)
    print(f"Cleared {total} vectors from '{index_name}'.")


if __name__ == "__main__":
    clear_index()

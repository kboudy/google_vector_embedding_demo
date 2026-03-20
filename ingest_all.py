from ingest.ingest_text import ingest_text
from ingest.ingest_images import ingest_images
from ingest.ingest_video import ingest_video
from ingest.ingest_pdfs import ingest_pdfs

if __name__ == "__main__":
    print("=== Text ===")
    ingest_text()
    print("\n=== Images ===")
    ingest_images()
    print("\n=== Video ===")
    ingest_video()
    print("\n=== PDFs ===")
    ingest_pdfs()
    print("\nAll done.")

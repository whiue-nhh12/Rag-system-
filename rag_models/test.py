from langchain_core.documents import Document

from .chunking import TextChunker
from .ingestion import PDFLoader


def main() -> None:
    url = "app/rag_models/datasets/01_SE intro1.pdf"

    loader = PDFLoader(url)
    doc = loader.load()

    chunker = TextChunker(chunk_size=180, chunk_overlap=40)
    chunks = chunker.chunk(doc)

    print(f"Số chunk tạo ra: {len(chunks)}")
    print("=" * 60)

    for index, chunk in enumerate(chunks, start=1):
        print(f"Chunk {index}")
        print("-" * 60)
        print(chunk.page_content.strip())
        print("Metadata:", chunk.metadata)
        print()


if __name__ == "__main__":
    main()

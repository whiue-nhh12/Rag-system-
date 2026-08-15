#!/usr/bin/env python
"""Test script for heading-based recursive text splitting."""

from langchain_core.documents import Document
from rag_models.chunking.recursive_structural_chunker import RecursiveStructuralChunker

# Create test document with multiple heading levels
test_text = """CHƯƠNG 1: Giới thiệu về Pháp luật
Nội dung chương 1, phần giới thiệu

1.1 Khái niệm cơ bản
Đây là phần giới thiệu khái niệm cơ bản về pháp luật. Pháp luật là tập hợp các quy tắc được nhà nước ban hành.

1.1.1 Định nghĩa
Định nghĩa chi tiết về pháp luật trong hệ thống pháp luật.

1.1.2 Phân loại
Các loại pháp luật khác nhau:
- Pháp luật dân sự
- Pháp luật hình sự
- Pháp luật hành chính
- Pháp luật kinh doanh

1.2 Nguồn gốc và phát triển
Lịch sử phát triển của pháp luật.

CHƯƠNG 2: Các nguyên tắc cơ bản
Nội dung chương 2 về nguyên tắc cơ bản của pháp luật.

2.1 Nguyên tắc bình đẳng
Tất cả mọi người đều bình đẳng trước pháp luật."""

# Initialize chunker
chunker = RecursiveStructuralChunker(max_chunk_size=300, chunk_overlap=50)

# Create Document objects
documents = [Document(page_content=test_text, metadata={"source": "test.txt", "page": 1})]

# Perform chunking
print("=" * 80)
print("Testing Heading-Based Recursive Text Splitting")
print("=" * 80)
print(f"\nOriginal text length: {len(test_text)} characters\n")

chunks = chunker.chunk(documents)

print(f"Total chunks created: {len(chunks)}\n")
print("-" * 80)

for i, chunk in enumerate(chunks, 1):
    print(f"\n📄 CHUNK {i}:")
    print(f"   Length: {len(chunk.page_content)} chars")
    print(f"   Hierarchy: {chunk.metadata.get('hierarchy_path', 'N/A')}")
    print(f"   Chapter: {chunk.metadata.get('chapter', 'N/A')}")
    print(f"   Section: {chunk.metadata.get('section', 'N/A')}")
    print(f"   Subsection: {chunk.metadata.get('subsection', 'N/A')}")
    print(f"   Content preview:")
    content_preview = chunk.page_content[:150].replace("\n", " ")
    print(f"   {content_preview}...")
    print(f"   {'-' * 76}")

print("\n" + "=" * 80)
print("✅ Test completed successfully!")
print("=" * 80)

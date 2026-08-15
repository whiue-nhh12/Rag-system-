import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .base import BaseChunking

title_dictionary = {
    "pháp luật" :"Pháp luật",
    "nhà nước" :"Nhà nước",
    "vi phạm pháp luật" :"Vi phạm",
    "trách nhiệm pháp lí" :"Trách nhiệm",
    "hiến pháp" :"Hiến pháp",
    "hành chính" :"Hành chính",
    "dân sự" :"Dân sự",
    "hình sự" :"Hình sự",
    "lao động" :"Lao động",
    "kinh doanh" :"Kinh doanh",
}

@dataclass
class StructuralBlock:
    text: str
    chapter: str = ""
    section: str = ""
    subsection: str = ""
    title : Optional[List[str]] = None
    source: Optional[str] = None
    start_page: Optional[int] = None
    end_page: Optional[int] = None
    source_doc_index: Optional[int] = None
    hierarchy_path: str = field(init=False, default="")

    @classmethod
    def from_document(
        cls,
        document: Document,
        source_doc_index: Optional[int] = None,
    ) -> "StructuralBlock":
        metadata = document.metadata or {}
        return cls(
            text=(document.page_content or "").strip(),
            chapter=str(metadata.get("chapter", "")) if metadata.get("chapter") is not None else "",
            section=str(metadata.get("section", "")) if metadata.get("section") is not None else "",
            subsection=str(metadata.get("subsection", "")) if metadata.get("subsection") is not None else "",
            source=metadata.get("source"),
            start_page=metadata.get("start_page") or metadata.get("page") or metadata.get("page_number") or metadata.get("page_num"),
            end_page=metadata.get("end_page") or metadata.get("page") or metadata.get("page_number") or metadata.get("page_num"),
            source_doc_index=source_doc_index if source_doc_index is not None else metadata.get("source_doc_index"),
        )

    def __post_init__(self) -> None:
        parts = [p for p in (self.chapter, self.section, self.subsection) if p]
        self.hierarchy_path = " > ".join(parts) if parts else "root"

logger = logging.getLogger(__name__)


class RecursiveStructuralChunker(BaseChunking):
    """Chunk structured documents by headings in a recursive, hierarchy-aware way."""

    def __init__(
        self,
        max_chunk_size: int = 800,
        chunk_overlap: int = 100,
    ):
        super().__init__(chunk_size=max_chunk_size, chunk_overlap=chunk_overlap)
        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = chunk_overlap
    
        # Separators for recursive splitting - ORDERED BY PRIORITY (highest first)
        # Headings have highest priority to maintain document structure
        self.split_separators = [
            ("heading_chapter", r"^CHƯƠNG\s+\d+.*"),       # Highest priority
            ("heading_section", r"^\d+\.\d+\s+.*"),         # Mid-high priority
            ("heading_subsection", r"^\d+\.\d+\.\d+\.?\s+.*"), # Mid priority
            ("bullet", r"^\s*[-*•]\s+"),                    # Lower prior                                  # Last resort
        ]

        self.fallback_splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", ". ", " ", ""],
        )
    
    def _calculate_text_length(self, text: str) -> int:
        """Calculate length of text for chunk size checks."""
        return len(text)

    def chunk(self, documents: List[Document]) -> List[Document]:
        """Split a list of page-like Documents into hierarchy-aware chunks."""
        logger.info(
            "Chunking %s document(s) with RecursiveStructuralChunker",
            len(documents),
        )
        all_chunks = self.build_flow(documents)

        logger.info(
            "Completed chunking: %s document(s) -> %s chunk(s)",
            len(documents),
            len(all_chunks),
        )
        return all_chunks

    def build_flow(
        self,
        documents: List[Document],
    ) -> List[Document]:
        """Assemble blocks and split them into chunks, exposing the processing flow."""
        logger.info(
            "Building structural chunking flow for %s document(s)",
            len(documents),
        )
        structural_blocks = self._assemble_structural_blocks(documents)

        all_chunks: List[Document] = []
        global_chunk_counter = 0

        for block in structural_blocks:
            block_chunks = self._split_structural_block(block, global_chunk_counter)
            all_chunks.extend(block_chunks)
            global_chunk_counter += len(block_chunks)

        return all_chunks

    def _assemble_structural_blocks(self, documents: List[Document]) -> List[StructuralBlock]:
        blocks: List[StructuralBlock] = []
        current_buffer: List[str] = []
        current_metadata = {
            "title": [],
            "chapter": "",
            "section": "",
            "subsection": "",
        }
        current_source: Optional[str] = None
        current_start_page: Optional[int] = None
        current_end_page: Optional[int] = None
        current_start_doc_idx: Optional[int] = None

        for doc_idx, doc in enumerate(documents):
            content = doc.page_content or ""
            if not content.strip():
                continue

            source = doc.metadata.get("source") or current_source
            if source is not None:
                current_source = source

            page_num = self._get_page_number(doc, doc_idx)
            if current_start_page is None:
                current_start_page = page_num
            current_end_page = page_num
            if current_start_doc_idx is None:
                current_start_doc_idx = doc_idx

            for raw_line in content.splitlines():
                line = raw_line.strip()
                if not line:
                    if current_buffer:
                        current_buffer.append("")
                    continue
                # Only detect structural headings here (chapter, section, subsection).
                heading_level = self._detect_heading_level(line)
                if heading_level:
                    if current_buffer:
                        blocks.append(
                            self._build_block(
                                current_buffer,
                                current_metadata,
                                current_source,
                                current_start_page,
                                current_end_page,
                                current_start_doc_idx,
                            )
                        )
                        current_buffer = []

                    if heading_level == "chapter":
                        line = line.lower()
                        topics = []
                        for key,value in title_dictionary.items():
                            if key in line:
                                topics.append(value)

                        current_metadata = {
                            "chapter": line,
                            "section": "",
                            "subsection": "",
                            "title": topics
                        }
                    elif heading_level == "section":
                        current_metadata["section"] = line
                        topics = []
                        for key,value in title_dictionary.items():
                            if key in line.lower():
                                topics.append(value)
                        if topics:
                            metadatatopics = current_metadata.get("title", [])
                            current_metadata["title"] = list(set(metadatatopics) & set(topics))
                        current_metadata["subsection"] = ""
                    elif heading_level == "subsection":
                        current_metadata["subsection"] = line
                        topics = []
                        for key,value in title_dictionary.items():
                            if key in line.lower():
                                topics.append(value)
                        if topics:
                            metadatatopics = current_metadata.get("title", [])
                            current_metadata["title"] = list(set(metadatatopics) & set(topics))

                    current_buffer = [line]
                    current_start_page = page_num
                    current_end_page = page_num
                    current_start_doc_idx = doc_idx
                    continue

                if not current_buffer:
                    current_buffer = []
                current_buffer.append(raw_line.rstrip())
                current_end_page = page_num

        if current_buffer:
            blocks.append(
                self._build_block(
                    current_buffer,
                    current_metadata,
                    current_source,
                    current_start_page,
                    current_end_page,
                    current_start_doc_idx,
                )
            )

        return blocks

    def _build_block(
        self,
        buffer_lines: List[str],
        metadata: Dict[str, str],
        source: Optional[str],
        start_page: Optional[int],
        end_page: Optional[int],
        start_doc_idx: Optional[int],
    ) -> StructuralBlock:
        text = "\n".join(line for line in buffer_lines if line is not None).strip()
        if not text:
            text = ""
        return StructuralBlock(
            text=text,
            chapter=metadata.get("chapter", ""),
            section=metadata.get("section", ""),
            subsection=metadata.get("subsection", ""),
            title=metadata.get("title", []),
            source=source,
            start_page=start_page,
            end_page=end_page,
            source_doc_index=start_doc_idx,
        )

    def _split_structural_block(
        self, block: StructuralBlock, global_chunk_counter: int
    ) -> List[Document]:
        """Split a structural block using recursive separator-based approach with heading priority."""
        block_text = str(block.text)
        if self._calculate_text_length(block_text) <= self.max_chunk_size:
            return [
                self._create_document_chunk(
                    block_text,
                    block,
                    global_chunk_counter,
                )
            ]
        
        # Use recursive splitting with separators (heading patterns have priority)
        split_texts = self._recursive_split_text(block_text, self.split_separators)
        
        # Merge splits respecting chunk size and overlap
        merged_chunks = self._merge_splits(split_texts)
        
        return [
            self._create_document_chunk(
                piece,
                block,
                global_chunk_counter + offset,
            )
            for offset, piece in enumerate(merged_chunks)
        ]

    def _recursive_split_text(
        self, text: str, separators: list
    ) -> list:
        """
        Recursively split text using separators with heading priority.
        Format: [(separator_type_name, separator_pattern), ...]
        Tries separators in order until finding one that exists in text.
        """
        final_chunks = []
        
        if not separators or not text:
            return [text] if text else []
        
        separator_type = None
        separator_pattern = None
        next_separators = []
        
        # Find the best separator that exists in the text (first match wins due to priority)
        for i, (sep_type, sep_pattern) in enumerate(separators):
            if sep_pattern == "":  # Empty separator means split by character
                separator_type = sep_type
                separator_pattern = sep_pattern
                next_separators = []
                break
            
            # Check if separator exists in text
            if self._separator_exists_in_text(text, sep_pattern):
                separator_type = sep_type
                separator_pattern = sep_pattern
                next_separators = separators[i + 1:]
                break
        
        # If no separator found, return text as-is
        if separator_pattern is None:
            return [text] if text else []
        
        # Split the text based on separator type
        if separator_type.startswith("heading_"):
            # Heading-based splitting
            splits = self._split_by_heading(text, separator_pattern)
        elif separator_type == "bullet":
            # Bullet-based splitting
            splits = self._split_by_regex(text, separator_pattern)
        elif separator_pattern == "":
            # Character-level splitting
            splits = list(text)
        else:
            # Simple string splitting (paragraph, line, sentence, word)
            splits = [s for s in text.split(separator_pattern) if s]
        
        temp = []
        
        for s in splits:
            if self._calculate_text_length(s) <= self.max_chunk_size:
                temp.append(s)
            else:
                if temp:
                    final_chunks.extend(self._merge_splits(temp))
                    temp = []
                
                if not next_separators:
                    # No more separators to try, add as-is
                    final_chunks.append(s)
                else:
                    # Recursively split with next separator
                    recursive_chunks = self._recursive_split_text(s, next_separators)
                    final_chunks.extend(recursive_chunks)
        
        if temp:
            final_chunks.extend(self._merge_splits(temp))
        
        return final_chunks

    def _separator_exists_in_text(self, text: str, separator: str) -> bool:
        """Check if a separator (string or regex pattern) exists in text."""
        if not separator:
            return False
        
        # For regex patterns (headings, bullets)
        if separator.startswith(r"^"):
            return any(re.match(separator, line) for line in text.splitlines())
        
        # For simple string separators
        return separator in text

    def _split_by_heading(self, text: str, pattern: str) -> List[str]:
        """Split text by heading pattern while preserving headers."""
        splits = []
        current_part = []
        
        for line in text.splitlines(keepends=False):
            if re.match(pattern, line):
                if current_part:
                    splits.append("\n".join(current_part))
                current_part = [line]
            else:
                current_part.append(line)
        
        if current_part:
            splits.append("\n".join(current_part))
        
        return [s for s in splits if s]

    def _split_by_regex(self, text: str, pattern: str) -> List[str]:
        """Split text using a regex pattern (for bullets) while preserving structure."""
        splits = []
        current_part = []
        
        for line in text.splitlines(keepends=False):
            if re.match(pattern, line):
                if current_part:
                    splits.append("\n".join(current_part))
                current_part = [line]
            else:
                current_part.append(line)
        
        if current_part:
            splits.append("\n".join(current_part))
        
        return [s for s in splits if s]

    def _merge_splits(self, splits: List[str]) -> List[str]:
        """
        Merge splits to reach target chunk size while respecting overlap.
        Inspired by RecursiveCharacterTextSplitter's merge_splits logic.
        """
        if not splits:
            return []
        
        if len(splits) == 1:
            return splits
        
        # Default separator is double newline for merging
        separator = "\n\n"
        merged_chunks = []
        current_group = []
        current_length = 0
        
        for s in splits:
            s_length = self._calculate_text_length(s)
            sep_length = self._calculate_text_length(separator) if current_group else 0
            
            # Check if adding this split would exceed max chunk size
            if current_group and current_length + sep_length + s_length > self.max_chunk_size:
                # Merge current group and start new one
                merged = separator.join(current_group)
                if merged:
                    merged_chunks.append(merged)
                current_group = [s]
                current_length = s_length
            else:
                # Add to current group
                current_group.append(s)
                current_length += sep_length + s_length
        
        # Don't forget the last group
        if current_group:
            merged = separator.join(current_group)
            if merged:
                merged_chunks.append(merged)
        
        return merged_chunks

    def _detect_heading_level(self, line: str) -> Optional[str]:
        candidate = line.strip()
        if re.match(self.heading_patterns["chapter"], candidate):
            return "chapter"
        if re.match(self.heading_patterns["section"], candidate):
            return "section"
        if re.match(self.heading_patterns["subsection"], candidate):
            return "subsection"
        return None

    def _create_document_chunk(
        self,
        text: str,
        parent_block: StructuralBlock,
        chunk_index: int,
    ) -> Document:
        metadata = {
            "source": parent_block.source,
            "start_page": parent_block.start_page,
            "end_page": parent_block.end_page,
            "chapter": parent_block.chapter or "",
            "section": parent_block.section or "",
            "subsection": parent_block.subsection or "",
            "hierarchy_path": parent_block.hierarchy_path,
            "chunk_index": chunk_index,
            "source_doc_index": parent_block.source_doc_index,
        }
        return Document(page_content=text, metadata=metadata)

    def _build_hierarchy_path(self, chapter: str, section: str, subsection: str) -> str:
        parts = [part for part in [chapter, section, subsection] if part]
        return " > ".join(parts) if parts else "root"

    def _get_page_number(self, doc: Document, fallback_index: int) -> int:
        for key in ("page", "page_number", "page_num"):
            value = doc.metadata.get(key)
            if isinstance(value, int):
                return value
            if isinstance(value, str) and value.isdigit():
                return int(value)
        return fallback_index + 1

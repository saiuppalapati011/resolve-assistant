"""
Document ingestion pipeline.
Converts PDFs → text chunks → Chroma vector store.
Run once (and re-run when docs change):

    cd resolve-assistant
    python -m backend.rag.ingest

Supports both PDF (via pypdf fast extraction) and pre-converted .txt / .md files.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import chromadb
from pypdf import PdfReader

from backend.config import settings
from backend.logging_config import get_logger, setup_logging
from backend.rag.embeddings import LocalEmbeddings
from html.parser import HTMLParser

logger = get_logger(__name__)

# ── HTML → text ───────────────────────────────────────────────────────────────

class TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.result = []
    def handle_data(self, data):
        text = data.strip()
        if text:
            self.result.append(text)

def html_to_text(html_path: Path) -> str:
    """Extract all text from an HTML file."""
    logger.info("Extracting text from HTML", file=str(html_path))
    parser = TextExtractor()
    content = html_path.read_text(encoding="utf-8", errors="replace")
    parser.feed(content)
    text = "\n\n".join(parser.result)
    logger.info("Extraction complete", file=html_path.name, chars=len(text))
    return text

# ── PDF → text ────────────────────────────────────────────────────────────────

def pdf_to_text(pdf_path: Path) -> str:
    """
    Extract all text from a PDF using pypdf (fast, no page rendering).
    Falls back to empty string for pages that can't be decoded.
    """
    logger.info("Extracting text from PDF", file=str(pdf_path))
    reader = PdfReader(str(pdf_path))
    total = len(reader.pages)
    logger.info("PDF loaded", pages=total)

    pages = []
    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
            if text.strip():
                pages.append(f"[Page {i + 1}]\n{text.strip()}")
        except Exception as exc:
            logger.warning("Could not extract page", page=i + 1, error=str(exc))

        # Progress log every 100 pages
        if (i + 1) % 100 == 0:
            logger.info("Extraction progress", done=i + 1, total=total)

    full_text = "\n\n".join(pages)
    logger.info("Extraction complete", file=pdf_path.name, pages_extracted=len(pages), chars=len(full_text))
    return full_text


# ── Chunking ──────────────────────────────────────────────────────────────────

_HEADING_RE = re.compile(
    r"^(#{1,4}\s.+|[A-Z][A-Za-z\s]{3,60}(?:\n[-=]{3,})?|\[Page \d+\])",
    re.MULTILINE,
)

def chunk_by_section(text: str, min_len: int = 100, max_len: int = 2000) -> list[str]:
    """
    Split text on section/heading boundaries.
    Falls back to fixed-size windows for pages without clear headings.

    Strategy:
    1. Split on ### headings (Markdown) or [Page N] markers from PDF extraction.
    2. Merge very short chunks with their predecessor.
    3. Split very long chunks at paragraph boundaries.
    """
    # Primary split on heading-like lines
    parts = _HEADING_RE.split(text)
    chunks: list[str] = []

    buffer = ""
    for part in parts:
        part = part.strip()
        if not part:
            continue
        combined = (buffer + "\n\n" + part).strip() if buffer else part
        if len(combined) >= min_len:
            if len(combined) <= max_len:
                chunks.append(combined)
                buffer = ""
            else:
                # Split oversized chunk at paragraph boundaries
                paragraphs = combined.split("\n\n")
                current = ""
                for para in paragraphs:
                    if len(current) + len(para) < max_len:
                        current = (current + "\n\n" + para).strip()
                    else:
                        if current:
                            chunks.append(current)
                        current = para
                if current:
                    chunks.append(current)
                buffer = ""
        else:
            buffer = combined

    if buffer and len(buffer) >= min_len:
        chunks.append(buffer)

    return [c for c in chunks if c.strip()]


# ── Ingestion ─────────────────────────────────────────────────────────────────

def ingest(
    pdf_dir: str | None = None,
    text_dir: str | None = None,
    collection_name: str | None = None,
    chroma_path: str | None = None,
) -> int:
    """
    Ingest all PDFs and text files into Chroma.
    Returns the total number of chunks stored.
    """
    pdf_root = Path(pdf_dir or settings.pdf_dir)
    txt_root = Path(text_dir or settings.text_dir)
    col_name = collection_name or settings.collection_name
    db_path = chroma_path or settings.chroma_path

    client = chromadb.PersistentClient(path=db_path)
    # Build into an isolated collection first. The currently published index
    # remains available if extraction or embedding fails.
    staging_name = f"{col_name}__staging"
    try:
        client.delete_collection(staging_name)
    except Exception:
        pass
    collection = client.get_or_create_collection(staging_name)
    embedder = LocalEmbeddings(settings.embeddings_model)

    all_chunks: list[str] = []
    all_ids: list[str] = []
    all_metas: list[dict] = []

    # ── PDFs ────────────────────────────────────────────────────────────────
    if pdf_root.exists():
        for pdf_path in sorted(pdf_root.glob("*.pdf")):
            text = pdf_to_text(pdf_path)
            # Also save converted text for inspection
            txt_root.mkdir(parents=True, exist_ok=True)
            out_txt = txt_root / (pdf_path.stem + ".txt")
            out_txt.write_text(text, encoding="utf-8")
            logger.info("Saved converted text", file=str(out_txt))

            chunks = chunk_by_section(text)
            logger.info("Chunked PDF", file=pdf_path.name, chunks=len(chunks))
            for i, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_ids.append(f"{pdf_path.stem}-{i}")
                all_metas.append({"source": pdf_path.name, "chunk_idx": i})
    else:
        logger.warning("PDF directory not found", path=str(pdf_root))

    # ── Text / Markdown / HTML files ───────────────────────────────────────
    if txt_root.exists():
        for txt_path in sorted(txt_root.glob("*.txt")) + sorted(txt_root.glob("*.md")) + sorted(txt_root.glob("*.html")):
            if txt_path.suffix == ".html":
                text = html_to_text(txt_path)
            else:
                text = txt_path.read_text(encoding="utf-8", errors="replace")
                
            chunks = chunk_by_section(text)
            logger.info("Chunked file", file=txt_path.name, chunks=len(chunks))
            for i, chunk in enumerate(chunks):
                cid = f"{txt_path.stem}-{txt_path.suffix.strip('.')}-{i}"
                if cid not in all_ids:
                    all_chunks.append(chunk)
                    all_ids.append(cid)
                    all_metas.append({"source": txt_path.name, "chunk_idx": i})

    if not all_chunks:
        logger.error("No documents found to ingest. Check pdf_dir and text_dir in config.")
        return 0

    # Embed and store in batches of 100
    BATCH = 100
    for start in range(0, len(all_chunks), BATCH):
        batch_chunks = all_chunks[start:start + BATCH]
        batch_ids    = all_ids[start:start + BATCH]
        batch_metas  = all_metas[start:start + BATCH]
        vectors = embedder.embed(batch_chunks)
        collection.add(
            ids=batch_ids,
            embeddings=vectors,
            documents=batch_chunks,
            metadatas=batch_metas,
        )
        logger.info("Stored batch", start=start, count=len(batch_chunks))

    total = collection.count()
    if total == 0:
        client.delete_collection(staging_name)
        logger.error("Staged ingestion produced no chunks; existing index preserved")
        return 0

    # Chroma has no collection rename. Publish only after staging succeeds;
    # keep a backup of the previous collection until the new collection is
    # fully installed so a failed build never destroys the only index.
    backup_name = f"{col_name}__previous"
    try:
        client.delete_collection(backup_name)
    except Exception:
        pass
    try:
        old = client.get_collection(col_name)
        old_data = old.get(include=["documents", "metadatas", "embeddings"])
        if old_data.get("ids"):
            backup = client.get_or_create_collection(backup_name)
            backup.add(
                ids=old_data["ids"],
                documents=old_data.get("documents"),
                metadatas=old_data.get("metadatas"),
                embeddings=old_data.get("embeddings"),
            )
        client.delete_collection(col_name)
    except Exception:
        # No previous collection is a valid first-ingestion case.
        try:
            client.delete_collection(col_name)
        except Exception:
            pass

    published = client.get_or_create_collection(col_name)
    staged = collection.get(include=["documents", "metadatas", "embeddings"])
    published.add(
        ids=staged["ids"],
        documents=staged.get("documents"),
        metadatas=staged.get("metadatas"),
        embeddings=staged.get("embeddings"),
    )
    client.delete_collection(staging_name)
    logger.info("Ingestion complete", total_chunks=total)
    return total


# ── CLI entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    setup_logging()
    parser = argparse.ArgumentParser(description="Ingest docs into the Resolve RAG store.")
    parser.add_argument("--pdf-dir",   default=None, help="Directory containing PDF docs")
    parser.add_argument("--text-dir",  default=None, help="Directory for converted text files")
    parser.add_argument("--collection", default=None, help="Chroma collection name")
    parser.add_argument("--chroma-path", default=None, help="Chroma persistence path")
    args = parser.parse_args()

    total = ingest(
        pdf_dir=args.pdf_dir,
        text_dir=args.text_dir,
        collection_name=args.collection,
        chroma_path=args.chroma_path,
    )
    if total > 0:
        print(f"\n✅ Ingested {total} chunks successfully.")
    else:
        print("\n❌ Ingestion failed — check the logs above.")
        sys.exit(1)

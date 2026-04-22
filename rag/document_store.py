# rag/document_store.py
"""
Persistent ChromaDB store for company documents.
Supported formats: PDF, DOCX, TXT, MD, CSV.
"""

import logging
import csv
from pathlib import Path
from typing import List, Dict, Any

from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    CSVLoader,
)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from fastembed import TextEmbedding

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".csv"}

logger = logging.getLogger(__name__)

# Persist RAG store next to the project root
RAG_STORE_DIR = str(Path(__file__).parent.parent / "rag_store" / "chroma_db")
COLLECTION_NAME = "company_policies"

# Same model already used in planner.py — no extra download needed
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class _FastEmbedWrapper:
    """
    Thin LangChain-compatible wrapper around fastembed.TextEmbedding.
    Avoids pulling in langchain-community's fastembed wrapper which has
    extra dependencies.
    """

    def __init__(self, model_name: str):
        self._model = TextEmbedding(model_name=model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [list(v) for v in self._model.embed(texts)]

    def embed_query(self, text: str) -> List[float]:
        return list(next(self._model.embed([text])))


class DocumentStore:
    """
    Handles PDF ingestion and semantic retrieval for the RAG agent.
    """

    def __init__(self):
        self._embeddings = _FastEmbedWrapper(EMBED_MODEL)
        self._store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=self._embeddings,
            persist_directory=RAG_STORE_DIR,
        )
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=120,
            separators=["\n\n", "\n", ". ", " "],
        )
        logger.info(f"DocumentStore ready — {self._store._collection.count()} chunks in index")

    # ── Ingestion ──────────────────────────────────────────────────────────────

    def add_document(self, file_path: str, source_label: str = "") -> int:
        """
        Load any supported file, chunk it, and add to the vector store.
        Supported: .pdf  .docx  .txt  .md  .csv

        Returns the number of chunks added.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = path.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type '{ext}'. "
                f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )

        label = source_label or path.name

        # Pick the right loader
        if ext == ".pdf":
            raw_docs = PyPDFLoader(str(path)).load()
        elif ext == ".docx":
            raw_docs = Docx2txtLoader(str(path)).load()
        elif ext in (".txt", ".md"):
            raw_docs = TextLoader(str(path), encoding="utf-8").load()
        elif ext == ".csv":
            raw_docs = CSVLoader(str(path), encoding="utf-8").load()
        else:
            raw_docs = []

        if not raw_docs:
            logger.warning(f"No text extracted from {path.name}")
            return 0

        for doc in raw_docs:
            doc.metadata["source"] = label

        chunks = self._splitter.split_documents(raw_docs)
        if not chunks:
            logger.warning(f"No chunks produced from {path.name}")
            return 0

        self._store.add_documents(chunks)
        logger.info(f"Indexed {len(chunks)} chunks from '{label}' ({ext})")
        return len(chunks)

    def add_pdf(self, file_path: str, source_label: str = "") -> int:
        """Alias for add_document — kept for backwards compatibility."""
        return self.add_document(file_path, source_label)

    def delete_source(self, source_label: str) -> int:
        """Remove all chunks belonging to a specific source document."""
        collection = self._store._collection
        results = collection.get(where={"source": source_label})
        ids = results.get("ids", [])
        if ids:
            collection.delete(ids=ids)
            logger.info(f"Deleted {len(ids)} chunks for source '{source_label}'")
        return len(ids)

    def list_sources(self) -> List[str]:
        """Return unique source document names currently indexed."""
        collection = self._store._collection
        results = collection.get(include=["metadatas"])
        sources = {m.get("source", "") for m in results.get("metadatas", [])}
        return sorted(s for s in sources if s)

    # ── Retrieval ──────────────────────────────────────────────────────────────

    def search(self, query: str, k: int = 4) -> List[Dict[str, Any]]:
        """
        Retrieve the top-k most relevant chunks.

        Returns list of dicts: {content, source, page, score}
        Scores are cosine distances — lower = more relevant.
        """
        if self._store._collection.count() == 0:
            return []

        results = self._store.similarity_search_with_relevance_scores(query, k=k)
        hits = []
        for doc, score in results:
            hits.append({
                "content": doc.page_content,
                "source": doc.metadata.get("source", "unknown"),
                "page": doc.metadata.get("page", "?"),
                "score": round(score, 3),
            })
        return hits

    def has_documents(self) -> bool:
        return self._store._collection.count() > 0

    @property
    def chunk_count(self) -> int:
        return self._store._collection.count()


# Singleton — imported by rag_agent.py and the API
document_store = DocumentStore()

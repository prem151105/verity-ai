"""
ChromaDB vector store wrapper.
Chunks text, embeds via Gemini embedding API, stores locally (no external infra).
"""

import hashlib
import logging
import textwrap
from dataclasses import dataclass
from typing import Optional

import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)

CHUNK_SIZE = 800        # characters per chunk
CHUNK_OVERLAP = 150     # overlap between consecutive chunks
MAX_RESULTS = 6         # default top-k retrieval


@dataclass
class RetrievedChunk:
    text: str
    source: str          # e.g. "10-K FY2024 — AAPL"
    chunk_id: str
    distance: float
    metadata: dict


def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks of roughly `chunk_size` chars."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        start += chunk_size - overlap
    return chunks


class VectorStore:
    """
    Local ChromaDB vector store.
    Uses Gemini embedding API for embeddings.
    One collection per research run (keyed by ticker + run_id).
    """

    def __init__(self, persist_dir: str, gemini_api_key: str):
        self._client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self._gemini_api_key = gemini_api_key
        self._embedding_fn = self._make_embedding_fn()

    def _make_embedding_fn(self):
        """Create a ChromaDB-compatible embedding function using Gemini."""
        from google import genai
        api_key = self._gemini_api_key
        client = genai.Client(api_key=api_key)

        class GeminiEmbeddingFunction(chromadb.EmbeddingFunction):
            def __call__(self, input: list[str]) -> list[list[float]]:
                embeddings = []
                for text in input:
                    result = client.models.embed_content(
                        model="models/gemini-embedding-2",
                        contents=text,
                    )
                    embeddings.append(result.embeddings[0].values)
                return embeddings

        return GeminiEmbeddingFunction()

    def get_or_create_collection(self, collection_name: str) -> chromadb.Collection:
        return self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=self._embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

    def add_document(
        self,
        collection_name: str,
        text: str,
        source: str,
        metadata: dict | None = None,
    ) -> int:
        """
        Chunk and embed a document into the collection.

        Returns:
            Number of chunks added.
        """
        collection = self.get_or_create_collection(collection_name)
        chunks = _chunk_text(text)

        if not chunks:
            logger.warning(f"No text chunks produced for source: {source}")
            return 0

        ids = []
        metadatas = []
        documents = []

        for i, chunk in enumerate(chunks):
            chunk_id = hashlib.sha256(f"{source}:{i}:{chunk[:50]}".encode()).hexdigest()[:16]
            ids.append(chunk_id)
            metadatas.append(
                {
                    "source": source,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    **(metadata or {}),
                }
            )
            documents.append(chunk)

        # Add in batches to avoid hitting embedding API limits
        BATCH_SIZE = 50
        for batch_start in range(0, len(ids), BATCH_SIZE):
            batch_end = batch_start + BATCH_SIZE
            collection.add(
                ids=ids[batch_start:batch_end],
                documents=documents[batch_start:batch_end],
                metadatas=metadatas[batch_start:batch_end],
            )

        logger.info(f"Added {len(chunks)} chunks from '{source}' to '{collection_name}'")
        return len(chunks)

    def query(
        self,
        collection_name: str,
        query_text: str,
        n_results: int = MAX_RESULTS,
        where: dict | None = None,
    ) -> list[RetrievedChunk]:
        """
        Retrieve top-k relevant chunks for a query.

        Returns:
            List of RetrievedChunk sorted by relevance (closest first).
        """
        collection = self.get_or_create_collection(collection_name)

        try:
            count = collection.count()
        except Exception:
            count = 0

        if count == 0:
            logger.warning(f"Collection '{collection_name}' is empty.")
            return []

        n_results = min(n_results, count)

        kwargs = {"query_texts": [query_text], "n_results": n_results}
        if where:
            kwargs["where"] = where

        results = collection.query(**kwargs)

        chunks = []
        for doc, meta, dist, chunk_id in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
            results["ids"][0],
        ):
            chunks.append(
                RetrievedChunk(
                    text=doc,
                    source=meta.get("source", "unknown"),
                    chunk_id=chunk_id,
                    distance=dist,
                    metadata=meta,
                )
            )

        return chunks

    def delete_collection(self, collection_name: str) -> None:
        """Remove a collection (e.g., after a run completes)."""
        try:
            self._client.delete_collection(collection_name)
            logger.info(f"Deleted collection: {collection_name}")
        except Exception as e:
            logger.warning(f"Could not delete collection '{collection_name}': {e}")

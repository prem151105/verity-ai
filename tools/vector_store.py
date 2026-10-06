"""
Persistent passage retrieval. Local BM25 by default; optional batched Gemini embeddings.
"""

import hashlib
import logging
import time
import textwrap
from dataclasses import dataclass
from typing import Optional


logger = logging.getLogger(__name__)

CHUNK_SIZE = 1800       # fewer requests while retaining passage-sized contexts
CHUNK_OVERLAP = 250
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


class GeminiVectorStore:
    """
    Local ChromaDB vector store.
    Uses Gemini embedding API for embeddings.
    One collection per research run (keyed by ticker + run_id).
    """

    def __init__(self, persist_dir: str, gemini_api_key: str):
        import chromadb
        from chromadb.config import Settings
        self._client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self._gemini_api_key = gemini_api_key
        from google import genai
        self._genai = genai.Client(api_key=gemini_api_key, http_options={"timeout": 60000})

    def _embed(self, texts, query=False):
        """Embedding-001 returns one vector per text; Embedding-2 aggregates lists."""
        from google.genai import types
        from google.genai.errors import APIError
        for attempt in range(3):
            try:
                result = self._genai.models.embed_content(
                    model='gemini-embedding-001', contents=texts,
                    config=types.EmbedContentConfig(
                        task_type='RETRIEVAL_QUERY' if query else 'RETRIEVAL_DOCUMENT',
                        output_dimensionality=768,
                    ),
                )
                if len(result.embeddings or []) != len(texts):
                    raise RuntimeError('Embedding service returned the wrong number of vectors')
                return [e.values for e in result.embeddings]
            except APIError as exc:
                if exc.code not in {429, 500, 502, 503, 504} or attempt == 2:
                    raise
                time.sleep(5 * (attempt + 1))

    def get_or_create_collection(self, collection_name: str):
        return self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=None,
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
        BATCH_SIZE = 32
        for batch_start in range(0, len(ids), BATCH_SIZE):
            batch_end = batch_start + BATCH_SIZE
            collection.add(
                ids=ids[batch_start:batch_end],
                documents=documents[batch_start:batch_end],
                metadatas=metadatas[batch_start:batch_end],
                embeddings=self._embed(documents[batch_start:batch_end]),
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

        kwargs = {"query_embeddings": self._embed([query_text], query=True), "n_results": n_results}
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


class BM25Store:
    """Persistent lexical passage search over real documents; no embedding calls."""
    def __init__(self, persist_dir, gemini_api_key=''):
        from pathlib import Path
        root = Path(persist_dir)
        root.mkdir(parents=True, exist_ok=True)
        self.path = root / 'passages.sqlite3'
        self._cached = {}
        with self._connect() as db:
            db.execute('CREATE TABLE IF NOT EXISTS passages '
                       '(collection TEXT, id TEXT, text TEXT, source TEXT, metadata TEXT, '
                       'PRIMARY KEY (collection, id))')

    def _connect(self):
        from contextlib import contextmanager
        import sqlite3
        @contextmanager
        def connection():
            db = sqlite3.connect(self.path, timeout=15)
            try:
                with db:
                    yield db
            finally:
                db.close()
        return connection()

    def add_document(self, collection_name, text, source, metadata=None):
        import json
        chunks = _chunk_text(text)
        rows = [(collection_name, hashlib.sha256((source+'\0'+chunk).encode()).hexdigest(),
                 chunk, source, json.dumps(metadata or {})) for chunk in chunks]
        with self._connect() as db:
            db.executemany('INSERT OR REPLACE INTO passages VALUES (?, ?, ?, ?, ?)', rows)
        self._cached.pop(collection_name, None)
        return len(chunks)

    def query(self, collection_name, query_text, n_results=MAX_RESULTS, where=None):
        import json
        import math
        from collections import Counter
        from research.evidence import tokens
        if n_results < 1:
            return []
        if collection_name not in self._cached:
            with self._connect() as db:
                rows = db.execute('SELECT id, text, source, metadata FROM passages WHERE collection = ?',
                                  (collection_name,)).fetchall()
            prepared = [(row, Counter(tokens(row[1]))) for row in rows]
            df = Counter(term for _, counts in prepared for term in counts)
            average = sum(sum(counts.values()) for _, counts in prepared) / max(len(prepared), 1)
            self._cached[collection_name] = (prepared, df, average)
        prepared, df, average = self._cached[collection_name]
        ranked = []
        for row, counts in prepared:
            metadata = {'source': row[2], **json.loads(row[3])}
            if where and any(metadata.get(k) != v for k,v in where.items()):
                continue
            score = 0.0
            length = sum(counts.values())
            for term in set(tokens(query_text)):
                frequency = counts[term]
                idf = math.log(1 + (len(prepared)-df[term]+.5)/(df[term]+.5))
                score += idf * frequency * 2.5 / (frequency + 1.5*(.25+.75*length/max(average, 1)))
            if score > 0:
                ranked.append((score, row, metadata))
        ranked.sort(key=lambda item: (-item[0], item[1][0]))
        return [RetrievedChunk(text=row[1], source=row[2], chunk_id=row[0],
                               distance=1/(1+score), metadata=metadata)
                for score,row,metadata in ranked[:n_results]]

    def delete_collection(self, collection_name):
        with self._connect() as db:
            db.execute('DELETE FROM passages WHERE collection = ?', (collection_name,))
        self._cached.pop(collection_name, None)


class VectorStore:
    """Select configured retrieval without changing agent call sites."""
    def __new__(cls, persist_dir, gemini_api_key):
        from config import settings
        backend = GeminiVectorStore if settings.retrieval_backend == 'gemini' else BM25Store
        return backend(persist_dir, gemini_api_key)

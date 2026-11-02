"""Simple RAG retrieval using Ollama embeddings and NumPy."""

import os
from pathlib import Path
from typing import TypedDict

import httpx
import numpy as np


class RetrievalResult(TypedDict):
    source: str
    text: str
    score: float


OLLAMA_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://localhost:11434",
)
EMBED_MODEL = "nomic-embed-text"

KNOWLEDGE_DIR = Path(__file__).parent / "knowledge_base"


def embed_texts(texts: list[str]) -> np.ndarray:
    """Convert texts into validated embedding vectors using Ollama."""

    if not texts or any(
        not isinstance(text, str) or not text.strip() for text in texts
    ):
        raise ValueError("Embedding input must contain non-empty text strings.")

    response = httpx.post(
        f"{OLLAMA_URL}/api/embed",
        json={
            "model": EMBED_MODEL,
            "input": texts,
        },
        timeout=120.0,
    )
    response.raise_for_status()

    try:
        payload = response.json()
    except ValueError as exc:
        raise ValueError("Ollama returned invalid JSON for embeddings.") from exc

    if not isinstance(payload, dict) or "embeddings" not in payload:
        raise ValueError("Ollama embedding response must contain 'embeddings'.")

    try:
        embeddings = np.asarray(payload["embeddings"], dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("Ollama returned malformed embedding values.") from exc

    if embeddings.ndim != 2:
        raise ValueError("Embeddings must be a two-dimensional array.")

    if embeddings.shape[0] != len(texts):
        raise ValueError("Ollama returned an unexpected number of embeddings.")

    if embeddings.shape[1] == 0:
        raise ValueError("Embedding vectors must have a non-zero dimension.")

    if not np.all(np.isfinite(embeddings)):
        raise ValueError("Embedding vectors must contain only finite values.")

    norms = np.linalg.norm(embeddings, axis=1)

    if np.any(norms == 0):
        raise ValueError("Embedding vectors must not be zero vectors.")

    return embeddings.astype(np.float32)


def load_documents() -> list[dict[str, str]]:
    """Load Markdown documents and split them by headings."""

    documents = []

    for path in sorted(KNOWLEDGE_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        sections = text.split("\n# ")

        for section in sections:
            content = section.strip()
            if not content:
                continue

            # Restore the heading marker removed by split().
            if not content.startswith("# "):
                content = "# " + content

            documents.append(
                {
                    "source": path.name,
                    "text": content,
                }
            )

    if not documents:
        raise RuntimeError("The wireless knowledge base is empty or unavailable.")

    return documents


def retrieve(
    query: str,
    top_k: int = 3,
) -> list[RetrievalResult]:
    """Return the most relevant knowledge sections."""

    if not query.strip():
        raise ValueError("Query cannot be empty.")

    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    documents = load_documents()

    # Embed the documents and question.
    try:
        document_vectors = embed_texts([document["text"] for document in documents])
        query_vector = embed_texts([query])[0]

        if document_vectors.shape[1] != query_vector.shape[0]:
            raise ValueError("Document and query embedding dimensions do not match.")

    except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
        raise RuntimeError(
            "Could not retrieve valid embeddings from Ollama. "
            "Check that Ollama is running and the embedding model "
            "is available."
        ) from exc

    # Cosine similarity: larger means more similar.
    # Use float64 for the norm calculation to reduce overflow risk.
    document_norms = np.linalg.norm(
        document_vectors.astype(np.float64),
        axis=1,
    )
    query_norm = np.linalg.norm(query_vector.astype(np.float64))

    similarities = (document_vectors @ query_vector) / (
        document_norms * max(float(query_norm), 1e-12)
    )

    if not np.all(np.isfinite(similarities)):
        raise RuntimeError("Embedding similarity calculation produced invalid values.")

    best_indices = np.argsort(similarities)[::-1][:top_k]

    results: list[RetrievalResult] = []

    for index in best_indices:
        document = documents[int(index)]

        results.append(
            {
                "source": document["source"],
                "text": document["text"],
                "score": float(similarities[index]),
            }
        )

    return results

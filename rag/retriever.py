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
    """Convert texts into embedding vectors using Ollama."""

    response = httpx.post(
        f"{OLLAMA_URL}/api/embed",
        json={
            "model": EMBED_MODEL,
            "input": texts,
        },
        timeout=120.0,
    )
    response.raise_for_status()

    embeddings = response.json()["embeddings"]
    return np.asarray(embeddings, dtype=np.float32)


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
        raise ValueError(f"No Markdown documents found in {KNOWLEDGE_DIR}")

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

    except httpx.HTTPError as exc:
        raise RuntimeError(
            "Could not retrieve embeddings from Ollama. "
            "Check that Ollama is running and the embedding model "
            "is available."
        ) from exc

    # Cosine similarity: larger means more similar.
    document_norms = np.linalg.norm(
        document_vectors,
        axis=1,
        keepdims=True,
    )
    query_norm = np.linalg.norm(query_vector)

    similarities = (document_vectors @ query_vector) / (
        document_norms[:, 0] * max(float(query_norm), 1e-12)
    )

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

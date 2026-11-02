# Test the RAG retriever
import numpy as np
import pytest

import rag.retriever as retriever
from rag.retriever import retrieve


def test_retrieval_returns_relevant_documents():
    results = retrieve(
        "What is the theoretical BER of BPSK over AWGN?",
        top_k=3,
    )

    assert results
    assert len(results) <= 3
    assert all("text" in result for result in results)
    assert all("score" in result for result in results)

    combined_text = " ".join(result["text"].lower() for result in results)

    assert "bpsk" in combined_text
    assert "awgn" in combined_text


class FakeEmbeddingResponse:
    """Minimal mock of an Ollama HTTP response."""

    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


def test_embed_texts_returns_valid_vectors(monkeypatch):
    monkeypatch.setattr(
        retriever.httpx,
        "post",
        lambda *args, **kwargs: FakeEmbeddingResponse(
            {"embeddings": [[1.0, 2.0], [3.0, 4.0]]}
        ),
    )

    result = retriever.embed_texts(["BPSK", "QPSK"])

    assert result.shape == (2, 2)
    assert np.all(np.isfinite(result))


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"embeddings": [[1.0, 2.0]]},
        {"embeddings": [1.0, 2.0]},
        {"embeddings": [[1.0, float("nan")], [3.0, 4.0]]},
        {"embeddings": [[1.0, float("inf")], [3.0, 4.0]]},
        {"embeddings": [[0.0, 0.0], [3.0, 4.0]]},
        {"embeddings": [[], []]},
        {"embeddings": [["invalid", 2.0], [3.0, 4.0]]},
    ],
)
def test_embed_texts_rejects_malformed_embeddings(monkeypatch, payload):
    monkeypatch.setattr(
        retriever.httpx,
        "post",
        lambda *args, **kwargs: FakeEmbeddingResponse(payload),
    )

    with pytest.raises(ValueError):
        retriever.embed_texts(["BPSK", "QPSK"])


def test_embed_texts_rejects_empty_input():
    with pytest.raises(ValueError, match="non-empty text"):
        retriever.embed_texts([])


def test_embed_texts_rejects_blank_input():
    with pytest.raises(ValueError, match="non-empty text"):
        retriever.embed_texts(["BPSK", "   "])


def test_retrieve_rejects_mismatched_embedding_dimensions(monkeypatch):
    monkeypatch.setattr(
        retriever,
        "load_documents",
        lambda: [{"source": "wireless.md", "text": "# BPSK"}],
    )

    monkeypatch.setattr(
        retriever,
        "embed_texts",
        lambda texts: (
            np.array([[1.0, 2.0]], dtype=np.float32)
            if len(texts) == 1 and texts[0] == "# BPSK"
            else np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        ),
    )

    with pytest.raises(RuntimeError, match="Could not retrieve valid embeddings"):
        retriever.retrieve("BPSK")

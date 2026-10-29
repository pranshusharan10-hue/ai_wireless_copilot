# Test the RAG retriever

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

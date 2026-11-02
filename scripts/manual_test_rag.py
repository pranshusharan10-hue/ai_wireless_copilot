from rag.retriever import retrieve

results = retrieve(
    "Explain Rayleigh fading and its average BER",
    top_k=3,
)

for result in results:
    print(f"Score: {result['score']:.3f}")
    print(result["text"])
    print("-" * 60)

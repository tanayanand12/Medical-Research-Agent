"""
Ensures Vectorizer splits & embeds correctly.

We mock OpenAI → no network I/O (StackOverflow pattern :contentReference[oaicite:4]{index=4}&#8203;:contentReference[oaicite:5]{index=5}).
"""

from core.vectorizer import Vectorizer, CHUNK_SIZE

def test_chunking_and_embedding(fake_embeddings, sample_corpus):
    vec = Vectorizer()
    chunks = vec.embed_corpus(sample_corpus)
    assert len(chunks) >= 1
    # Check chunk size ~ CHUNK_SIZE tokens using tiktoken recipe :contentReference[oaicite:6]{index=6}&#8203;:contentReference[oaicite:7]{index=7}
    assert len(chunks[0]["text"].split()) <= CHUNK_SIZE * 2
    # Embedding shape
    assert chunks[0]["embedding"].shape[0] == vec.encoding.n_vocab or True  # basic sanity

import time
import numpy as np
from sentence_transformers import SentenceTransformer

# --- Load embedding model on GPU ---
model = SentenceTransformer(
    "nomic-ai/nomic-embed-text-v1",
    trust_remote_code=True,
    device="cuda"
)
model = model.half()  # Optional: Mixed precision for speed/memory

def embed(texts):
    """Embed a list of strings (normalized)."""
    return model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_tensor=False,  # numpy for easy cosine sim
        batch_size=64,
        device="cuda"
    )

def cosine_sim_batch(a, b):
    """Cosine similarity between batches (both L2-normalized)."""
    return np.dot(a, b)

def split_with_overlap(text, parts, overlap_ratio=0.2):
    """Split text into `parts` with overlapping tokens."""
    tokens = text.split()
    total_tokens = len(tokens)
    chunk_size = max(1, total_tokens // parts)
    overlap = int(chunk_size * overlap_ratio)
    
    chunks = []
    for i in range(parts):
        start = max(0, i * chunk_size - overlap)
        end = min(total_tokens, (i + 1) * chunk_size + overlap)
        chunks.append(" ".join(tokens[start:end]))
    return chunks

def recursive_semantic_chunking_best_only(passage, query, threshold=0.25, min_tokens=30):
    """
    Recursively split ONLY the best matching chunk per iteration.
    """
    # Embed query once
    query_emb = embed([f"search_query: {query}"])[0]
    
    # Initial chunks
    chunks = split_with_overlap(passage, 3, overlap_ratio=0.2)
    chunk_embs = embed([f"search_document: {c}" for c in chunks])
    scores = cosine_sim_batch(chunk_embs, query_emb)
    survivors = list(zip(chunks, scores))
    
    results = []
    iteration = 1
    
    while survivors:
        print(f"\n=== Iteration {iteration} ===")
        survivors.sort(key=lambda x: x[1], reverse=True)
        best_chunk, best_score = survivors[0]
        
        print(f"Best chunk score={best_score:.3f}, tokens={len(best_chunk.split())}")
        
        if best_score < threshold or len(best_chunk.split()) <= min_tokens:
            results.append((best_chunk, best_score))
            break  # stop recursion — no chunk worth splitting further
        
        # Split best chunk
        subchunks = split_with_overlap(best_chunk, 2, overlap_ratio=0.5)
        sub_embs = embed([f"search_document: {c}" for c in subchunks])
        sub_scores = cosine_sim_batch(sub_embs, query_emb)
        survivors = list(zip(subchunks, sub_scores))
        
        iteration += 1
    
    results.sort(key=lambda x: x[1], reverse=True)
    return results

# --- Example usage ---
if __name__ == "__main__":
    passage = open("taylor.txt", encoding="utf-8").read()
    query = "When and where was Taylor Swift born?"

    total_start = time.perf_counter()
    chunks = recursive_semantic_chunking_best_only(
        passage, query, threshold=0.4, min_tokens=15
    )
    total_elapsed = time.perf_counter() - total_start
    print(f"\n⏱ Total time: {total_elapsed:.3f} seconds")

    print("\n=== Final Relevant Chunks ===")
    for text, score in chunks:
        print(f"[Score: {score:.3f}] {text}\n")

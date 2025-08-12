import time
import numpy as np
from sentence_transformers import SentenceTransformer

# --- Load embedding model on GPU ---
model = SentenceTransformer(
    "nomic-ai/nomic-embed-text-v1",
    trust_remote_code=True,
    device="cuda"
)
model = model.half()  # Mixed precision

def embed(texts):
    """Embed a list of strings (normalized)."""
    return model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_tensor=False,  # numpy for cosine sim
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

def expand_to_full_sentence(passage, chunk):
    """Expand the given chunk to full sentence(s) in original passage."""
    idx = passage.find(chunk)
    if idx == -1:
        return chunk  # fallback if not found
    
    # Find sentence start
    start = idx
    while start > 0 and passage[start - 1] not in ".!?":
        start -= 1
    if start > 0:  # skip punctuation
        start += 1
    while start < len(passage) and passage[start].isspace():
        start += 1
    
    # Find sentence end
    end = idx + len(chunk)
    while end < len(passage) and passage[end] not in ".!?":
        end += 1
    if end < len(passage):
        end += 1

    return passage[start:end].strip()

def recursive_semantic_chunking_multi(passage, query, start_candidates=3, threshold=0.25, min_tokens=30, tolerance=0.01):
    """Recursively split each of the top start_candidates until stopping conditions are met."""
    query_emb = embed([f"search_query: {query}"])[0]

    # Initial candidates
    chunks = split_with_overlap(passage, start_candidates, overlap_ratio=0.2)
    chunk_embs = embed([f"search_document: {c}" for c in chunks])
    scores = cosine_sim_batch(chunk_embs, query_emb)
    survivors = list(zip(chunks, scores))

    final_results = []

    for idx, (chunk, score) in enumerate(survivors, start=1):
        iteration = 1
        best_chunk = chunk
        best_score = score
        candidates = [(chunk, score)]

        print(f"\n=== Candidate {idx} initial score={score:.3f} ===")

        while candidates:
            candidates.sort(key=lambda x: x[1], reverse=True)
            best_chunk, best_score = candidates[0]
            print(f"  Iter {iteration}: score={best_score:.3f}, tokens={len(best_chunk.split())}")

            if best_score < threshold or len(best_chunk.split()) <= min_tokens:
                break

            # Split into halves
            subchunks = split_with_overlap(best_chunk, 2, overlap_ratio=0.5)
            sub_embs = embed([f"search_document: {c}" for c in subchunks])
            sub_scores = cosine_sim_batch(sub_embs, query_emb)

            # Stop if both worse
            if all(s < best_score - tolerance for s in sub_scores):
                break

            candidates = list(zip(subchunks, sub_scores))
            iteration += 1

        expanded_chunk = expand_to_full_sentence(passage, best_chunk)
        final_results.append((expanded_chunk, best_score))

    # Sort final results by score
    final_results.sort(key=lambda x: x[1], reverse=True)
    return final_results

# --- Example usage ---
if __name__ == "__main__":
    passage = open("taylor.txt", encoding="utf-8").read()
    while True:
        query = input(">> ")
        if query.strip().lower() == "exit":
            break
        
        total_start = time.perf_counter()
        chunks = recursive_semantic_chunking_multi(
            passage, query, start_candidates=3, threshold=0.4, min_tokens=15, tolerance=0.05
        )
        total_elapsed = time.perf_counter() - total_start
        print(f"\n⏱ Total time: {total_elapsed:.3f} seconds")

        print("\n=== Final Relevant Chunks ===")
        for text, score in chunks:
            print(f"[Score: {score:.3f}] {text}\n")

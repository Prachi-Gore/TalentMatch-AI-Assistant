from functools import lru_cache
import pickle

import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer

from hireflow.config import settings
from hireflow.llm import get_llm
from hireflow.pinecone_client import get_resume_index


@lru_cache(maxsize=1)
def get_dense_model() -> SentenceTransformer:
    return SentenceTransformer(settings.dense_model_name)


@lru_cache(maxsize=1)
def get_rerank_model() -> CrossEncoder:
    return CrossEncoder(settings.rerank_model_name)


def expand_query(user_query: str) -> str:
    prompt = f"""
Imagine you are helping retrieve the most suitable candidate resumes.
Write 3 variations based on the user query.
Return only the rewritten queries.

Query:
{user_query}
""".strip()

    response = get_llm().invoke(prompt)
    content = getattr(response, "content", response)
    return f"{user_query}\n{content}"


def hybrid_query(query_text: str, alpha: float = 0.5, top_k: int = 8) -> list[dict]:
    alpha = max(0.0, min(1.0, float(alpha)))

    with settings.tfidf_path.open("rb") as f:
        vectorizer = pickle.load(f)

    q_dense = get_dense_model().encode([query_text], normalize_embeddings=True)[0]
    q_dense = (np.asarray(q_dense, dtype=float) * (1.0 - alpha)).tolist()

    q_sparse_csr = vectorizer.transform([query_text]).tocoo()
    if q_sparse_csr.nnz == 0:
        q_sparse = {"indices": [0], "values": [0.0]}
    else:
        q_sparse = {
            "indices": q_sparse_csr.col.tolist(),
            "values": (q_sparse_csr.data.astype(float) * alpha).tolist(),
        }

    response = get_resume_index().query(
        vector=q_dense,
        sparse_vector=q_sparse,
        top_k=top_k,
        include_metadata=True,
    )

    results = []
    for match in response.get("matches", []):
        metadata = match.get("metadata", {}) or {}
        text = metadata.get("text", "") or ""
        preview = " ".join(text.split()[:120])
        results.append(
            {
                "id": match["id"],
                "score": float(match["score"]),
                "preview": preview,
                "metadata": metadata,
            }
        )

    return results


def rerank_results(query: str, results: list[dict]) -> list[dict]:
    if not results:
        return []

    pairs = [(query, result["preview"]) for result in results]
    scores = get_rerank_model().predict(pairs)

    rescored = []
    for result, score in zip(results, scores):
        updated = dict(result)
        updated["rerank_cross_score"] = float(score)
        rescored.append(updated)

    return sorted(
        rescored,
        key=lambda item: item["rerank_cross_score"],
        reverse=True,
    )


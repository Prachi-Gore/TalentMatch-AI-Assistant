from functools import lru_cache

from pinecone import Pinecone, ServerlessSpec

from hireflow.config import settings


@lru_cache(maxsize=1)
def get_pinecone() -> Pinecone:
    return Pinecone(api_key=settings.pinecone_api_key)


def ensure_resume_index():
    pc = get_pinecone()
    existing = [idx["name"] for idx in pc.list_indexes()]

    if settings.resume_index_name not in existing:
        # Create index with hybrid search support (dense + sparse)
        pc.create_index(
            name=settings.resume_index_name,
            dimension=384,  # Dense vector dimension
            metric="dotproduct",
            spec=ServerlessSpec(
                cloud=settings.pinecone_cloud,
                region=settings.pinecone_region,
            ),
        )

    return pc.Index(settings.resume_index_name)


@lru_cache(maxsize=1)
def get_resume_index():
    pc = get_pinecone()
    return pc.Index(settings.resume_index_name)


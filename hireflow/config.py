from dataclasses import dataclass
from pathlib import Path
import os

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Settings:
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    pinecone_api_key: str | None = os.getenv("PINECONE_API_KEY")

    resume_index_name: str = os.getenv("RESUME_INDEX_NAME")
    pinecone_cloud: str = os.getenv("PINECONE_CLOUD", "aws")
    pinecone_region: str = os.getenv("PINECONE_REGION", "us-east-1")

    dense_model_name: str = os.getenv(
        "DENSE_MODEL_NAME",
        "sentence-transformers/all-MiniLM-L6-v2",
    )
    rerank_model_name: str = os.getenv(
        "RERANK_MODEL_NAME",
        "cross-encoder/ms-marco-MiniLM-L-6-v2",
    )
    llm_model: str = os.getenv("OPENAI_MODEL", "gpt-4.1")

    resume_dir: Path = Path(os.getenv("RESUME_DIR", "resume_dir"))
    tfidf_path: Path = Path(os.getenv("TFIDF_PATH", "tfidf_vectorizer.pkl"))
    prompt_path: Path = Path(os.getenv("PROMPT_PATH", "prompt.yaml"))


settings = Settings()


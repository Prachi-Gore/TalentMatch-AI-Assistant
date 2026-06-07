from pathlib import Path
from typing import Any
import json
import pickle

from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.feature_extraction.text import TfidfVectorizer

from hireflow.config import settings
from hireflow.llm import get_llm
from hireflow.pinecone_client import ensure_resume_index


def load_resume_text(path: Path) -> str:
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        docs = PyPDFLoader(str(path)).load()
    elif suffix == ".docx":
        docs = Docx2txtLoader(str(path)).load()
    elif suffix in {".txt", ".md"}:
        docs = TextLoader(str(path), encoding="utf-8").load()
    else:
        raise ValueError(f"Unsupported resume file type: {path.suffix}")

    return "\n".join(doc.page_content for doc in docs)


def parse_resume_with_llm(raw_text: str) -> dict[str, Any]:
    prompt = f"""
You are a strict JSON resume parser.
Return ONLY valid minified JSON. No markdown, no commentary.

Schema:
{{
  "summary": "string",
  "skills": ["string"],
  "CERTIFICATIONS": ["string"],
  "email": ["string"],
  "Location": ["string"],
  "experiences": [
    {{
      "title": "string",
      "company": "string",
      "location": "string",
      "start_date": "string",
      "end_date": "string",
      "description": "string",
      "skills": ["string"]
    }}
  ],
  "education": [
    {{
      "degree": "string",
      "institution": "string",
      "year": "string"
    }}
  ],
  "projects": [
    {{
      "name": "string",
      "description": "string",
      "skills": ["string"]
    }}
  ]
}}

If something is missing, use "" or []. Do not invent facts.

Resume:
\"\"\"{raw_text[:12000]}\"\"\"
""".strip()

    response = get_llm().invoke(prompt)
    content = getattr(response, "content", response)
    return json.loads(content)


def build_resume_document(parsed: dict[str, Any], resume_id: str, filename: str) -> Document:
    skills = parsed.get("skills") or []
    experiences = parsed.get("experiences") or []
    location = parsed.get("Location") or []

    searchable_parts = []
    if parsed.get("summary"):
        searchable_parts.append(f"Summary: {parsed['summary']}")
    if skills:
        searchable_parts.append("Skills: " + ", ".join(skills))
    if location:
        searchable_parts.append("Location: " + ", ".join(location))

    roles = {
        (exp.get("title") or "").strip()
        for exp in experiences
        if (exp.get("title") or "").strip()
    }
    companies = {
        (exp.get("company") or "").strip()
        for exp in experiences
        if (exp.get("company") or "").strip()
    }

    metadata = {
        "resume_id": resume_id,
        "filename": filename,
        "link": str(settings.resume_dir / filename),
        "skills": sorted({skill.strip() for skill in skills if skill.strip()}),
        "roles": sorted(roles),
        "companies": sorted(companies),
        "location": sorted(location),
    }

    return Document(page_content="\n".join(searchable_parts).strip(), metadata=metadata)


def load_and_parse_resumes(resume_dir: Path = settings.resume_dir) -> list[Document]:
    docs = []

    for path in sorted(resume_dir.iterdir()):
        if path.is_dir():
            continue

        raw_text = load_resume_text(path).strip()
        if not raw_text:
            continue

        parsed = parse_resume_with_llm(raw_text)
        docs.append(build_resume_document(parsed, path.stem, path.name))

    return docs


def encode_documents(docs: list[Document]):
    corpus = [doc.page_content for doc in docs]

    embedder = HuggingFaceEmbeddings(
        model_name=settings.dense_model_name,
        encode_kwargs={"normalize_embeddings": True},
    )
    dense_vectors = embedder.embed_documents(corpus) # semantic meaning

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
    )
    tfidf_matrix = vectorizer.fit_transform(corpus)

    with settings.tfidf_path.open("wb") as f:
        pickle.dump(vectorizer, f)

    return dense_vectors, tfidf_matrix


def csr_row_to_pinecone_sparse(csr_row) -> dict[str, list[float]]:
    # Sparse vector = exact keyword matching
    coo = csr_row.tocoo()
    return {
        "indices": coo.col.tolist(),
        "values": coo.data.astype(float).tolist(),
    }


def upsert_documents(docs: list[Document], dense_vectors, tfidf_matrix) -> int:
    index = ensure_resume_index()
    vectors = []

    for i, (doc, dense) in enumerate(zip(docs, dense_vectors)):
        metadata = {
            **doc.metadata,
            "text": doc.page_content[:1200],
        }
        vectors.append(
            {
                "id": doc.metadata["resume_id"],
                "values": dense,
                "sparse_values": csr_row_to_pinecone_sparse(tfidf_matrix[i]),
                "metadata": metadata,
            }
        )

    if vectors:
        index.upsert(vectors=vectors)

    return len(vectors)


def ingest_resumes(resume_dir: Path = settings.resume_dir) -> int:
    docs = load_and_parse_resumes(resume_dir)
    dense_vectors, tfidf_matrix = encode_documents(docs)
    return upsert_documents(docs, dense_vectors, tfidf_matrix)


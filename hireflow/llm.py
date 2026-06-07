from functools import lru_cache

from langchain_openai import ChatOpenAI

from hireflow.config import settings


@lru_cache(maxsize=1)
def get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.openai_api_key,
    )


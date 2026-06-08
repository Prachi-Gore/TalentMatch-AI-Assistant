from langchain_core.runnables import RunnableLambda

from hireflow.config import settings
from hireflow.llm import get_llm
from hireflow.prompts import load_prompt, render_resume_prompt, resume_parser
from hireflow.retrieval import expand_query, hybrid_query, rerank_results


def _start(user_query: str) -> dict:
    return {"query": user_query}


def _extract_answer(answer_obj) -> str:
    content = getattr(answer_obj, "content", answer_obj)
    return str(content).strip()


def build_retrieval_chain():
    return (
        RunnableLambda(_start)
        .assign(multi_query=RunnableLambda(lambda q: expand_query(q["query"])))
        .assign(results=RunnableLambda(lambda q: hybrid_query(q["multi_query"])))
        .assign(
            re_ranked_results=RunnableLambda(
                lambda q: rerank_results(q["query"], q["results"])
            )
        )
        .assign(prompt_temp=RunnableLambda(lambda q: load_prompt(settings.prompt_path)))
        .assign(
            render_prompt=RunnableLambda(
                lambda q: render_resume_prompt(
                    q["prompt_temp"],
                    q["multi_query"],
                    q["re_ranked_results"],
                )
            )
        )
        .assign(output=RunnableLambda(lambda q: get_llm().invoke(q["render_prompt"])))
        .assign(answer_raw=RunnableLambda(lambda q: _extract_answer(q["output"])))
        .assign(answer_parsed=RunnableLambda(lambda q: resume_parser.parse(q["answer_raw"])))
    )


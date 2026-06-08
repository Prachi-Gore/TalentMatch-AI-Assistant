import json
import uuid
from functools import lru_cache

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

from hireflow.chains import build_retrieval_chain
from hireflow.llm import get_llm
from hireflow.schemas import ResumeResponse


_session_store: dict[str, dict] = {}
_history_store: dict[str, ChatMessageHistory] = {}


def get_history(session_id: str) -> ChatMessageHistory:
    if session_id not in _history_store:
        _history_store[session_id] = ChatMessageHistory()

    return _history_store[session_id]


qa_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an assistant answering questions about candidate resumes.\n"
            "Use ONLY the Resume JSON as ground truth.\n"
            "If unavailable say you don't know.\n\n"
            "Initial query:\n{initial_query}\n\n"
            "Resume JSON:\n{resume_json}",
        ),
        MessagesPlaceholder("history"),
        ("human", "{question}"),
    ]
)


def _prep_qa_inputs(payload: dict) -> dict:
    resume_json = payload["resume_json"]
    return {
        "question": payload["question"],
        "history": payload.get("history", []),
        "resume_json": json.dumps(resume_json, ensure_ascii=False),
        "initial_query": resume_json.get("query", ""),
    }


@lru_cache(maxsize=1)
def get_qa_agent() -> RunnableWithMessageHistory:
    resume_qa_chain = (
        RunnableLambda(_prep_qa_inputs)
        | qa_prompt
        | get_llm()
        | StrOutputParser()
    )

    return RunnableWithMessageHistory(
        resume_qa_chain,
        get_history,
        input_messages_key="question",
        history_messages_key="history",
    )


def handle_user_message(user_input: str, session_id: str | None = None) -> dict:
    if session_id is None:
        session_id = str(uuid.uuid4())

    if session_id not in _session_store:
        out = build_retrieval_chain().invoke(user_input)
        parsed: ResumeResponse = out["answer_parsed"]

        _session_store[session_id] = parsed.model_dump()
        _history_store[session_id] = ChatMessageHistory()

        resumes = _session_store[session_id].get("resumes", [])
        lines = [
            f"Parsed {len(resumes)} resumes.\n",
            "Shortlisted resumes:\n",
        ]

        for resume in resumes:
            filename = resume.get("filename") or resume["resume_id"]
            relevance = float(resume.get("jd_relevance", 0.0))
            lines.append(f"- {filename} (relevance = {relevance:.2f})")

        return {
            "session_id": session_id,
            "response": "\n".join(lines),
        }

    response = get_qa_agent().invoke(
        {
            "question": user_input,
            "resume_json": _session_store[session_id],
        },
        config={"configurable": {"session_id": session_id}},
    )

    return {
        "session_id": session_id,
        "response": response,
    }

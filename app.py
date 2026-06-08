import streamlit as st
from pathlib import Path

from hireflow.chat import handle_user_message
from hireflow.ingestion import ingest_resumes
from hireflow.config import settings


st.set_page_config(page_title="AI TalentMatch", layout="wide")


def render_sidebar() -> None:
    st.sidebar.header("Navigation")

    if st.sidebar.button("Home", use_container_width=True):
        st.session_state.page = "main"

    if st.sidebar.button("Chat", use_container_width=True):
        st.session_state.page = "chat"

    if st.sidebar.button("New Session", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.session_id = None
        st.session_state.page = "chat"
        st.rerun()


def render_upload_section() -> None:
    st.header("Add Resumes")

    files = st.file_uploader(
        "Select PDF resumes",
        type="pdf",
        accept_multiple_files=True,
    )

    if files:
        st.success(f"{len(files)} file(s) selected")

        if st.button("Process & Index", use_container_width=True):
            # Create resume directory if it doesn't exist
            settings.resume_dir.mkdir(parents=True, exist_ok=True)

            # Save uploaded files to resume_dir
            with st.spinner(f"Uploading {len(files)} file(s)..."):
                for file in files:
                    file_path = settings.resume_dir / file.name
                    file_path.write_bytes(file.read())

            st.success(f"✅ Saved {len(files)} file(s) to resume_dir/")

            # Index resumes in Pinecone
            with st.spinner("Indexing resumes in Pinecone... (this may take a minute)"):
                try:
                    count = ingest_resumes(settings.resume_dir)
                    st.success(f"✅ Successfully indexed {count} resumes!")
                    st.info(f"💡 Go to Chat tab and search for candidates matching your job description")
                except Exception as e:
                    st.error(f"❌ Error during indexing: {str(e)}")
                    st.error("Check that your OPENAI_API_KEY and PINECONE_API_KEY are set in .env")


def render_search_section() -> None:
    st.header("Search Candidates")

    with st.form("search_form"):
        job_title = st.text_input("Job Title")
        job_desc = st.text_area("Job Description")
        skills = st.text_area("Required Skills")
        top_k = st.slider("Results", 3, 10, 5)

        submitted = st.form_submit_button("Find Candidates")

    if submitted:
        query = "\n".join(
            part
            for part in [
                f"Job title: {job_title}" if job_title else "",
                f"Job description: {job_desc}" if job_desc else "",
                f"Required skills: {skills}" if skills else "",
                f"Shortlist top {top_k} candidates.",
            ]
            if part
        )
        st.session_state.pending_query = query
        st.session_state.page = "chat"
        st.rerun()


def render_chat_page() -> None:
    st.header("Chat")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    pending_query = st.session_state.pop("pending_query", None)
    if pending_query:
        run_chat_turn(pending_query)

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    user_message = st.chat_input("Ask about candidates...")

    if user_message:
        run_chat_turn(user_message)
        st.rerun()


def run_chat_turn(user_message: str) -> None:
    st.session_state.chat_history.append(
        {"role": "user", "content": user_message}
    )

    with st.spinner("Thinking..."):
        output = handle_user_message(
            user_message,
            session_id=st.session_state.get("session_id"),
        )

    st.session_state.session_id = output["session_id"]
    st.session_state.chat_history.append(
        {"role": "assistant", "content": output["response"]}
    )


def main() -> None:
    st.title("AI TalentMatch")

    if "page" not in st.session_state:
        st.session_state.page = "main"

    render_sidebar()

    if st.session_state.page == "chat":
        render_chat_page()
        return

    col1, col2 = st.columns(2)
    with col1:
        render_upload_section()
    with col2:
        render_search_section()


if __name__ == "__main__":
    main()

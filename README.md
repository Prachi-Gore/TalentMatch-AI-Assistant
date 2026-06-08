# TalentMatch

TalentMatch is an AI-powered resume shortlisting and candidate matching system built with Retrieval Augmented Generation (RAG). It helps recruiters instantly find the best-fit candidates from hundreds of resumes by understanding both semantic meaning and exact keywords.

**Live Demo:** https://talentmatch-ai-assistant.streamlit.app/

**Portfolio:** https://prachi-gore-portfolio.netlify.app/

## Features

- **Multi-Format Support** - Parses PDF, DOCX, TXT, and Markdown resumes
- **Intelligent Parsing** - Extracts skills, experience, and metadata using LLM
- **Hybrid Vector Search** - Dense embeddings (semantic) + TF-IDF (keywords)
- **Production Vector DB** - Stores hybrid vectors in Pinecone for scalability
- **Smart Reranking** - Cross-encoder improves ranking precision after retrieval
- **Structured Output** - Pydantic-validated JSON for reliable API/UI integration
- **Stateful Conversations** - Chat memory for follow-up questions without re-retrieval
- **Dual Interfaces** - Streamlit dashboard + CLI for power users

## Tech Stack

- Python
- Streamlit
- LangChain
- OpenAI
- Pinecone
- Sentence Transformers
- Scikit-learn

## Project Structure

```text
talentmatch/
  config.py            # Configuration & environment settings
  ingestion.py         # Resume ingestion, parsing, embedding pipeline
  retrieval.py         # Hybrid search, query expansion, reranking
  prompts.py           # YAML prompt templates & Pydantic parser
  schemas.py           # Pydantic data models
  chains.py            # LangChain orchestration
  chat.py              # Session management & memory
  llm.py               # LLM interface (OpenAI GPT-4)
  pinecone_client.py   # Vector database client
scripts/
  ingest_resumes.py    # CLI: Index resumes to Pinecone
  chat_cli.py          # CLI: Interactive chat
app.py                 # Streamlit web dashboard
prompt.yaml            # LLM prompt template
```

## Setup

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Add your API keys inside `.env`:

```env
OPENAI_API_KEY=your_openai_key
PINECONE_API_KEY=your_pinecone_key
```

For Pinecone serverless, the default values are:

```env
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1
```

## Run TalentMatch Dashboard

```powershell
streamlit run app.py
```

Open the local URL shown in the terminal, usually:

```text
http://localhost:8501
```

## Run CLI Chat

```powershell
python scripts/chat_cli.py
```

## How It Works

```text
Resume files
  -> text extraction
  -> LLM resume parsing
  -> dense embeddings + sparse TF-IDF
  -> Pinecone hybrid index
  -> query expansion
  -> hybrid retrieval
  -> cross-encoder reranking
  -> structured LLM response
  -> chat memory
  -> Streamlit UI
```

The first user message performs resume retrieval and shortlisting. Follow-up questions are answered from the stored shortlist for that session.

## Example Query

```text
Shortlist the top 5 Python backend developers with FastAPI, AWS, and SQL experience.
```

Follow-up:

```text
Which candidate has the strongest cloud experience?
```


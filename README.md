# 🔬 Analyst AI — Evidence-Driven Research Agent

A powerful AI research assistant built with LangGraph, LangChain, FAISS, and Streamlit. Upload documents, search the web, compare sources, generate professional reports, and manage citations — all from one interface.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Getting Started](#getting-started)
- [Configuration](#configuration)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Optional Features Implemented](#optional-features-implemented)

---

## Overview

**Analyst AI** is an agentic research assistant designed for students, researchers, and professionals who need to:

- Analyse documents quickly and accurately
- Compare multiple sources side by side
- Generate professional Word reports from research findings
- Manage and export citations in APA, MLA, or Chicago format
- Search the live web for up-to-date information

**Target users:** Students, researchers, job seekers, analysts, and anyone working with large volumes of documents.

---

## Features

| Feature | Description |
|---|---|
| 💬 **Research Chat** | LangGraph ReAct agent with RAG + web search |
| 📄 **Document Library** | Upload PDF, TXT, MD, CSV files into FAISS vector store |
| 📑 **Report Generation** | Generate and download professional `.docx` reports |
| ⚖️ **Compare Sources** | Side-by-side document comparison with agent synthesis |
| 📚 **Citation Manager** | Save, format, and export citations (APA, MLA, Chicago) |
| 📊 **Data Analytics** | Interactive charts for CSV files (distribution, bar, scatter, correlation, line) |
| 👤 **User Authentication** | Secure login with session management |
| 🌐 **Web Search** | Live web search via Tavily API with result caching |
| 🧠 **Hybrid RAG** | Combines FAISS document search + Tavily web search |

---

## Architecture

```
┌─────────────────────────────────────────┐
│              Streamlit UI               │
│  Dashboard │ Chat │ Docs │ Reports ...  │
└────────────────────┬────────────────────┘
                     │
         ┌───────────▼───────────┐
         │   LangGraph ReAct     │
         │       Agent           │
         └───────────┬───────────┘
                     │
        ┌────────────┼────────────┐
        │            │            │
   ┌────▼────┐  ┌────▼────┐  ┌───▼─────┐
   │  FAISS  │  │ Tavily  │  │  Tools  │
   │  (RAG)  │  │  Web    │  │(report, │
   │         │  │ Search  │  │citation)│
   └────┬────┘  └────┬────┘  └───┬─────┘
        │            │            │
        └────────────▼────────────┘
                     │
              ┌──────▼──────┐
              │   SQLite    │
              │  Database   │
              └─────────────┘
```

**Key components:**

- `agent/graph.py` — LangGraph ReAct agent with tool calling
- `agent/tools.py` — All agent tools (search, web, report, citations)
- `agent/prompts.py` — System prompt and agent instructions
- `rag/retriever.py` — FAISS vector store search
- `db/database.py` — SQLite for sessions, documents, reports, citations
- `app.py` — Streamlit UI

---

## Getting Started

### Prerequisites

- Python 3.11+
- An OpenAI API key
- (Optional) A Tavily API key for web search

### Installation

1. **Clone the repository:**

```bash
git clone <your-repo-url>
cd ai_research_and_analysis_agent
```

2. **Create and activate a virtual environment:**

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

3. **Install dependencies:**

```bash
pip install -r requirements.txt
```

4. **Set up environment variables:**

```bash
# Rename the example file
cp .env.example .env

# Open .env and fill in your API keys
```

5. **Run the app:**

```bash
streamlit run app.py
```

6. **Open your browser at:** `http://localhost:8501`

---

## Configuration

Rename `.env.example` to `.env` and fill in your keys:

```env
# Required
OPENAI_API_KEY=your_openai_api_key_here

# Optional — enables live web search
TAVILY_API_KEY=your_tavily_api_key_here
```

### Getting API Keys

- **OpenAI:** https://platform.openai.com/api-keys
- **Tavily (free tier available):** https://app.tavily.com

---

## Usage

### 1. Upload Documents
Go to **📄 Documents** → Upload PDF, TXT, MD, or CSV files → Click **Ingest Documents**

### 2. Research Chat
Go to **💬 Research Chat** → Ask any question about your documents or the web

**Example prompts:**
- `"Summarise the key findings from all uploaded documents"`
- `"Search the web for AI engineer salaries in Uganda"`
- `"Generate a professional research report on our findings"`
- `"Save the best web source as a citation in APA format"`

### 3. Compare Sources
Go to **⚖️ Compare** → Select documents → Enter a topic → Click **Run Comparison** → Click **Ask agent to synthesise**

### 4. Download Reports
Go to **📑 Reports** → Click **⬇️ Download** next to any report

### 5. Export Citations
Go to **📚 Citations** → Select format (APA/MLA/Chicago) → Click **⬇️ Download as .txt**

---

## Project Structure

```
ai_research_and_analysis_agent/
│
├── agent/
│   ├── __init__.py
│   ├── graph.py          # LangGraph ReAct agent
│   ├── prompts.py        # System prompt + instructions
│   └── tools.py          # Agent tools (search, report, citations)
│
├── rag/
│   ├── __init__.py
│   └── retriever.py      # FAISS vector store ingestion + search
│# 🔬 Analyst AI — Evidence-Driven Research Agent

A powerful AI research assistant built with LangGraph, LangChain, FAISS, and Streamlit. Upload documents, search the web, compare sources, generate professional reports, and manage citations — all from one interface.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Getting Started](#getting-started)
- [Configuration](#configuration)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Optional Features Implemented](#optional-features-implemented)
- [Known Limitations](#known-limitations)

---

## Overview

**Analyst AI** is an evidence-driven agentic research assistant designed for students, researchers, and professionals who need to:

- Analyse multiple documents quickly and accurately
- Compare sources side by side with AI-generated synthesis
- Generate professional downloadable Word reports from research findings
- Manage and export citations in APA, MLA, or Chicago format
- Search the live web for up-to-date information alongside uploaded documents

**Target users:** Students doing literature reviews, business analysts comparing market reports, researchers extracting insights from large document collections, and anyone working with multiple data sources at once.

**The problem it solves:** Researchers waste hours manually reading through documents, switching between tools, and struggling to cross-reference sources. Analyst AI combines RAG document search, live web search, source comparison, and report generation into a single intelligent interface.

---

## Features

| Feature | Description |
|---|---|
| 💬 **Research Chat** | LangGraph ReAct agent combining RAG + live web search |
| 📄 **Document Library** | Upload PDF, TXT, MD, CSV files into FAISS vector store |
| 📊 **Data Analytics** | Interactive charts for CSV files (distribution, bar, scatter, correlation, line) |
| 📑 **Report Generation** | Generate and download professional `.docx` reports |
| ⚖️ **Compare Sources** | Side-by-side document comparison with agent synthesis |
| 📚 **Citation Manager** | Save, format, and export citations (APA, MLA, Chicago) |
| 👤 **User Authentication** | Secure multi-user login with PBKDF2 password hashing |
| 🤖 **Multi-Model Support** | Switch between OpenAI, Anthropic, Google Gemini, and Groq |
| 🌐 **Web Search** | Live web search via Tavily API with SQLite result caching |
| 🔧 **Plugin System** | Enable/disable any of 9 agent tools from the sidebar |
| 💰 **Cost Tracking** | Real-time token usage and cost display per session |
| 🔁 **Feedback Loop** | Thumbs up/down rating on every agent response |

---

## Architecture

```
┌─────────────────────────────────────────────────┐
│                  Streamlit UI                   │
│  Dashboard │ Chat │ Docs │ Reports │ Compare ... │
└───────────────────────┬─────────────────────────┘
                        │
            ┌───────────▼───────────┐
            │    LangGraph ReAct    │
            │        Agent          │
            │   (Reason → Act loop) │
            └───────────┬───────────┘
                        │
        ┌───────────────┼───────────────┐
        │               │               │
   ┌────▼────┐     ┌────▼────┐     ┌───▼──────┐
   │  FAISS  │     │ Tavily  │     │  Tools   │
   │  (RAG)  │     │  Web    │     │ (report, │
   │ Vector  │     │ Search  │     │citations,│
   │  Store  │     │         │     │ analyse) │
   └────┬────┘     └────┬────┘     └───┬──────┘
        │               │               │
        └───────────────▼───────────────┘
                        │
                 ┌──────▼──────┐
                 │   SQLite    │
                 │  Database   │
                 │(users,      │
                 │ sessions,   │
                 │ messages,   │
                 │ reports,    │
                 │ citations)  │
                 └─────────────┘
```

**Key components:**

- `agent/graph.py` — LangGraph ReAct agent, LLM factory for all providers, token extraction
- `agent/tools.py` — All 9 agent tools with `@tool` decorator and JSON schemas
- `agent/prompts.py` — Jinja2 system prompt builder with personality and document context
- `rag/retriever.py` — FAISS vector store ingestion, embedding, and semantic search
- `db/database.py` — SQLite layer for all persistence (users, sessions, messages, documents, reports, citations, cache)
- `app.py` — Streamlit UI with all 7 tabs, sidebar, auth wall, and session management

---

## Getting Started

### Prerequisites

- Python 3.11+
- An OpenAI API key (required)
- A Tavily API key (optional — enables live web search, free tier available)

### Installation

1. **Clone the repository:**

```bash
git clone <your-repo-url>
cd ai_research_and_analysis_agent
```

2. **Create and activate a virtual environment:**

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

3. **Install dependencies:**

```bash
pip install -r requirements.txt
```

4. **Set up environment variables:**

```bash
# Rename the example file
cp .env.example .env

# Open .env and fill in your API keys
```

5. **Run the app:**

```bash
streamlit run app.py
```

6. **Open your browser at:** `http://localhost:8501`

7. **First time setup:** You will be prompted to create an admin account on first launch.

---

## Configuration

Rename `.env.example` to `.env` and fill in your keys:

```env
# Required — for document embeddings and LLM
OPENAI_API_KEY=your_openai_api_key_here

# Optional — enables live web search
TAVILY_API_KEY=your_tavily_api_key_here

# Optional — for Anthropic Claude models
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Optional — for Google Gemini models
GOOGLE_API_KEY=your_google_api_key_here

# Optional — for Groq models (llama, mixtral, gemma)
GROQ_API_KEY=your_groq_api_key_here
```

### Getting API Keys

- **OpenAI:** https://platform.openai.com/api-keys
- **Tavily (free tier available):** https://app.tavily.com
- **Anthropic:** https://console.anthropic.com
- **Google AI:** https://aistudio.google.com/apikey
- **Groq (free tier available):** https://console.groq.com

---

## Usage

### 1. Create an Account
On first launch you will be prompted to create an account. On subsequent visits, log in with your username and password.

### 2. Upload Documents
Go to **📄 Documents** → Upload PDF, TXT, MD, or CSV files → Click **Ingest Documents**

The documents will be chunked, embedded using OpenAI embeddings, and stored in the FAISS vector index so the agent can search them semantically.

### 3. Research Chat
Go to **💬 Research Chat** → Ask any question about your documents or the web

**Example prompts:**
- `"Summarise the key findings from all uploaded documents"`
- `"Search the web for AI engineer salaries in Uganda"`
- `"Compare what the different documents say about market risks"`
- `"Generate a professional research report on our findings"`
- `"Save the best web source as a citation in APA format"`
- `"Analyse the uploaded CSV and show key trends"`

### 4. Compare Sources
Go to **⚖️ Compare** → Select 2 or more documents → Enter a topic → Click **Run Comparison** → Click **Ask agent to synthesise** for AI analysis

### 5. Download Reports
Go to **📑 Reports** → Click **⬇️ Download** next to any generated report

### 6. Export Citations
Go to **📚 Citations** → Select format (APA/MLA/Chicago) → Click **⬇️ Download as .txt**

### 7. Analyse CSV Data
Upload a CSV file and go to **📄 Documents** → scroll down to the **Data Analytics Dashboard** for interactive charts

---

## Project Structure

```
ai_research_and_analysis_agent/
│
├── agent/
│   ├── __init__.py
│   ├── graph.py          # LangGraph ReAct agent + LLM factory
│   ├── prompts.py        # Jinja2 system prompt builder
│   └── tools.py          # 9 agent tools (@tool decorated)
│
├── rag/
│   ├── __init__.py
│   └── retriever.py      # FAISS vector store — ingest + search
│
├── db/
│   ├── __init__.py
│   ├── database.py       # SQLite — all persistence operations
│   ├── analyst.db        # SQLite database (auto-created on first run)
│   └── faiss_index/      # FAISS index files (auto-created on ingest)
│       ├── index.faiss   # Vector embeddings
│       └── index.pkl     # Document metadata
│
├── uploads/              # Uploaded source files (auto-created)
├── reports/              # Generated .docx reports (auto-created)
│
├── app.py                # Streamlit UI — all 7 tabs + sidebar
├── requirements.txt      # Python dependencies
├── .env.example          # Environment variable template
├── .gitignore            # Git ignore rules
└── README.md             # This file
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Frontend** | Streamlit |
| **Agent Framework** | LangGraph ReAct, LangChain |
| **LLM Providers** | OpenAI (GPT-4o, GPT-4o-mini), Anthropic (Claude), Google (Gemini), Groq (Llama, Mixtral) |
| **Vector Store** | FAISS (faiss-cpu) |
| **Embeddings** | OpenAI text-embedding-ada-002 |
| **Web Search** | Tavily API |
| **Database** | SQLite (via Python stdlib sqlite3) |
| **Document Parsing** | PyPDF2, python-docx |
| **Report Generation** | python-docx |
| **Password Security** | PBKDF2-HMAC-SHA256 (260,000 iterations — NIST standard) |
| **Templating** | Jinja2 (system prompt rendering) |

---

## Optional Features Implemented

### ✅ Easy
- **Agent personality selection** — professional, concise, or academic tone via sidebar
- **Multi-LLM provider selector** — switch between OpenAI, Anthropic, Google, Groq
- **Temperature slider** — tune model creativity from 0.0 to 1.0
- **Token usage and cost display** — live tracking per session in sidebar and footer

### ✅ Medium (10 completed)
- **Token usage and cost calculation** — per session with per-model pricing table
- **Retry logic** — exponential backoff on API failures in tools
- **Long-term memory** — SQLite persists sessions, messages, documents, citations, reports across logins
- **Short-term memory** — `st.session_state.messages` maintains conversation context within a session
- **User authentication** — multi-user login with PBKDF2-HMAC-SHA256 password hashing and legacy SHA-256 migration
- **Web search tool** — live Tavily API integration with SQLite result caching (60-minute TTL)
- **Plugin system** — users can enable/disable any of 9 agent tools dynamically from the sidebar
- **Multi-session support** — create, switch between, and resume named research sessions
- **Feedback loop** — thumbs up/down rating on every agent response stored in session state
- **Multi-model support** — unified `_build_llm()` factory in `graph.py` supports 4 providers with identical agent interface

### ✅ Hard (1 completed)
- **Agentic RAG** — FAISS semantic vector search over uploaded documents combined with live Tavily web search. The LangGraph ReAct agent dynamically decides on every turn whether to search documents, search the web, or combine both sources — based on the nature of the question.

---

## Known Limitations

- **Shared FAISS index** — the vector knowledge base is a single shared index across all users. Document records are scoped by `user_id` in SQLite and the agent system prompt only references the current user's files, but at the embedding level the index is shared. Future fix: per-user `db/faiss_index/{user_id}/` directories.
- **No rate limiting** — there is no per-user API call limit. A future improvement would be a daily token budget per user.
- **Full history sent every turn** — the complete conversation history is passed to the LLM on every message. For very long sessions this increases token costs. Future fix: conversation summarisation after N turns.
- **Local storage only** — reports and uploaded files are stored on the local filesystem. A production deployment would use cloud storage (e.g. AWS S3).
├── db/
│   ├── __init__.py
│   ├── database.py       # SQLite operations
│   ├── analyst.db        # SQLite database (auto-created)
│   └── faiss_index/      # FAISS index files (auto-created)
│
├── uploads/              # Uploaded source files (auto-created)
├── reports/              # Generated reports (auto-created)
│
├── app.py                # Streamlit UI
├── requirements.txt      # Python dependencies
├── .env.example          # Environment variable template
├── .gitignore            # Git ignore rules
└── README.md             # This file
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Frontend** | Streamlit |
| **Agent** | LangGraph ReAct, LangChain |
| **LLM** | OpenAI GPT-4o-mini |
| **Vector Store** | FAISS (faiss-cpu) |
| **Web Search** | Tavily API |
| **Database** | SQLite |
| **Document Parsing** | PyPDF, python-docx |
| **Embeddings** | OpenAI text-embedding-ada-002 |

---

## Optional Features Implemented

### ✅ Easy
- Agent personality selection (professional, friendly, concise)
- OpenAI settings exposed as sliders (temperature)
- Token usage and cost display

### ✅ Medium
- Token usage and cost calculation per session
- Retry logic for API failures
- Long-term memory via SQLite (sessions, documents, citations)
- User authentication and personalisation
- Web search via Tavily (external API tool)
- Plugin system — users can enable/disable tools in the sidebar
- Multi-session support with session history

### ✅ Hard
- **Agentic RAG** — FAISS vector search + Tavily web search combined
- LangGraph ReAct agent architecture with state management
- Human-in-the-loop via Streamlit UI interactions

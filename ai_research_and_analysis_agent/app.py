"""
app.py
------
Streamlit entry point for Analyst AI — Evidence-Driven Research Agent.

Handles UI rendering, user authentication, session management, and
coordinates between the LangGraph agent, FAISS vector store, and SQLite
database across all application tabs.

FIXES APPLIED:
  1. doc_sources is now fetched fresh inside the chat handler (race condition fix)
  2. Pre-warmer silently catches all exceptions so it never blocks startup
  3. retriever_loaded flag is reset to False after a new ingest so the
     pre-warmer runs again on the next interaction
  4. Ingest success path calls get_retriever() immediately so the index is
     hot before the user sends their first message
  5. token key names align with graph.py ("input_tokens" / "output_tokens")
  6. token_source badge tracks real vs estimated counts
  7. Provider selector covers OpenAI, Anthropic, Google, Groq
  8. Citation auto-save: fires on every [Doc:] response, uses db.database.save_citation
     directly with correct parameter names and types
  9. FIX: All auth forms (setup, login, register) now reset all user-scoped
     session state on login so previous user's chat/data never leaks to a
     new user.
"""

import os
import sys
from pathlib import Path
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv

# Constants
APP_TITLE = "Analyst AI — Research Agent"
APP_ICON = "🔬"
APP_LAYOUT = "wide"
MAX_SESSION_DISPLAY = 8
SHORT_TITLE_MAX_LEN = 50
SUPPORTED_FILE_TYPES = ["pdf", "txt", "md", "csv"]
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_TEMPERATURE = 0.3
DEFAULT_PERSONALITY = "professional"

# Setup
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env", override=True)
load_dotenv(override=True)

st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout=APP_LAYOUT,
    initial_sidebar_state="expanded",
)

from db.database import (
    init_db, get_all_documents, get_all_sessions, get_all_reports,
    create_session, get_session_messages, save_message, delete_session,
    rename_session, add_document, delete_document, clear_documents,
    delete_report, get_session_citations, delete_citation,
    clear_session_citations, format_citation,
    create_user, verify_user, update_password, update_display_name,
    get_user, user_count, save_citation as db_save_citation,
)
from agent.tools import ALL_TOOLS, TOOL_DESCRIPTIONS, CORE_TOOLS, set_agent_context
from agent.graph import run_agent
from rag.retriever import (
    ingest_documents, has_knowledge_base, clear_knowledge_base,
    get_document_sources,
)

init_db()

# Custom CSS 
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

#MainMenu, footer, header { visibility: hidden; }

.main .block-container { background: #F8FAFC; padding-top: 1rem; padding-bottom: 2rem; }

.top-banner {
    background: linear-gradient(120deg, #1e3a5f 0%, #2563EB 45%, #0EA5E9 100%);
    background-size: 250% 250%;
    animation: brandPulse 10s ease infinite;
    color: white;
    padding: 1.8rem 2.4rem;
    border-radius: 8px;
    margin-bottom: 2rem;
    display: flex;
    align-items: center;
    gap: 1.5rem;
    position: relative;
    overflow: hidden;
    box-shadow: 0 10px 40px rgba(37,99,235,0.3);
    border-left: 6px solid #F59E0B;
}
@keyframes brandPulse {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
.top-banner::before {
    content: "";
    position: absolute; inset: 0;
    background-image: linear-gradient(rgba(255,255,255,0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,0.03) 1px, transparent 1px);
    background-size: 32px 32px;
    z-index: 0;
}
.top-banner::after {
    content: "";
    position: absolute; right: -60px; top: -60px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, rgba(245,158,11,0.25) 0%, transparent 70%);
    z-index: 0;
}
.top-banner .brand-name {
    font-family: "Space Grotesk", sans-serif;
    font-size: 2.1rem; font-weight: 700;
    color: white; letter-spacing: -0.03em;
    margin: 0; line-height: 1.1; position: relative; z-index: 1;
}
.top-banner .brand-name span { color: #F59E0B; }
.top-banner .brand-tagline { font-size: 0.88rem; opacity: 0.85; margin: 0.25rem 0 0; position: relative; z-index: 1; }
.top-banner .brand-sub { font-size: 0.76rem; opacity: 0.6; margin: 0.3rem 0 0; position: relative; z-index: 1; letter-spacing: 0.03em; }
.top-banner .brand-icon { font-size: 3.2rem; position: relative; z-index: 1; filter: drop-shadow(0 0 12px rgba(245,158,11,0.5)); }

.metric-card {
    background: white; border: 1px solid #E2E8F0;
    border-top: 4px solid #F59E0B; border-radius: 8px;
    padding: 1.3rem 1rem; text-align: center;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05); transition: all 0.25s ease;
}
.metric-card:hover { transform: translateY(-4px); box-shadow: 0 10px 28px rgba(37,99,235,0.15); border-top-color: #2563EB; }
.metric-card .value {
    font-family: "Space Grotesk", sans-serif; font-size: 2.2rem; font-weight: 700;
    background: linear-gradient(135deg, #2563EB, #0EA5E9);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
}
.metric-card .label { font-size: 0.72rem; color: #64748B; margin-top: 0.3rem; text-transform: uppercase; letter-spacing: 0.09em; font-weight: 600; }

.section-header {
    font-family: "Space Grotesk", sans-serif; font-size: 1.25rem; font-weight: 700;
    color: #0F172A; margin-bottom: 1rem; padding-bottom: 0.5rem;
    border-bottom: 3px solid #2563EB; display: inline-block; letter-spacing: -0.02em;
}

.doc-card {
    background: linear-gradient(to right, #FFF7ED, white);
    border: 1px solid #FED7AA; border-left: 5px solid #F59E0B;
    border-radius: 8px; padding: 1rem 1.2rem; margin-bottom: 0.7rem;
    box-shadow: 0 2px 6px rgba(245,158,11,0.08); transition: box-shadow 0.2s;
}
.doc-card:hover { box-shadow: 0 6px 18px rgba(245,158,11,0.18); }
.report-card {
    background: linear-gradient(to right, #EFF6FF, white);
    border: 1px solid #BFDBFE; border-left: 5px solid #2563EB;
    border-radius: 8px; padding: 1rem 1.2rem; margin-bottom: 0.7rem;
    box-shadow: 0 2px 6px rgba(37,99,235,0.08);
}
.citation-card {
    background: linear-gradient(to right, #F0F9FF, white);
    border: 1px solid #BAE6FD; border-left: 5px solid #0EA5E9;
    border-radius: 8px; padding: 1rem 1.2rem; margin-bottom: 0.7rem;
    box-shadow: 0 2px 6px rgba(14,165,233,0.08);
}

.badge { display: inline-block; padding: 0.22rem 0.8rem; border-radius: 4px; font-size: 0.74rem; font-weight: 700; letter-spacing: 0.04em; text-transform: uppercase; }
.badge-green  { background: #DCFCE7; color: #15803D; }
.badge-blue   { background: #DBEAFE; color: #1D4ED8; }
.badge-orange { background: #FEF3C7; color: #B45309; }
.badge-sky    { background: #E0F2FE; color: #0369A1; }
.badge-gray   { background: #F1F5F9; color: #475569; }

.tracker-panel { background: white; border: 1px solid #E2E8F0; border-radius: 8px; padding: 1.2rem 1.5rem; box-shadow: 0 2px 8px rgba(0,0,0,0.05); }
.tracker-step { display: flex; align-items: center; gap: 0.8rem; padding: 0.6rem 0; border-bottom: 1px solid #F1F5F9; font-size: 0.9rem; font-weight: 500; color: #374151; }
.tracker-step:last-child { border-bottom: none; }
.tracker-dot { width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }
.dot-active  { background: #2563EB; box-shadow: 0 0 0 3px rgba(37,99,235,0.2); animation: pulse 1.5s ease infinite; }
.dot-done    { background: #22C55E; }
.dot-waiting { background: #CBD5E1; }
@keyframes pulse { 0%,100% { box-shadow: 0 0 0 3px rgba(37,99,235,0.2); } 50% { box-shadow: 0 0 0 6px rgba(37,99,235,0.1); } }

.tree-panel { background: #0F172A; color: #E2E8F0; border-radius: 8px; padding: 1.2rem 1.5rem; font-family: "Courier New", monospace; font-size: 0.85rem; line-height: 1.9; box-shadow: 0 4px 20px rgba(0,0,0,0.2); }
.tree-panel .tree-topic { color: #F59E0B; font-weight: 700; font-size: 0.95rem; }
.tree-panel .tree-source { color: #60A5FA; }
.tree-panel .tree-evidence { color: #34D399; font-size: 0.8rem; }

.prompt-pill { background: #EFF6FF; border: 1px solid #BFDBFE; border-left: 3px solid #2563EB; border-radius: 4px; padding: 0.6rem 1rem; margin-bottom: 0.45rem; font-size: 0.87rem; color: #1D4ED8; transition: all 0.2s; }
.prompt-pill:hover { background: #DBEAFE; border-left-color: #F59E0B; }

.step-card { text-align: center; padding: 1.4rem 1rem; background: white; border-radius: 8px; border: 1px solid #E2E8F0; border-bottom: 4px solid #F59E0B; min-height: 145px; box-shadow: 0 2px 8px rgba(0,0,0,0.04); transition: all 0.25s; }
.step-card:hover { border-bottom-color: #2563EB; transform: translateY(-4px); box-shadow: 0 12px 28px rgba(37,99,235,0.14); }

.stChatMessage { border-radius: 8px !important; }
[data-testid="stChatMessageContent"] { font-size: 0.95rem; line-height: 1.75; }

[data-testid="stSidebar"] { background: linear-gradient(180deg, #0F172A 0%, #1E293B 100%) !important; border-right: 1px solid #1E3A5F; }
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stCheckbox label,
[data-testid="stSidebar"] .stTextInput label,
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #CBD5E1 !important; }
[data-testid="stSidebar"] .stButton button { background: #1E293B; color: #94A3B8; border: 1px solid #334155; border-radius: 6px; font-size: 0.85rem; transition: all 0.2s; }
[data-testid="stSidebar"] .stButton button:hover { background: #2563EB; border-color: #2563EB; color: white; }

.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #2563EB 0%, #0EA5E9 100%) !important;
    border: none !important; border-radius: 6px !important;
    font-weight: 700 !important; letter-spacing: 0.02em !important;
    box-shadow: 0 3px 10px rgba(37,99,235,0.3) !important; transition: all 0.2s !important;
}
.stButton > button[kind="primary"]:hover { transform: translateY(-1px) !important; box-shadow: 0 8px 20px rgba(37,99,235,0.4) !important; }

.stTabs [data-baseweb="tab-list"] { gap: 0.3rem; background: #E2E8F0; padding: 0.35rem; border-radius: 6px; }
.stTabs [data-baseweb="tab"] { border-radius: 5px !important; font-weight: 600 !important; font-size: 0.86rem !important; padding: 0.4rem 0.9rem !important; color: #475569 !important; }
.stTabs [aria-selected="true"] { background: linear-gradient(135deg, #2563EB, #0EA5E9) !important; color: white !important; box-shadow: 0 2px 8px rgba(37,99,235,0.35) !important; }

.stAlert { border-radius: 6px !important; }
hr { border-color: #E2E8F0 !important; }
.stDataFrame { border-radius: 6px !important; overflow: hidden; }
.analyst-footer { background: #F8FAFC !important; border-top: 1px solid #E2E8F0 !important; }

.qa-card { border-radius: 10px; padding: 1rem 1.2rem; margin-bottom: 0.8rem; border: 1px solid; transition: all 0.2s; }
.qa-blue   { background: #EFF6FF; border-color: #BFDBFE; }
.qa-teal   { background: #F0FDFA; border-color: #99F6E4; }
.qa-orange { background: #FFF7ED; border-color: #FED7AA; }
.qa-yellow { background: #FFFBEB; border-color: #FDE68A; }
.qa-title  { font-weight: 700; font-size: 0.95rem; color: #1E293B; }
.qa-desc   { font-size: 0.8rem; color: #475569; margin-top: 0.2rem; }

.tracker-title    { font-weight: 600; font-size: 0.88rem; color: #111827; }
.tracker-subtitle { font-size: 0.76rem; color: #6B7280; margin-top: 0.1rem; }

.step-title { font-family: "Space Grotesk", sans-serif; font-weight: 700; color: #111827; margin: 0.4rem 0; font-size: 1.05rem; }
.step-desc  { font-size: 0.8rem; color: #6B7280; line-height: 1.4; }

.card-title { font-weight: 600; font-size: 1rem; color: #111827; }
.card-meta  { color: #6B7280; font-size: 0.8rem; margin-left: 0.5rem; }
</style>
""", unsafe_allow_html=True)

# Session State 
if "session_id" not in st.session_state or st.session_state.session_id is None:
    _uid = (st.session_state.get("current_user") or {}).get("id")
    # On login: resume the most recent session that has real messages.
    # New sessions are created immediately by the New Session button — no lazy create.
    _resume_id = None
    for _s in get_all_sessions(user_id=_uid):
        if get_session_messages(_s["id"]):
            _resume_id = _s["id"]
            break
    st.session_state.session_id = _resume_id  # None if no sessions with messages yet

if "messages" not in st.session_state:
    if st.session_state.session_id:
        st.session_state.messages = get_session_messages(st.session_state.session_id)
    else:
        st.session_state.messages = []

if "retriever_loaded" not in st.session_state:
    if has_knowledge_base():
        try:
            from rag.retriever import get_retriever
            get_retriever()
            st.session_state.retriever_loaded = True
        except Exception:
            st.session_state.retriever_loaded = False
    else:
        st.session_state.retriever_loaded = False

for k, v in [
    ("tokens_in", 0), ("tokens_out", 0), ("cost", 0.0), ("feedback", {}),
    ("page", "home"), ("dark_mode", False), ("logged_in", False),
    ("current_user", None), ("token_source", "—"),
]:
    if k not in st.session_state:
        st.session_state[k] = v

MODEL_COSTS: dict[str, tuple[float, float]] = {
    "gpt-4o":         (0.005,   0.015),
    "gpt-4o-mini":    (0.00015, 0.0006),
    "gpt-4-turbo":    (0.01,    0.03),
    "gpt-3.5-turbo":  (0.0005,  0.0015),
    "claude-sonnet-4-5":          (0.003,   0.015),
    "claude-haiku-4-5-20251001":  (0.00025, 0.00125),
    "claude-opus-4-5":            (0.015,   0.075),
    "gemini-1.5-flash": (0.000075, 0.0003),
    "gemini-1.5-pro":   (0.00125,  0.005),
    "gemini-2.0-flash": (0.0001,   0.0004),
    "llama-3.3-70b-versatile": (0.00059, 0.00079),
    "mixtral-8x7b-32768":      (0.00024, 0.00024),
    "gemma2-9b-it":            (0.0002,  0.0002),
}


def calc_cost(model: str, in_t: int, out_t: int) -> float:
    ci, co = MODEL_COSTS.get(model, (0.001, 0.003))
    return (in_t / 1000 * ci) + (out_t / 1000 * co)


# Helper: reset all user-scoped session state on login 
def _reset_user_session_state():
    """Clear all state that belongs to the previous user.

    Called immediately after setting logged_in=True and current_user so
    that chat history, tokens, feedback and session ID from the previous
    user never bleed into the newly logged-in user's view.
    """
    # Delete empty sessions left over from previous logins before resetting state
    _cleanup_uid = (st.session_state.get("current_user") or {}).get("id")
    for _s in get_all_sessions(user_id=_cleanup_uid):
        if not get_session_messages(_s["id"]):
            delete_session(_s["id"])
    st.session_state.messages          = []
    st.session_state.session_id        = None
    st.session_state.tokens_in         = 0
    st.session_state.tokens_out        = 0
    st.session_state.cost              = 0.0
    st.session_state.feedback          = {}
    st.session_state.retriever_loaded  = False
    st.session_state.token_source      = "—"


# Authentication Wall
if not st.session_state.logged_in:
    first_time = user_count() == 0

    st.markdown("""
    <style>
    #MainMenu, footer, header         { visibility: hidden; }
    [data-testid="stHeader"]          { display: none !important; }
    [data-testid="stToolbar"]         { display: none !important; }
    [data-testid="stDecoration"]      { display: none !important; }
    [data-testid="stStatusWidget"]    { display: none !important; }
    .stDeployButton                   { display: none !important; }
    [data-testid="collapsedControl"]  { display: none !important; }
    section[data-testid="stSidebar"]  { display: none !important; }

    html, body, .stApp,
    [data-testid="stAppViewContainer"],
    section[data-testid="stMain"], .main {
        background: linear-gradient(135deg, #1e3a5f 0%, #2563EB 55%, #0ea5e9 100%) !important;
        min-height: 100vh !important;
    }
    .main .block-container {
        background: transparent !important;
        padding-top: 4rem !important;
        max-width: 100% !important;
    }

    .auth-card {
        background: white;
        padding: 2.8rem 2.5rem 2.2rem;
        border-radius: 20px;
        box-shadow: 0 25px 60px rgba(0,0,0,0.3);
        margin-bottom: 1rem;
        animation: authFadeIn 0.65s cubic-bezier(0.16,1,0.3,1) both;
    }
    @keyframes authFadeIn {
        from { opacity: 0; transform: translateY(28px) scale(0.98); }
        to   { opacity: 1; transform: translateY(0)    scale(1);    }
    }

    .auth-brand { text-align: center; margin-bottom: 1.8rem; }
    .auth-brand .auth-icon { font-size: 3.4rem; display: block; margin-bottom: 0.5rem; filter: drop-shadow(0 4px 16px rgba(255,255,255,0.5)); }
    .auth-brand h1 { font-family: 'Space Grotesk', 'Inter', sans-serif !important; font-size: 2.2rem !important; font-weight: 800 !important; color: white !important; -webkit-text-fill-color: white !important; margin: 0 !important; letter-spacing: -0.04em; text-shadow: 0 2px 20px rgba(0,0,0,0.15); }
    .auth-brand p { color: rgba(255,255,255,0.85) !important; font-size: 0.92rem !important; margin: 0.3rem 0 0 !important; letter-spacing: 0.03em; }

    .auth-welcome { background: linear-gradient(135deg, #eff6ff, #dbeafe); border: 1px solid #bfdbfe; border-radius: 12px; padding: 0.9rem 1.2rem; margin-bottom: 1.4rem; text-align: center; }
    .auth-welcome strong { color: #1e40af; font-size: 0.9rem; }
    .auth-welcome span   { color: #3b82f6; font-size: 0.8rem; display: block; margin-top: 0.2rem; }

    .auth-card .stButton > button,
    .auth-card .stFormSubmitButton > button {
        width: 100% !important; border-radius: 10px !important; height: 46px !important;
        font-size: 0.95rem !important; font-weight: 700 !important;
        background: linear-gradient(135deg, #2563eb, #0ea5e9) !important;
        color: white !important; border: none !important; letter-spacing: 0.02em !important;
        box-shadow: 0 4px 15px rgba(37,99,235,0.35) !important; transition: all 0.2s ease !important;
        margin-top: 0.5rem !important;
    }
    .auth-card .stButton > button:hover,
    .auth-card .stFormSubmitButton > button:hover { transform: translateY(-2px) !important; box-shadow: 0 10px 28px rgba(37,99,235,0.45) !important; }

    .auth-card input { border-radius: 9px !important; border: 1.5px solid #e2e8f0 !important; font-size: 0.9rem !important; transition: border-color 0.2s !important; background: #f8fafc !important; color: #0f172a !important; }
    .auth-card input:focus { border-color: #2563eb !important; box-shadow: 0 0 0 3px rgba(37,99,235,0.12) !important; background: white !important; }
    .auth-card label, .auth-card .stTextInput label p { color: #374151 !important; font-weight: 600 !important; font-size: 0.84rem !important; }

    .auth-card .stTabs [data-baseweb="tab-list"] { background: #f1f5f9 !important; border-radius: 10px !important; padding: 0.3rem !important; margin-bottom: 1.2rem !important; gap: 0.2rem !important; }
    .auth-card .stTabs [data-baseweb="tab"] { border-radius: 8px !important; font-weight: 600 !important; font-size: 0.86rem !important; color: #64748b !important; }
    .auth-card .stTabs [aria-selected="true"] { background: linear-gradient(135deg, #2563eb, #0ea5e9) !important; color: white !important; box-shadow: 0 2px 8px rgba(37,99,235,0.3) !important; }

    .auth-footer { text-align: center; color: rgba(255,255,255,0.72); font-size: 0.8rem; margin-top: 1rem; letter-spacing: 0.04em; }
    .auth-card [data-testid="stForm"] { border: none !important; padding: 0 !important; background: transparent !important; }
    </style>
    """, unsafe_allow_html=True)

    st.write("")
    _a, auth_col, _b = st.columns([1, 2, 1])

    with auth_col:
        st.markdown('<div class="auth-card">', unsafe_allow_html=True)

        st.markdown("""
        <div class="auth-brand">
            <span class="auth-icon">🔬</span>
            <h1>Analyst AI</h1>
            <p>Evidence-Driven Research Agent</p>
        </div>
        """, unsafe_allow_html=True)

        if first_time:
            st.markdown("""
            <div class="auth-welcome">
                <strong>👋 Welcome! Set up your account to get started.</strong>
                <span>One-time setup — choose your own username and password.</span>
            </div>
            """, unsafe_allow_html=True)

            with st.form("setup_form"):
                setup_display = st.text_input("Your Name", placeholder="e.g. Joseline")
                setup_user    = st.text_input("Choose a Username", placeholder="e.g. joseline123")
                setup_pass    = st.text_input("Choose a Password", type="password", placeholder="At least 6 characters")
                setup_pass2   = st.text_input("Confirm Password",  type="password")
                if st.form_submit_button("🚀 Create Account & Sign In", use_container_width=True, type="primary"):
                    if not setup_user or not setup_pass:
                        st.error("Please fill in all fields.")
                    elif setup_pass != setup_pass2:
                        st.error("Passwords do not match.")
                    else:
                        result = create_user(setup_user, setup_pass, setup_display or setup_user)
                        if result["ok"]:
                            user = get_user(setup_user)
                            st.session_state.logged_in = True
                            st.session_state.current_user = user
                            _reset_user_session_state()  # ← FIX: clear previous user state
                            st.success(f"✅ Account created! Welcome, {user['display_name']}!")
                            st.rerun()
                        else:
                            st.error(result["error"])
        else:
            auth_tab1, auth_tab2 = st.tabs(["🔑 Sign In", "📝 Create Account"])

            with auth_tab1:
                with st.form("login_form"):
                    login_user = st.text_input("Username", placeholder="Enter your username")
                    login_pass = st.text_input("Password", type="password", placeholder="Enter your password")
                    if st.form_submit_button("🔑 Sign In", use_container_width=True, type="primary"):
                        if not login_user or not login_pass:
                            st.error("Please enter username and password.")
                        else:
                            result = verify_user(login_user, login_pass)
                            if result["ok"]:
                                st.session_state.logged_in = True
                                st.session_state.current_user = result["user"]
                                _reset_user_session_state()  # ← FIX: clear previous user state
                                st.success(f"✅ Welcome back, {result['user']['display_name']}!")
                                st.rerun()
                            else:
                                st.error(result["error"])

            with auth_tab2:
                with st.form("register_form"):
                    reg_display = st.text_input("Your Name", placeholder="e.g. Joseline")
                    reg_user    = st.text_input("Choose a Username", placeholder="Minimum 3 characters")
                    reg_pass    = st.text_input("Choose a Password", type="password", placeholder="Minimum 6 characters")
                    reg_pass2   = st.text_input("Confirm Password",  type="password")
                    if st.form_submit_button("📝 Create Account", use_container_width=True, type="primary"):
                        if not reg_user or not reg_pass:
                            st.error("Please fill in all fields.")
                        elif reg_pass != reg_pass2:
                            st.error("Passwords do not match.")
                        else:
                            result = create_user(reg_user, reg_pass, reg_display or reg_user)
                            if result["ok"]:
                                user = get_user(reg_user)
                                st.session_state.logged_in = True
                                st.session_state.current_user = user
                                _reset_user_session_state()  # ← FIX: clear previous user state
                                st.success(f"✅ Welcome, {user['display_name']}!")
                                st.rerun()
                            else:
                                st.error(result["error"])

        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="auth-footer">
            🔒 Secure · Private · Evidence-Driven
        </div>
        """, unsafe_allow_html=True)

    st.stop()

# Sidebar
with st.sidebar:
    user = st.session_state.get("current_user") or {}
    display_name = user.get("display_name", "User")
    username = user.get("username", "")
    st.markdown(f"""
        <div style='text-align:center; padding: 1rem 0 0.5rem 0;'>
            <div style='font-size:2.5rem;'>🔬</div>
            <div style='color:#E2E8F0; font-size:1.4rem; font-weight:700;'>Analyst AI</div>
            <div style='color:#94A3B8; font-size:0.8rem;'>Research & Analysis Agent</div>
            <div style='margin-top:0.8rem; background:#1E293B; border-radius:8px; padding:0.5rem 0.8rem;'>
                <span style='color:#34D399; font-size:0.8rem;'>● </span>
                <span style='color:#E2E8F0; font-size:0.85rem; font-weight:600;'>{display_name}</span>
                <div style='color:#64748B; font-size:0.75rem;'>@{username}</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    if st.button("🚪 Sign Out", use_container_width=True):
        st.session_state.logged_in = False
        st.session_state.current_user = None
        st.session_state.messages = []
        st.session_state.session_id = None
        st.rerun()

    dark_mode = st.toggle("🌙 Dark Mode", value=st.session_state.dark_mode)
    st.session_state.dark_mode = dark_mode
    st.divider()

    openai_key_env = os.getenv("OPENAI_API_KEY", "")
    if openai_key_env:
        st.success("🔑 OpenAI API Key loaded", icon="✅")
    else:
        api_key = st.text_input("🔑 OpenAI API Key", type="password",
                                help="Or add OPENAI_API_KEY to your .env file")
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key

    tavily_key_env = os.getenv("TAVILY_API_KEY", "")
    if tavily_key_env:
        st.success("🌐 Tavily API Key loaded", icon="✅")
    else:
        tavily_key = st.text_input("🌐 Tavily API Key", type="password",
                                   help="Optional — free at tavily.com")
        if tavily_key:
            os.environ["TAVILY_API_KEY"] = tavily_key

    st.divider()

    st.markdown(
        "<p style='color:#94A3B8; font-size:0.75rem; text-transform:uppercase;"
        " letter-spacing:0.1em;'>⚙️ MODEL SETTINGS</p>",
        unsafe_allow_html=True,
    )

    PROVIDER_MODELS = {
        "OpenAI":             ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"],
        "Anthropic (Claude)": ["claude-sonnet-4-5", "claude-haiku-4-5-20251001", "claude-opus-4-5"],
        "Google (Gemini)":    ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash"],
        "Groq":               ["llama-3.3-70b-versatile", "mixtral-8x7b-32768", "gemma2-9b-it"],
    }

    provider = st.selectbox("Provider", list(PROVIDER_MODELS.keys()), key="provider")
    model_name = st.selectbox("Model", PROVIDER_MODELS[provider], label_visibility="collapsed")

    if provider == "Anthropic (Claude)":
        anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
        if anthropic_key:
            st.success("🟣 Anthropic key loaded", icon="✅")
        else:
            ak = st.text_input("🟣 Anthropic API Key", type="password",
                               help="Or add ANTHROPIC_API_KEY to your .env file")
            if ak:
                os.environ["ANTHROPIC_API_KEY"] = ak
    elif provider == "Google (Gemini)":
        google_key = os.getenv("GOOGLE_API_KEY", "")
        if google_key:
            st.success("🟡 Google key loaded", icon="✅")
        else:
            gk = st.text_input("🟡 Google API Key", type="password",
                               help="Or add GOOGLE_API_KEY to your .env file")
            if gk:
                os.environ["GOOGLE_API_KEY"] = gk
    elif provider == "Groq":
        groq_key = os.getenv("GROQ_API_KEY", "")
        if groq_key:
            st.success("🟠 Groq key loaded", icon="✅")
        else:
            grk = st.text_input("🟠 Groq API Key", type="password",
                                help="Or add GROQ_API_KEY to your .env file")
            if grk:
                os.environ["GROQ_API_KEY"] = grk

    temperature = st.slider("Temperature", 0.0, 1.0, 0.3, 0.05)
    personality = st.selectbox("Agent Style", ["professional", "concise", "academic"])

    st.divider()
    st.markdown("<p style='color:#94A3B8; font-size:0.75rem; text-transform:uppercase; letter-spacing:0.1em;'>🔧 ACTIVE TOOLS</p>", unsafe_allow_html=True)
    enabled_tools = []
    for name, desc in TOOL_DESCRIPTIONS.items():
        if st.checkbox(desc, value=True, key=f"tool_{name}"):
            enabled_tools.append(name)

    st.divider()
    st.markdown("<p style='color:#94A3B8; font-size:0.75rem; text-transform:uppercase; letter-spacing:0.1em;'>📊 SESSION USAGE</p>", unsafe_allow_html=True)
    cu1, cu2 = st.columns(2)
    cu1.metric("Tokens In", f"{st.session_state.tokens_in:,}")
    cu2.metric("Tokens Out", f"{st.session_state.tokens_out:,}")
    st.metric("Cost", f"${st.session_state.cost:.4f}")
    token_src = st.session_state.get("token_source", "—")
    src_color = "🟢" if token_src == "openai_metadata" else "🟡"
    st.caption(f"{src_color} Token count: {token_src}")

    st.divider()
    st.markdown("<p style='color:#94A3B8; font-size:0.75rem; text-transform:uppercase; letter-spacing:0.1em;'>🗂️ RESEARCH SESSIONS</p>", unsafe_allow_html=True)
    if st.button("➕ New Session", use_container_width=True):
        _new_sid = create_session(
            user_id=(st.session_state.get("current_user") or {}).get("id")
        )
        st.session_state.session_id = _new_sid
        st.session_state.messages = []
        st.session_state.tokens_in = 0
        st.session_state.tokens_out = 0
        st.session_state.cost = 0.0
        st.session_state.feedback = {}
        st.rerun()

    for s in get_all_sessions(user_id=(st.session_state.get("current_user") or {}).get("id"))[:8]:
        is_active = s["id"] == st.session_state.session_id
        label = f"{'▶ ' if is_active else ''}{s['title'][:28]}"
        if st.button(label, key=f"sess_{s['id']}", use_container_width=True,
                     type="primary" if is_active else "secondary"):
            st.session_state.session_id = s["id"]
            st.session_state.messages = get_session_messages(s["id"])
            st.rerun()

# Theme CSS injection 
if st.session_state.dark_mode:
    st.markdown("""
    <style>
    html, body { background-color: #0B1120 !important; }
    .stApp { background-color: #0B1120 !important; }
    .main  { background-color: #0B1120 !important; }
    .main .block-container { background-color: #0B1120 !important; }
    section[data-testid="stMain"] { background-color: #0B1120 !important; }
    section[data-testid="stMain"] > div { background-color: #0B1120 !important; }
    [data-testid="stAppViewContainer"] { background-color: #0B1120 !important; }
    .stApp p, .stApp span, .stApp div, .stApp label, .stApp li { color: #CBD5E1 !important; }
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 { color: #F1F5F9 !important; }
    .stApp strong, .stApp b { color: #F1F5F9 !important; }
    .stMarkdown p, .stMarkdown li { color: #CBD5E1 !important; }
    .metric-card { background: #131F35 !important; border: 1px solid #1E3A5F !important; border-top: 4px solid #F59E0B !important; box-shadow: 0 4px 16px rgba(0,0,0,0.5) !important; }
    .metric-card .value { background: linear-gradient(135deg, #60A5FA, #38BDF8) !important; -webkit-background-clip: text !important; -webkit-text-fill-color: transparent !important; background-clip: text !important; }
    .metric-card .label { color: #64748B !important; }
    .doc-card      { background: linear-gradient(to right,#1C1408,#131F35) !important; border:1px solid #374151 !important; border-left:5px solid #F59E0B !important; }
    .report-card   { background: linear-gradient(to right,#0D1829,#131F35) !important; border:1px solid #1E3A5F !important; border-left:5px solid #3B82F6 !important; }
    .citation-card { background: linear-gradient(to right,#071828,#131F35) !important; border:1px solid #164E63 !important; border-left:5px solid #38BDF8 !important; }
    .section-header { color: #E2E8F0 !important; border-bottom: 3px solid #3B82F6 !important; }
    .tracker-panel    { background: #131F35 !important; border: 1px solid #1E3A5F !important; }
    .tracker-step     { border-bottom-color: #1E3A5F !important; }
    .tracker-title    { color: #E2E8F0 !important; }
    .tracker-subtitle { color: #64748B !important; }
    .prompt-pill { background: #131F35 !important; border: 1px solid #1E3A5F !important; border-left: 3px solid #3B82F6 !important; color: #93C5FD !important; }
    .step-card  { background: #131F35 !important; border: 1px solid #1E3A5F !important; border-bottom: 4px solid #F59E0B !important; }
    .step-title { color: #E2E8F0 !important; }
    .step-desc  { color: #64748B !important; }
    .qa-blue   { background: #0D1F3A !important; border-color: #1E3A5F !important; }
    .qa-teal   { background: #061F1A !important; border-color: #134E4A !important; }
    .qa-orange { background: #1C1408 !important; border-color: #78350F !important; }
    .qa-yellow { background: #1A1208 !important; border-color: #713F12 !important; }
    .qa-title  { color: #E2E8F0 !important; }
    .qa-desc   { color: #94A3B8 !important; }
    .card-title { color: #E2E8F0 !important; }
    .card-meta  { color: #64748B !important; }
    .stTabs [data-baseweb="tab-list"] { background: #131F35 !important; }
    .stTabs [data-baseweb="tab"] { color: #64748B !important; }
    .stTabs [aria-selected="true"] { background: linear-gradient(135deg,#1D4ED8,#0369A1) !important; color: #F8FAFC !important; }
    .stTextInput input, .stTextArea textarea { background: #131F35 !important; color: #E2E8F0 !important; border: 1px solid #1E3A5F !important; }
    [data-baseweb="input"], [data-baseweb="base-input"] { background: #131F35 !important; color: #E2E8F0 !important; }
    [data-baseweb="select"] > div { background: #131F35 !important; color: #E2E8F0 !important; }
    [data-testid="stSelectbox"] > div > div { background: #131F35 !important; color: #E2E8F0 !important; }
    [data-testid="stExpander"] { background: #131F35 !important; border: 1px solid #1E3A5F !important; border-radius: 8px !important; }
    [data-testid="stExpander"] summary { color: #CBD5E1 !important; }
    [data-testid="stForm"] { background: #131F35 !important; border: 1px solid #1E3A5F !important; }
    [data-testid="stVerticalBlockBorderWrapper"] { background: #131F35 !important; border-color: #1E3A5F !important; }
    [data-testid="stDataFrame"] { background: #131F35 !important; }
    [data-testid="stDataFrame"] td, [data-testid="stDataFrame"] th { color: #CBD5E1 !important; background: #131F35 !important; }
    .badge-orange { background:#431407 !important; color:#FCD34D !important; border:1px solid #78350F !important; }
    .badge-green  { background:#052E16 !important; color:#86EFAC !important; border:1px solid #14532D !important; }
    .badge-blue   { background:#0F2847 !important; color:#93C5FD !important; border:1px solid #1E3A5F !important; }
    .badge-sky    { background:#082F49 !important; color:#7DD3FC !important; border:1px solid #0C4A6E !important; }
    .badge-gray   { background:#1E293B !important; color:#94A3B8 !important; border:1px solid #334155 !important; }
    .stAlert { background: #131F35 !important; border-color: #1E3A5F !important; }
    .stAlert p, .stAlert div, .stAlert span { color: #CBD5E1 !important; }
    hr { border-color: #1E3A5F !important; }
    .analyst-footer { background: #0B1120 !important; border-top-color: #1E3A5F !important; }
    [data-testid="stMetricValue"] { color: #60A5FA !important; }
    [data-testid="stMetricLabel"] { color: #64748B !important; }
    [data-testid="stChatInput"] textarea { color: #E2E8F0 !important; background: #1E293B !important; caret-color: #E2E8F0 !important; }
    [data-testid="stChatInput"] textarea::placeholder { color: #64748B !important; }
    [data-testid="stChatInput"] { background: #1E293B !important; border: 1px solid #334155 !important; border-radius: 8px !important; }
    [data-testid="stChatInput"] > div { background: #1E293B !important; }
    [data-testid="stChatMessage"] { background: #131F35 !important; border: 1px solid #1E3A5F !important; }
    [data-testid="stChatMessage"] p, [data-testid="stChatMessage"] li, [data-testid="stChatMessage"] span, [data-testid="stChatMessage"] div { color: #E2E8F0 !important; }
    [data-testid="stChatMessage"] strong, [data-testid="stChatMessage"] b { color: #F1F5F9 !important; }
    [data-testid="stChatMessage"] code { background: #0F172A !important; color: #7DD3FC !important; }
    [data-testid="stChatMessage"] pre { background: #0F172A !important; border: 1px solid #1E3A5F !important; }
    [data-testid="stChatMessage"] pre code { color: #86EFAC !important; }
    hr { border-color: #1E3A5F !important; }
    .stApp p { color: #CBD5E1 !important; }
    .stApp li { color: #CBD5E1 !important; }
    </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <style>
    html, body, .stApp, .main, .main .block-container { background-color: #F8FAFC !important; }
    .stApp p, .stApp span, .stApp label, .stApp li { color: #1E293B !important; }
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 { color: #0F172A !important; }
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] div, [data-testid="stSidebar"] label { color: #CBD5E1 !important; }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 { color: #E2E8F0 !important; }
    [data-testid="stSidebar"] .stCheckbox label,
    [data-testid="stSidebar"] .stCheckbox label p,
    [data-testid="stSidebar"] .stCheckbox span { color: #CBD5E1 !important; }
    [data-testid="stSidebar"] [data-testid="stMetricValue"] { color: #60A5FA !important; }
    [data-testid="stSidebar"] [data-testid="stMetricLabel"] { color: #94A3B8 !important; }
    .tree-panel { color: #E2E8F0 !important; }
    .tree-panel .tree-topic    { color: #F59E0B !important; }
    .tree-panel .tree-source   { color: #60A5FA !important; }
    .tree-panel .tree-evidence { color: #34D399 !important; }
    .stTextInput input, .stTextArea textarea { background: #FFFFFF !important; color: #0F172A !important; border: 1px solid #CBD5E1 !important; border-radius: 6px !important; }
    .stTabs [data-baseweb="tab-list"] { background: #E2E8F0 !important; }
    .stTabs [data-baseweb="tab"] { color: #475569 !important; }
    .stTabs [aria-selected="true"] { background: linear-gradient(135deg,#2563EB,#0EA5E9) !important; color: white !important; }
    [data-testid="stExpander"] { background: #FFFFFF !important; border: 1px solid #E2E8F0 !important; border-radius: 8px !important; }
    [data-testid="stForm"] { background: #FFFFFF !important; border: 1px solid #E2E8F0 !important; border-radius: 8px !important; }
    </style>
    """, unsafe_allow_html=True)

# Top Banner 
_user_docs = get_all_documents(user_id=(st.session_state.get("current_user") or {}).get("id"))
# Scope doc_sources to this user only — never expose other users' files.
doc_sources = [d["filename"] for d in _user_docs]
kb_status = f"📚 {len(_user_docs)} DOC(S)" if _user_docs else "📭 No docs"
web_status = "🌐 Web ON" if os.getenv("TAVILY_API_KEY") else "🌐 Web OFF"
model_badge = f"🤖 {model_name}"

st.markdown(f"""
<div class="top-banner">
    <div class="brand-icon">🔎</div>
    <div style="position:relative; z-index:1;">
        <div class="brand-name">Analyst<span> AI</span></div>
        <div class="brand-tagline">Evidence-Driven Research Agent</div>
        <div class="brand-sub">Analyze documents &nbsp;·&nbsp; Compare sources &nbsp;·&nbsp; Generate professional reports</div>
    </div>
    <div style="margin-left:auto; display:flex; gap:0.7rem; align-items:center; flex-wrap:wrap; position:relative; z-index:1;">
        <span class="badge badge-orange">{kb_status}</span>
        <span class="badge badge-green">{web_status}</span>
        <span class="badge badge-sky">{model_badge}</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Main Tabs
tab_home, tab_chat, tab_docs, tab_reports, tab_compare, tab_citations, tab_account = st.tabs([
    "🏠 Dashboard", "💬 Research Chat", "📄 Documents",
    "📑 Reports", "⚖️ Compare", "📚 Citations", "👤 Account"
])

# TAB 0 — DASHBOARD

with tab_home:
    all_docs     = get_all_documents(user_id=(st.session_state.get("current_user") or {}).get("id"))
    all_sessions = get_all_sessions(user_id=(st.session_state.get("current_user") or {}).get("id"))
    all_reports  = get_all_reports(user_id=(st.session_state.get("current_user") or {}).get("id"))
    all_citations = []
    seen_ids = set()
    for sess in all_sessions:
        for c in get_session_citations(sess["id"]):
            if c["id"] not in seen_ids:
                all_citations.append(c)
                seen_ids.add(c["id"])

    st.markdown('<div class="section-header">📊 Overview</div>', unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns(5)
    stats = [
        ("📄", len(all_docs),      "Documents"),
        ("🗂️", len(all_sessions),  "Sessions"),
        ("📑", len(all_reports),   "Reports"),
        ("📚", len(all_citations), "Citations"),
        ("💬", sum(len(get_session_messages(s["id"])) for s in all_sessions) // 2, "Exchanges"),
    ]
    for col, (icon, val, label) in zip([c1,c2,c3,c4,c5], stats):
        col.markdown(f"""
        <div class="metric-card">
            <div style="font-size:1.5rem;">{icon}</div>
            <div class="value">{val}</div>
            <div class="label">{label}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    left, right = st.columns([3, 2])

    with left:
        st.markdown('<div class="section-header">🚀 Quick Actions</div>', unsafe_allow_html=True)
        qa1, qa2 = st.columns(2)
        actions = [
            ("💬", "Start Researching", "Go to the chat tab and start asking questions"),
            ("📄", "Upload Documents",  "Add PDFs, CSVs, or text files to your library"),
            ("📑", "View Reports",      "Download your generated research reports"),
            ("📚", "Manage Citations",  "View and export your saved references"),
        ]
        action_classes = ["qa-blue", "qa-teal", "qa-orange", "qa-yellow"]
        for i, (icon, title, desc) in enumerate(actions):
            col = qa1 if i % 2 == 0 else qa2
            with col:
                st.markdown(f"""
                <div class="qa-card {action_classes[i]}">
                    <div style="font-size:1.5rem; margin-bottom:0.3rem;">{icon}</div>
                    <div class="qa-title">{title}</div>
                    <div class="qa-desc">{desc}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-header">💡 Example Prompts</div>', unsafe_allow_html=True)
        prompts = [
            ("🔍", "Summarise the key findings from all uploaded documents"),
            ("🌐", "Search the web for AI engineer salaries in Uganda"),
            ("⚖️", "Compare what the different reports say about market risks"),
            ("📑", "Generate a professional research report on our findings"),
            ("📚", "Save the best web source as a citation in APA format"),
            ("📊", "Analyse the uploaded CSV and show key trends"),
        ]
        for icon, p in prompts:
            st.markdown(f'<div class="prompt-pill">{icon} &nbsp;{p}</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="section-header">🔄 Research Process Tracker</div>', unsafe_allow_html=True)
        tracker_steps = [
            ("dot-done",    "✅ Documents Indexed",       f"{len(all_docs)} file(s) in knowledge base"),
            ("dot-active",  "🔍 Agent Ready to Search",   "RAG + Web search enabled"),
            ("dot-waiting", "📊 Analyse & Compare",       "Upload docs and start a query"),
            ("dot-waiting", "📑 Generate Report",         "Ask agent to create a report"),
            ("dot-waiting", "📚 Export Citations",        "Save and download references"),
        ]
        st.markdown('<div class="tracker-panel">', unsafe_allow_html=True)
        for dot_class, title, subtitle in tracker_steps:
            st.markdown(f"""
            <div class="tracker-step">
                <div class="tracker-dot {dot_class}"></div>
                <div>
                    <div class="tracker-title">{title}</div>
                    <div class="tracker-subtitle">{subtitle}</div>
                </div>
            </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-header">🌳 Knowledge Graph</div>', unsafe_allow_html=True)
        _user_db_docs = get_all_documents(user_id=(st.session_state.get("current_user") or {}).get("id"))
        sources = [d["filename"] for d in _user_db_docs]
        if sources:
            tree_lines = ['<div class="tree-panel">']
            tree_lines.append('<span class="tree-topic">📌 Research Topic</span>')
            for i, src in enumerate(sources[:4]):
                connector = "├──" if i < len(sources[:4]) - 1 else "└──"
                tree_lines.append(f'<br>{connector} <span class="tree-source">📄 {src[:30]}</span>')
                tree_lines.append(f'<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└── <span class="tree-evidence">↳ Indexed & searchable</span>')
            tree_lines.append('</div>')
            st.markdown("".join(tree_lines), unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="tree-panel">
                <span class="tree-topic">📌 Research Topic</span><br>
                ├── <span class="tree-source">📄 Upload a document...</span><br>
                │&nbsp;&nbsp;&nbsp;&nbsp;└── <span class="tree-evidence">↳ Evidence will appear here</span><br>
                └── <span class="tree-source">🌐 Web sources...</span><br>
                &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└── <span class="tree-evidence">↳ Live search results</span>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-header">📑 Recent Reports</div>', unsafe_allow_html=True)
        if not all_reports:
            st.info("No reports yet. Ask the agent to generate one.")
        else:
            for r in all_reports[:3]:
                type_colors = {"research": "#1A56DB", "summary": "#10B981",
                               "comparison": "#F59E0B", "analysis": "#8B5CF6"}
                color = type_colors.get(r["report_type"], "#6B7280")
                st.markdown(f"""
                <div class="report-card">
                    <div class="card-title">📑 {r['title']}</div>
                    <div style="margin-top:0.3rem;">
                        <span style="background:{color}30; color:{color}; padding:0.1rem 0.5rem;
                              border-radius:4px; font-size:0.75rem; font-weight:700;">
                            {r['report_type'].title()}
                        </span>
                        <span class="card-meta">{r['created_at'][:10]}</span>
                    </div>
                </div>""", unsafe_allow_html=True)

    st.divider()
    st.markdown('<div class="section-header">🏗️ How It Works</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    hw1, hw2, hw3, hw4 = st.columns(4)
    steps = [
        ("1️⃣", "Upload", "Add PDFs, CSVs, or text documents to your library"),
        ("2️⃣", "Research", "Chat with the agent — it searches docs + web"),
        ("3️⃣", "Analyse", "Compare sources, extract claims, visualise data"),
        ("4️⃣", "Export", "Download reports and export citations"),
    ]
    for col, (num, title, desc) in zip([hw1,hw2,hw3,hw4], steps):
        with col:
            st.markdown(f"""
            <div class="step-card">
                <div style="font-size:2rem;">{num}</div>
                <div class="step-title">{title}</div>
                <div class="step-desc">{desc}</div>
            </div>""", unsafe_allow_html=True)

# TAB 1 — RESEARCH CHAT

with tab_chat:
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("ANTHROPIC_API_KEY") \
            and not os.getenv("GOOGLE_API_KEY") and not os.getenv("GROQ_API_KEY"):
        st.markdown("""
        <div style="background:#FEF3C7; border:1px solid #FCD34D; border-radius:10px;
                    padding:1rem 1.5rem; color:#92400E;">
            ⚠️ <strong>API Key Required</strong> — Enter an API key for your selected provider in the sidebar.
        </div>""", unsafe_allow_html=True)
        st.stop()

    # Show info banner when user has no documents uploaded
    _chat_user_docs = get_all_documents(
        user_id=(st.session_state.get("current_user") or {}).get("id")
    )
    if not _chat_user_docs:
        st.info(
            "📭 **No documents uploaded yet.** The agent will use web search and its own "
            "knowledge only. Go to the **Documents** tab to upload files to enable RAG search."
        )

    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"], avatar="🧑‍💻" if msg["role"] == "user" else "🔬"):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                if i not in st.session_state.feedback:
                    fc1, fc2, _ = st.columns([1, 1, 10])
                    with fc1:
                        if st.button("👍", key=f"up_{i}"):
                            st.session_state.feedback[i] = "👍"
                            st.rerun()
                    with fc2:
                        if st.button("👎", key=f"dn_{i}"):
                            st.session_state.feedback[i] = "👎"
                            st.rerun()
                else:
                    st.caption(f"{st.session_state.feedback[i]} Thanks for your feedback!")

    st.divider()
    st.markdown("<p style='color:#6B7280; font-size:0.85rem; font-weight:600;'>⚡ QUICK PROMPTS</p>", unsafe_allow_html=True)
    qp_cols = st.columns(4)
    quick_prompts = [
        ("🔍 Summarise Docs",   "Summarise the key findings from all uploaded documents"),
        ("🌐 Web Research",     "Search the web for the latest information on this topic"),
        ("⚖️ Compare Sources",  "Compare what the different uploaded sources say about the main topic"),
        ("📑 Generate Report",  "Generate a comprehensive research report based on our discussion so far"),
    ]
    for col, (label, prompt) in zip(qp_cols, quick_prompts):
        with col:
            if st.button(label, use_container_width=True):
                st.session_state._qp = prompt
                st.rerun()

    prefill = ""
    if hasattr(st.session_state, "_qp") and st.session_state._qp:
        prefill = st.session_state._qp
        st.session_state._qp = None

    user_input = st.chat_input("Ask anything — I'll search your documents and the web...")
    if prefill and not user_input:
        user_input = prefill

    if user_input:
        # If no session exists yet (brand new user), create one now.
        if not st.session_state.session_id:
            st.session_state.session_id = create_session(
                user_id=(st.session_state.get("current_user") or {}).get("id")
            )

        # Scope document sources to THIS user only — never show other users' files.
        _uid_for_docs = (st.session_state.get("current_user") or {}).get("id")
        _user_db_docs = get_all_documents(user_id=_uid_for_docs)
        _user_has_docs = bool(_user_db_docs)
        # Pass only filenames belonging to this user (empty list if none).
        fresh_doc_sources = [d["filename"] for d in _user_db_docs] if _user_has_docs else []

        # If this user has no documents, disable RAG tools so the agent cannot
        # search the shared FAISS index and reference other users' files.
        _active_tools = enabled_tools if _user_has_docs else [
            t for t in enabled_tools
            if t not in ("search_documents", "compare_sources", "analyse_csv")
        ]

        if not st.session_state.get("retriever_loaded") and has_knowledge_base():
            try:
                from rag.retriever import get_retriever
                get_retriever()
                st.session_state.retriever_loaded = True
            except Exception:
                pass

        st.session_state.messages.append({"role": "user", "content": user_input})
        save_message(st.session_state.session_id, "user", user_input)

        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(user_input)

        with st.chat_message("assistant", avatar="🔬"):
            with st.spinner("🔍 Researching..."):
                try:
                    try:
                        from rag.retriever import _ensure_store_loaded
                        _ensure_store_loaded()
                    except Exception:
                        pass

                    set_agent_context(
                        user_id=(st.session_state.get("current_user") or {}).get("id"),
                        session_id=st.session_state.session_id,
                    )
                    response, usage = run_agent(
                        user_message=user_input,
                        history=st.session_state.messages[:-1],
                        model_name=model_name,
                        temperature=temperature,
                        enabled_tools=_active_tools,
                        personality=personality,
                        document_sources=fresh_doc_sources,
                    )
                    st.markdown(response)

                    # AUTO-SAVE CITATIONS 
                    if "[Doc:" in response:
                        import re as _re
                        _clean_response = _re.sub(r'\*{1,2}', '', response)
                        _cite_lines = _re.findall(
                            r'([A-Za-z][^:\n]{2,80}):\s*\[Doc:\s*([^\]]+)\]',
                            _clean_response,
                        )
                        _saved = 0
                        _errors = []
                        for _raw_title, _src in _cite_lines:
                            _title = _raw_title.strip().lstrip("•-* 0123456789.")
                            _src = _src.strip()
                            if _title and _saved < 10:
                                try:
                                    db_save_citation(
                                        session_id=st.session_state.session_id,
                                        source_type="document",
                                        title=_title,
                                        authors="",
                                        year=str(datetime.now().year),
                                        url="",
                                        publisher=_src,
                                        page_numbers="",
                                        citation_format="APA",
                                    )
                                    _saved += 1
                                except Exception as _e:
                                    _errors.append(str(_e))
                        if _saved > 0:
                            st.caption(f"📚 {_saved} citation(s) auto-saved to Citations tab")
                        elif _cite_lines and _errors:
                            st.warning(f"⚠️ Citation save failed: {_errors[0]}")
                        elif not _cite_lines:
                            st.caption(f"ℹ️ [Doc:] found but no title:source pattern matched")

                    in_t  = usage.get("input_tokens", 0)
                    out_t = usage.get("output_tokens", 0)
                    st.session_state.tokens_in  += in_t
                    st.session_state.tokens_out += out_t
                    st.session_state.cost       += calc_cost(model_name, in_t, out_t)
                    st.session_state["token_source"] = usage.get("source", "character_estimate")
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    save_message(st.session_state.session_id, "assistant", response)
                    if len(st.session_state.messages) == 2:
                        short_title = user_input[:50] + ("..." if len(user_input) > 50 else "")
                        rename_session(st.session_state.session_id, short_title)
                except ValueError as e:
                    err = f"❌ Configuration error: {str(e)}"
                    st.error(err)
                    st.session_state.messages.append({"role": "assistant", "content": err})
                except RuntimeError as e:
                    err = f"❌ Agent error: {str(e)}"
                    st.error(err)
                    st.session_state.messages.append({"role": "assistant", "content": err})
                except Exception as e:
                    err = f"❌ Unexpected error: {str(e)}"
                    st.error(err)
                    st.session_state.messages.append({"role": "assistant", "content": err})
        st.rerun()

# TAB 2 — DOCUMENTS

with tab_docs:
    st.markdown('<div class="section-header">📄 Document Library</div>', unsafe_allow_html=True)

    if not os.getenv("OPENAI_API_KEY"):
        st.warning("⚠️ OpenAI API key required for document ingestion.")
    else:
        up_col, stat_col = st.columns([3, 1])
        with up_col:
            uploaded = st.file_uploader(
                "Upload documents (PDF, TXT, MD, CSV)",
                type=["pdf", "txt", "md", "csv"],
                accept_multiple_files=True,
            )
            b1, b2 = st.columns(2)
            with b1:
                if st.button("📥 Ingest Documents", use_container_width=True,
                             disabled=not uploaded, type="primary"):
                    with st.spinner("⚙️ Processing documents..."):
                        upload_dir = ROOT / "uploads"
                        upload_dir.mkdir(exist_ok=True)
                        paths = []
                        for uf in uploaded:
                            dest = upload_dir / uf.name
                            dest.write_bytes(uf.read())
                            paths.append(str(dest))
                        try:
                            result = ingest_documents(paths)
                            for uf in uploaded:
                                dest = ROOT / "uploads" / uf.name
                                add_document(
                                    filename=uf.name,
                                    file_type=uf.name.split(".")[-1].upper(),
                                    size_bytes=dest.stat().st_size if dest.exists() else 0,
                                    chunk_count=result["chunks"] // max(len(uploaded), 1),
                                    user_id=(st.session_state.get("current_user") or {}).get("id"),
                                )
                            try:
                                from rag.retriever import get_retriever
                                get_retriever()
                                st.session_state.retriever_loaded = True
                            except Exception:
                                st.session_state.retriever_loaded = False

                            st.success(
                                f"✅ {result['files']} file(s) ingested — "
                                f"{result['chunks']} chunks indexed. "
                                "You can now ask questions immediately!"
                            )
                        except FileNotFoundError as e:
                            st.error(f"❌ File not found: {e}")
                        except ValueError as e:
                            st.error(f"❌ Invalid file or API key: {e}")
                        except RuntimeError as e:
                            st.error(f"❌ Ingestion failed: {e}")
                        except OSError as e:
                            st.error(f"❌ File system error: {e}")
            with b2:
                if st.button("🗑️ Clear All", use_container_width=True):
                    clear_knowledge_base()
                    clear_documents()
                    st.session_state.retriever_loaded = False
                    st.success("Knowledge base cleared.")
                    st.rerun()

        with stat_col:
            kb_active = has_knowledge_base()
            st.markdown(f"""
            <div class="metric-card" style="margin-top:1.5rem;">
                <div style="font-size:1.5rem;">{'✅' if kb_active else '⚠️'}</div>
                <div class="value" style="font-size:1.1rem;">{'Active' if kb_active else 'Empty'}</div>
                <div class="label">Knowledge Base</div>
            </div>""", unsafe_allow_html=True)
            _user_doc_count = len(get_all_documents(user_id=(st.session_state.get("current_user") or {}).get("id")))
            st.markdown(f"""
            <div class="metric-card" style="margin-top:0.8rem;">
                <div class="value">{_user_doc_count}</div>
                <div class="label">Sources Indexed</div>
            </div>""", unsafe_allow_html=True)

    st.divider()
    docs = get_all_documents(user_id=(st.session_state.get("current_user") or {}).get("id"))
    if not docs:
        st.info("📭 No documents yet. Upload files above to get started.")
    else:
        st.markdown(f'<div class="section-header">📚 Library — {len(docs)} document(s)</div>', unsafe_allow_html=True)
        for doc in docs:
            icon = {"PDF": "📕", "CSV": "📊", "TXT": "📝", "MD": "📝"}.get(doc["file_type"], "📄")
            size_kb = doc["size_bytes"] / 1024
            dc1, dc2 = st.columns([5, 1])
            with dc1:
                st.markdown(f"""
                <div class="doc-card">
                    <div class="card-title">{icon} {doc['filename']}</div>
                    <div class="card-meta" style="margin-top:0.3rem;">
                        <span class="badge badge-blue">{doc['file_type']}</span>
                        &nbsp;{size_kb:.1f} KB &nbsp;·&nbsp; {doc['chunk_count']} chunks
                        &nbsp;·&nbsp; Indexed {doc['ingested_at'][:10]}
                    </div>
                </div>""", unsafe_allow_html=True)
            with dc2:
                if st.button("🗑️ Remove", key=f"del_doc_{doc['id']}", use_container_width=True):
                    delete_document(doc["filename"])
                    st.rerun()

    st.divider()
    st.markdown('<div class="section-header">📊 Data Analytics Dashboard</div>', unsafe_allow_html=True)
    csv_docs = [d for d in docs if d["file_type"] == "CSV"] if docs else []

    if not csv_docs:
        st.info("Upload a CSV file to see interactive charts and analytics.")
    else:
        import pandas as pd
        sel = st.selectbox("Choose CSV to visualise", [d["filename"] for d in csv_docs])
        csv_path = ROOT / "uploads" / sel

        if csv_path.exists():
            df = pd.read_csv(str(csv_path))
            numeric_cols = df.select_dtypes(include="number").columns.tolist()
            cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

            m1, m2, m3, m4 = st.columns(4)
            for col, (icon, val, label) in zip([m1,m2,m3,m4], [
                ("📋", f"{len(df):,}", "Total Rows"),
                ("📐", len(df.columns), "Columns"),
                ("🔢", len(numeric_cols), "Numeric"),
                ("🔤", len(cat_cols), "Text Cols"),
            ]):
                col.markdown(f"""
                <div class="metric-card">
                    <div style="font-size:1.3rem;">{icon}</div>
                    <div class="value">{val}</div>
                    <div class="label">{label}</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            with st.expander("🗃️ Raw Data Preview"):
                st.dataframe(df.head(50), use_container_width=True)

            chart_tabs = st.tabs(["📈 Distribution", "📊 Bar Chart", "🔵 Scatter", "🌡️ Correlation", "📉 Line Chart"])

            with chart_tabs[0]:
                if numeric_cols:
                    dist_col = st.selectbox("Column", numeric_cols, key="dist_col")
                    hist_data = df[dist_col].dropna()
                    if len(hist_data) <= 2:
                        bins = len(hist_data)
                    else:
                        _max_bins = min(100, len(hist_data) - 1)
                        _def_bins = min(20, _max_bins)
                        bins = st.slider("Bins", 2, _max_bins, _def_bins, key="dist_bins")
                    hist_df = pd.DataFrame(
                        {"Count": pd.cut(hist_data, bins=bins).value_counts().sort_index().values},
                        index=[str(i) for i in pd.cut(hist_data, bins=bins).value_counts().sort_index().index],
                    )
                    st.bar_chart(hist_df, use_container_width=True)
                    st.caption(f"**{dist_col}** — mean: {hist_data.mean():.2f}, std: {hist_data.std():.2f}")
                else:
                    st.info("No numeric columns.")

            with chart_tabs[1]:
                if cat_cols and numeric_cols:
                    bc1, bc2 = st.columns(2)
                    bar_cat = bc1.selectbox("Category", cat_cols, key="bar_cat")
                    bar_num = bc2.selectbox("Value", numeric_cols, key="bar_num")
                    _unique_n = len(df[bar_cat].unique())
                    if _unique_n <= 1:
                        top_n = _unique_n
                    else:
                        top_n = st.slider("Top N", 1, _unique_n, min(10, _unique_n), key="bar_topn")
                    st.bar_chart(df.groupby(bar_cat)[bar_num].mean().nlargest(top_n), use_container_width=True)
                elif numeric_cols:
                    st.bar_chart(df[numeric_cols].mean(), use_container_width=True)
                else:
                    st.info("Need at least one numeric column.")

            with chart_tabs[2]:
                if len(numeric_cols) >= 2:
                    sc1, sc2 = st.columns(2)
                    x_col = sc1.selectbox("X axis", numeric_cols, index=0, key="scatter_x")
                    y_col = sc2.selectbox("Y axis", numeric_cols, index=1, key="scatter_y")
                    scatter_df = df[[x_col, y_col]].dropna().head(500)
                    st.scatter_chart(scatter_df, x=x_col, y=y_col, use_container_width=True)
                    st.caption(f"Correlation: **{scatter_df[x_col].corr(scatter_df[y_col]):.3f}**")
                else:
                    st.info("Need 2+ numeric columns.")

            with chart_tabs[3]:
                if len(numeric_cols) >= 2:
                    corr = df[numeric_cols].corr().round(2)
                    def color_corr(val):
                        if val == 1.0:
                            return "background-color: #1A56DB; color: white"
                        elif val > 0.5:
                            return "background-color: #86EFAC; color: black"
                        elif val < -0.5:
                            return "background-color: #FCA5A5; color: black"
                        else:
                            return "background-color: #F1F5F9; color: black"
                    st.dataframe(corr.style.map(color_corr), use_container_width=True)
                    pairs = sorted(
                        [(numeric_cols[i], numeric_cols[j], corr.iloc[i, j])
                         for i in range(len(numeric_cols))
                         for j in range(i + 1, len(numeric_cols))],
                        key=lambda x: abs(x[2]), reverse=True,
                    )
                    st.markdown("**Strongest correlations:**")
                    for a, b, v in pairs[:5]:
                        em = "🟢" if v > 0.5 else "🔴" if v < -0.5 else "🟡"
                        st.caption(f"{em} **{a}** ↔ **{b}**: {v:.3f}")
                else:
                    st.info("Need 2+ numeric columns.")

            with chart_tabs[4]:
                if numeric_cols:
                    line_cols = st.multiselect(
                        "Columns", numeric_cols,
                        default=numeric_cols[:min(3, len(numeric_cols))],
                        key="line_cols",
                    )
                    if line_cols:
                        if len(df) <= 1:
                            max_rows = len(df)
                        else:
                            max_rows = st.slider("Max rows", 1, len(df), min(len(df), 200), key="line_rows")
                        st.line_chart(df[line_cols].head(max_rows), use_container_width=True)
                else:
                    st.info("No numeric columns.")


# TAB 3 — REPORTS

with tab_reports:
    st.markdown('<div class="section-header">📑 Research Reports</div>', unsafe_allow_html=True)

    reports = get_all_reports(user_id=(st.session_state.get("current_user") or {}).get("id"))
    if not reports:
        st.markdown("""
        <div style="background:#F0FFF4; border:1px solid #86EFAC; border-radius:10px; padding:1.5rem; text-align:center;">
            <div style="font-size:2rem;">📑</div>
            <div style="font-weight:600; color:#166534; margin:0.5rem 0;">No reports yet</div>
            <div style="color:#15803D; font-size:0.9rem;">In the chat, ask: <em>"Generate a research report on our findings"</em></div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f'<p style="color:#6B7280;">{len(reports)} report(s) generated</p>', unsafe_allow_html=True)
        for r in reports:
            file_path = ROOT / "reports" / Path(r["file_path"]).name
            type_colors = {"research": "#1A56DB", "summary": "#10B981",
                           "comparison": "#F59E0B", "analysis": "#8B5CF6"}
            color = type_colors.get(r["report_type"], "#6B7280")
            type_icons = {"research": "🔬", "summary": "📋", "comparison": "⚖️", "analysis": "📊"}
            t_icon = type_icons.get(r["report_type"], "📄")

            with st.container(border=True):
                rc1, rc2, rc3 = st.columns([4, 2, 1])
                with rc1:
                    st.markdown(f"""
                    <div style="font-weight:700; font-size:1rem; margin-bottom:0.3rem;">{t_icon} {r['title']}</div>
                    <span style="background:{color}15; color:{color}; padding:0.2rem 0.6rem;
                          border-radius:999px; font-size:0.75rem; font-weight:600;">
                        {r['report_type'].title()}
                    </span>
                    <span style="color:#6B7280; font-size:0.8rem; margin-left:0.8rem;">
                        📅 {r['created_at'][:16].replace('T', ' ')}
                    </span>""", unsafe_allow_html=True)
                with rc2:
                    st.caption(f"📁 {file_path.name}")
                with rc3:
                    if file_path.exists():
                        with open(str(file_path), "rb") as f:
                            st.download_button(
                                "⬇️ Download",
                                data=f.read(),
                                file_name=file_path.name,
                                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                                key=f"dl_{r['id']}",
                                use_container_width=True,
                                type="primary",
                            )
                    else:
                        st.warning("File missing")
                    if st.button("🗑️", key=f"del_rep_{r['id']}", use_container_width=True):
                        delete_report(r["id"])
                        if file_path.exists():
                            file_path.unlink()
                        st.rerun()

# TAB 4 — COMPARE SOURCES

with tab_compare:
    st.markdown('<div class="section-header">⚖️ Compare Sources</div>', unsafe_allow_html=True)
    st.markdown("Select documents and enter a topic to see a structured side-by-side comparison.")

    _cmp_db_docs = get_all_documents(user_id=(st.session_state.get("current_user") or {}).get("id"))
    sources = [d["filename"] for d in _cmp_db_docs]
    if not sources:
        st.markdown("""
        <div style="background:#EFF6FF; border:1px solid #BFDBFE; border-radius:10px;
                    padding:1.5rem; text-align:center;">
            <div style="font-size:2rem;">📭</div>
            <div style="font-weight:600; color:#1E40AF;">No documents loaded</div>
            <div style="color:#3B82F6; font-size:0.9rem;">Upload documents in the Documents tab first.</div>
        </div>""", unsafe_allow_html=True)
    else:
        cmp1, cmp2 = st.columns([3, 1])
        with cmp1:
            compare_topic = st.text_input(
                "Topic to compare",
                placeholder="e.g. revenue growth, market risks, key recommendations...",
            )
            selected_sources = st.multiselect(
                "Sources to compare", options=sources,
                default=sources[:min(3, len(sources))],
            )
        with cmp2:
            st.markdown("**Available sources:**")
            _src_bg     = "#0D1F3A" if st.session_state.dark_mode else "#F0F7FF"
            _src_border = "#1E3A5F" if st.session_state.dark_mode else "#BFDBFE"
            _src_color  = "#93C5FD" if st.session_state.dark_mode else "#1E40AF"
            for s in sources:
                st.markdown(f"""
                <div style="background:{_src_bg}; border:1px solid {_src_border}; border-radius:6px;
                            padding:0.4rem 0.8rem; margin-bottom:0.3rem; font-size:0.85rem; color:{_src_color};">
                    📄 {s}
                </div>""", unsafe_allow_html=True)

        if st.button("⚖️ Run Comparison", type="primary",
                     disabled=not compare_topic or len(selected_sources) < 1):
            with st.spinner("Comparing sources..."):
                from rag.retriever import search_documents as _search
                all_content = {s: _search(compare_topic, k=4, source_filter=s)
                               for s in selected_sources}
                all_content = {k: v for k, v in all_content.items() if v}
                st.session_state["cmp_content"] = all_content
                st.session_state["cmp_topic"] = compare_topic
                st.session_state["cmp_synthesis"] = None

        if st.session_state.get("cmp_content"):
            all_content = st.session_state["cmp_content"]
            compare_topic_display = st.session_state.get("cmp_topic", "")

            st.markdown(f'<div class="section-header">Results: "{compare_topic_display}"</div>', unsafe_allow_html=True)

            if not all_content:
                st.warning("No relevant content found for this topic.")
            else:
                cols = st.columns(len(all_content))
                for col, (source, results) in zip(cols, all_content.items()):
                    with col:
                        with st.container(border=True):
                            st.markdown(f"**📄 {source}**")
                            for r in results[:3]:
                                page = f"p.{r['page']}" if r.get("page") else ""
                                relevance = max(0, 1 - r['score'])
                                st.markdown(f"""
                                <span class="badge badge-blue">{page}</span>
                                <span class="badge badge-green">{relevance:.0%} relevant</span>
                                """, unsafe_allow_html=True)
                                st.markdown(r["content"][:400])
                                st.divider()

                synthesis_prompt = (
                    f"Compare the sources {', '.join(all_content.keys())} on '{compare_topic_display}'. "
                    "Identify agreements, contradictions, and unique insights. Give your own conclusion."
                )

                if st.button("🤖 Ask agent to synthesise", type="secondary"):
                    with st.spinner("🔍 Synthesising..."):
                        fresh_doc_sources = get_document_sources()
                        try:
                            set_agent_context(
                                user_id=(st.session_state.get("current_user") or {}).get("id"),
                                session_id=st.session_state.session_id,
                            )
                            response, usage = run_agent(
                                user_message=synthesis_prompt,
                                history=st.session_state.messages,
                                model_name=model_name,
                                temperature=temperature,
                                enabled_tools=enabled_tools,
                                personality=personality,
                                document_sources=fresh_doc_sources,
                            )
                            in_t  = usage.get("input_tokens", 0)
                            out_t = usage.get("output_tokens", 0)
                            st.session_state.tokens_in  += in_t
                            st.session_state.tokens_out += out_t
                            st.session_state.cost       += calc_cost(model_name, in_t, out_t)
                            st.session_state["token_source"] = usage.get("source", "character_estimate")
                            st.session_state.messages.append({"role": "user", "content": synthesis_prompt})
                            st.session_state.messages.append({"role": "assistant", "content": response})
                            save_message(st.session_state.session_id, "user", synthesis_prompt)
                            save_message(st.session_state.session_id, "assistant", response)
                            st.session_state["cmp_synthesis"] = response
                        except ValueError as e:
                            st.error(f"❌ Configuration error: {str(e)}")
                        except RuntimeError as e:
                            st.error(f"❌ Agent error: {str(e)}")
                        except Exception as e:
                            st.error(f"❌ Unexpected error: {str(e)}")

                if st.session_state.get("cmp_synthesis"):
                    st.divider()
                    st.markdown("### 🤖 Agent Synthesis")
                    st.markdown(st.session_state["cmp_synthesis"])

        with st.expander("💡 Tips for effective comparisons"):
            st.markdown("""
**Good topics:** `key findings` · `risks` · `recommendations` · `methodology` · `revenue`

**Workflow:** Upload 2+ docs → Enter topic → View side-by-side → Ask agent to synthesise → Generate report
            """)

# TAB 5 — CITATIONS

with tab_citations:
    st.markdown('<div class="section-header">📚 Citation Manager</div>', unsafe_allow_html=True)
    st.markdown("All sources referenced during your research. Export in APA, MLA, or Chicago format.")

    all_sessions_list = get_all_sessions(user_id=(st.session_state.get("current_user") or {}).get("id"))
    citations = []
    seen_ids = set()
    for sess in all_sessions_list:
        for c in get_session_citations(sess["id"]):
            if c["id"] not in seen_ids:
                citations.append(c)
                seen_ids.add(c["id"])

    ctrl1, ctrl2, ctrl3 = st.columns([2, 2, 2])
    with ctrl1:
        cite_format = st.selectbox("Citation Format", ["APA", "MLA", "Chicago"], key="cite_fmt")
    with ctrl2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="value">{len(citations)}</div>
            <div class="label">Total Citations</div>
        </div>""", unsafe_allow_html=True)
    with ctrl3:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🗑️ Clear All Citations", use_container_width=True):
            for sess in all_sessions_list:
                clear_session_citations(sess["id"])
            st.rerun()

    st.divider()

    if not citations:
        st.markdown("""
        <div style="background:#FFFBEB; border:1px solid #FCD34D; border-radius:10px;
                    padding:1.5rem; text-align:center;">
            <div style="font-size:2rem;">📚</div>
            <div style="font-weight:600; color:#92400E; margin:0.5rem 0;">No citations yet</div>
            <div style="color:#B45309; font-size:0.9rem;">
                Ask the agent: <em>"Save this source as a citation"</em> or add one manually below.
            </div>
        </div>""", unsafe_allow_html=True)
    else:
        by_type = {}
        for c in citations:
            by_type.setdefault(c["source_type"], []).append(c)

        type_icons = {"web": "🌐", "document": "📄", "book": "📕", "journal": "📰", "report": "📊"}
        all_formatted = []

        for source_type, items in by_type.items():
            icon = type_icons.get(source_type, "📄")
            st.markdown(
                f'<div class="section-header">{icon} {source_type.title()} Sources ({len(items)})</div>',
                unsafe_allow_html=True,
            )

            for c in items:
                import re
                c_copy = {**c, "citation_format": cite_format}
                formatted = format_citation(c_copy)
                all_formatted.append(formatted)
                clean_formatted = re.sub(r'<[^>]+>', '', formatted).replace("*", "").strip()

                cc1, cc2 = st.columns([5, 1])
                with cc1:
                    st.markdown(f"""
                    <div class="citation-card">
                        <div style="font-size:0.92rem; color:#111827;">{clean_formatted}</div>
                        {f'<div style="margin-top:0.4rem;"><a href="{c["url"]}" style="color:#1A56DB; font-size:0.8rem;">🔗 {c["url"]}</a></div>' if c.get("url") else ""}
                    </div>""", unsafe_allow_html=True)
                    st.caption(f"Saved: {c['created_at'][:16].replace('T', ' ')}")
                with cc2:
                    if st.button("🗑️", key=f"del_cite_{c['id']}", help="Remove"):
                        delete_citation(c["id"])
                        st.rerun()

        st.divider()
        st.markdown('<div class="section-header">📋 Export Reference List</div>', unsafe_allow_html=True)
        export_text = f"References ({cite_format} Format)\n{'='*40}\n\n"
        for i, s in enumerate(all_formatted, 1):
            export_text += f"[{i}] {s.replace('*','')}\n\n"

        st.text_area("Copy citation list", export_text, height=200)
        st.download_button(
            "⬇️ Download as .txt",
            data=export_text,
            file_name=f"citations_{cite_format.lower()}_{datetime.now().strftime('%Y%m%d')}.txt",
            mime="text/plain",
            use_container_width=True,
        )

    st.divider()
    with st.expander("➕ Add Citation Manually"):
        with st.form("manual_citation"):
            mc1, mc2 = st.columns(2)
            with mc1:
                m_title = st.text_input("Title *")
                m_authors = st.text_input("Author(s)")
                m_year = st.text_input("Year")
            with mc2:
                m_type = st.selectbox("Source Type", ["web", "document", "book", "journal", "report"])
                m_url = st.text_input("URL (if web)")
                m_publisher = st.text_input("Publisher / Journal")
            m_pages = st.text_input("Page numbers (optional)")
            if st.form_submit_button("💾 Save Citation", use_container_width=True, type="primary") and m_title:
                db_save_citation(
                    session_id=st.session_state.session_id,
                    source_type=m_type,
                    title=m_title,
                    authors=m_authors,
                    year=m_year,
                    url=m_url,
                    publisher=m_publisher,
                    page_numbers=m_pages,
                    citation_format=cite_format,
                )
                st.success(f"✅ Citation saved: {m_title}")
                st.rerun()

# TAB 6 — ACCOUNT SETTINGS

with tab_account:
    st.markdown('<div class="section-header">👤 Account Settings</div>', unsafe_allow_html=True)

    user = st.session_state.get("current_user") or {}
    acct1, acct2 = st.columns([3, 2])

    with acct1:
        joined = user.get("created_at", "")[:10]
        last_login = (user.get("last_login") or "")[:16].replace("T", " ") or "First login"
        st.markdown(f"""
        <div style="background:white; border:1px solid #E5E7EB; border-radius:16px;
                    padding:2rem; margin-bottom:1.5rem;
                    border-top:5px solid #1A56DB; box-shadow:0 2px 10px rgba(0,0,0,0.06);">
            <div style="font-size:3.5rem; text-align:center;">👤</div>
            <div style="text-align:center; margin-top:0.8rem;">
                <div style="font-size:1.4rem; font-weight:700; color:#111827;">{user.get('display_name', 'User')}</div>
                <div style="color:#6B7280; font-size:0.9rem;">@{user.get('username', '')}</div>
            </div>
            <hr style="margin:1.2rem 0; border-color:#E5E7EB;">
            <div style="display:flex; justify-content:space-between; font-size:0.85rem; color:#6B7280;">
                <span>📅 Joined: <strong>{joined}</strong></span>
                <span>🕐 Last login: <strong>{last_login}</strong></span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-header">✏️ Update Display Name</div>', unsafe_allow_html=True)
        with st.form("update_name_form"):
            new_display = st.text_input("New Display Name", value=user.get("display_name", ""),
                                        placeholder="How you want to be addressed")
            if st.form_submit_button("💾 Save Name", use_container_width=True, type="primary"):
                if new_display.strip():
                    update_display_name(user["username"], new_display.strip())
                    st.session_state.current_user["display_name"] = new_display.strip()
                    st.success("✅ Display name updated!")
                    st.rerun()
                else:
                    st.error("Name cannot be empty.")

        st.divider()

        st.markdown('<div class="section-header">🔒 Change Password</div>', unsafe_allow_html=True)
        with st.form("change_password_form"):
            current_pass = st.text_input("Current Password", type="password")
            new_pass = st.text_input("New Password", type="password", placeholder="Minimum 6 characters")
            new_pass2 = st.text_input("Confirm New Password", type="password")
            if st.form_submit_button("🔒 Update Password", use_container_width=True, type="primary"):
                if not current_pass or not new_pass:
                    st.error("Please fill in all fields.")
                elif new_pass != new_pass2:
                    st.error("New passwords do not match.")
                else:
                    check = verify_user(user["username"], current_pass)
                    if not check["ok"]:
                        st.error("Current password is incorrect.")
                    else:
                        result = update_password(user["username"], new_pass)
                        if result["ok"]:
                            st.success("✅ Password updated successfully!")
                        else:
                            st.error(result["error"])

    with acct2:
        st.markdown('<div class="section-header">📊 Your Stats</div>', unsafe_allow_html=True)
        all_sessions = get_all_sessions(user_id=(st.session_state.get("current_user") or {}).get("id"))
        all_docs = get_all_documents(user_id=(st.session_state.get("current_user") or {}).get("id"))
        all_reports = get_all_reports(user_id=(st.session_state.get("current_user") or {}).get("id"))
        all_cites = get_session_citations(st.session_state.session_id)

        for icon, val, label in [
            ("🗂️", len(all_sessions), "Research Sessions"),
            ("📄", len(all_docs), "Documents Ingested"),
            ("📑", len(all_reports), "Reports Generated"),
            ("📚", len(all_cites), "Citations Saved"),
            ("💬", len(st.session_state.messages) // 2, "Conversations"),
            ("💰", f"${st.session_state.cost:.4f}", "Session Cost"),
        ]:
            st.markdown(f"""
            <div style="background:white; border:1px solid #E5E7EB; border-radius:10px;
                        padding:0.8rem 1rem; margin-bottom:0.5rem; display:flex;
                        align-items:center; gap:1rem; box-shadow:0 1px 4px rgba(0,0,0,0.05);">
                <span style="font-size:1.5rem;">{icon}</span>
                <div>
                    <div style="font-weight:700; font-size:1.1rem; color:#1A56DB;">{val}</div>
                    <div style="font-size:0.75rem; color:#6B7280; text-transform:uppercase;
                                letter-spacing:0.05em;">{label}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.divider()
        st.markdown('<div class="section-header">🚪 Sign Out</div>', unsafe_allow_html=True)
        st.markdown(
            f"<p style='color:#6B7280; font-size:0.88rem;'>You are currently signed in as "
            f"<strong>@{user['username']}</strong>.</p>",
            unsafe_allow_html=True,
        )
        if st.button("🚪 Sign Out of Analyst AI", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.current_user = None
            _reset_user_session_state()
            st.rerun()

#  Footer 
st.markdown(f"""
<div class="analyst-footer" style="margin-top:2rem; padding:1rem 1.5rem; background:#F8FAFC;
            border-top:1px solid #E5E7EB; border-radius:8px;
            display:flex; justify-content:space-between; align-items:center;
            flex-wrap:wrap; gap:0.5rem;">
    <div style="font-size:0.85rem;">
        🔬 <strong>Analyst AI</strong> — Research & Analysis Agent
    </div>
    <div style="display:flex; gap:1.5rem;">
        <span style="font-size:0.8rem;">🤖 {model_name}</span>
        <span style="font-size:0.8rem;">💰 ${st.session_state.cost:.4f}</span>
        <span style="font-size:0.8rem;">🔢 {st.session_state.tokens_in + st.session_state.tokens_out:,} tokens</span>
        <span style="font-size:0.8rem;">{'🌙 Dark' if st.session_state.dark_mode else '☀️ Light'}</span>
    </div>
</div>
""", unsafe_allow_html=True)
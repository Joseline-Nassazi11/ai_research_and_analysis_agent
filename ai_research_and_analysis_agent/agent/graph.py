"""
agent/graph.py
--------------
LangGraph ReAct agent for Analyst AI.

Builds a stateful agent graph that alternates between an LLM call node
and a tool execution node until the model produces a final response.

FIXES APPLIED:
  1. System prompt race condition fixed — run_agent() no longer calls
     _update_system_prompt_in_cache() before build_graph() because
     build_graph() was overwriting it. Now build_graph() always receives
     the latest document_sources and updates the container directly.
  2. Multi-provider support added — Anthropic, Google Gemini, and Groq
     are now supported alongside OpenAI via a _build_llm() factory.
  3. Graph cache key now correctly excludes document_sources (they only
     affect the system prompt text, not graph structure).
  4. Token extraction handles all provider metadata formats.
  5. FIX: Search rules are now conditional on document_sources — when the
     user has no documents the agent is explicitly told NOT to call
     search_documents so it never references other users' indexed files.
"""

import os
from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from agent.prompts import get_system_prompt
from agent.tools import ALL_TOOLS, CORE_TOOLS

# Constants 
DEFAULT_MODEL       = "gpt-4o-mini"
DEFAULT_TEMPERATURE = 0.3
MAX_TOKENS          = 16000
CHARS_PER_TOKEN     = 4  # Fallback estimate only

# Graph Cache
# Maps (model, temperature, tools_tuple, personality, _PROMPT_VERSION) → compiled graph.
# document_sources intentionally excluded — they only affect the system
# prompt text, not the graph structure.
_GRAPH_CACHE: dict[tuple, object] = {}
_PROMPT_VERSION = "v4"  # Bumped to force cache invalidation after this fix


# State Schema
class AgentState(TypedDict):
    """State schema for the LangGraph ReAct agent."""
    messages: Annotated[list[BaseMessage], add_messages]


# LLM Factory

def _build_llm(model_name: str, temperature: float):
    """Instantiate the correct LLM based on the model name prefix.

    Supports OpenAI, Anthropic (Claude), Google (Gemini), and Groq models.
    Reads the appropriate API key from environment variables.

    Args:
        model_name: The model string (e.g. 'gpt-4o-mini', 'claude-sonnet-4-5').
        temperature: Sampling temperature between 0.0 and 1.0.

    Returns:
        A LangChain chat model instance with tools-bindable interface.

    Raises:
        ValueError: If the required API key for the provider is not set.
        ImportError: If the required LangChain provider package is not installed.
    """
    # OpenAI
    if model_name.startswith("gpt-"):
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY is not set. "
                "Add it to your .env file or enter it in the sidebar."
            )
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            max_tokens=MAX_TOKENS,
            api_key=api_key,
        )

    # Anthropic (Claude)
    if model_name.startswith("claude-"):
        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY is not set. "
                "Add it to your .env file or enter it in the sidebar."
            )
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError:
            raise ImportError(
                "langchain-anthropic is not installed. "
                "Run: pip install langchain-anthropic"
            )
        return ChatAnthropic(
            model=model_name,
            temperature=temperature,
            max_tokens=MAX_TOKENS,
            api_key=api_key,
        )

    # Google 
    if model_name.startswith("gemini-"):
        api_key = os.getenv("GOOGLE_API_KEY", "")
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY is not set. "
                "Add it to your .env file or enter it in the sidebar."
            )
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError:
            raise ImportError(
                "langchain-google-genai is not installed. "
                "Run: pip install langchain-google-genai"
            )
        return ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            max_output_tokens=MAX_TOKENS,
            google_api_key=api_key,
        )

    # Groq
    if model_name in (
        "llama-3.3-70b-versatile",
        "mixtral-8x7b-32768",
        "gemma2-9b-it",
    ):
        api_key = os.getenv("GROQ_API_KEY", "")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY is not set. "
                "Add it to your .env file or enter it in the sidebar."
            )
        try:
            from langchain_groq import ChatGroq
        except ImportError:
            raise ImportError(
                "langchain-groq is not installed. "
                "Run: pip install langchain-groq"
            )
        return ChatGroq(
            model=model_name,
            temperature=temperature,
            max_tokens=MAX_TOKENS,
            api_key=api_key,
        )

    # Unknown model — fall back to OpenAI with a warning
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise ValueError(
            f"Unknown model '{model_name}' and OPENAI_API_KEY is not set. "
            "Please select a supported model or add the correct API key."
        )
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model=model_name,
        temperature=temperature,
        max_tokens=MAX_TOKENS,
        api_key=api_key,
    )


# Prompt Suffix Builder

def _build_prompt_suffix(document_sources: list | None) -> str:
    """Build the search rules and citation instructions appended to the system prompt.

    CRITICAL FIX: Search rules are now conditional on whether the user has
    documents. When document_sources is empty the agent is explicitly told
    NOT to call search_documents — preventing it from hitting the shared
    FAISS index and referencing other users' files.

    Args:
        document_sources: List of document filenames for this user, or empty/None.

    Returns:
        A string to append to the base system prompt.
    """
    has_docs = bool(document_sources)

    if has_docs:
        # User HAS documents — instruct agent to search them first
        search_rules = """
## Search Rules (MANDATORY — READ FIRST)

CRITICAL: You have a `search_documents` tool that searches uploaded files using FAISS vector search. Files ARE indexed and available — never assume otherwise.

1. **ALWAYS call `search_documents` FIRST** before answering ANY question about uploaded documents or datasets. Never answer from memory.
2. **NEVER say** "the dataset is not available", "please upload the file", or "I cannot access the data" — the files are indexed. If search returns empty, rephrase and search again.
3. If the user mentions any filename (e.g. "uganda_tech_jobs.csv", "GOOG-10-K"), call `search_documents` with relevant keywords immediately.
4. For CSV/data questions, always call `search_documents` with the data topic (e.g. "salary", "job roles", "average").
"""
    else:
        # User has NO documents — explicitly forbid document search
        search_rules = """
## Search Rules (MANDATORY — READ FIRST)

IMPORTANT: This user has NO documents uploaded. The knowledge base is empty for this user.

1. **DO NOT call `search_documents`** — there are no documents to search. The tool will return empty results.
2. **DO NOT reference any document filenames** (e.g. GOOG-10-K, .pdf, .csv files) — you have not been given any.
3. Answer using **web_search** (Tavily) and your own knowledge ONLY.
4. If the user asks about a specific document, politely tell them to upload it first via the Documents tab.
"""

    citation_rules = """
## Citation Rules (MANDATORY)
You have access to save_citation and list_citations tools. Follow these rules strictly:

1. ANY time the user says "cite", "citation", "save citation", or "extract and cite" — you MUST call save_citation for each fact or source. Do not just list facts as plain text.

2. After every search_documents or web_search call that returns useful results — call save_citation once for the source used.

3. When generating a report — call save_citation for each major source referenced.

4. When the user asks "list citations" or "show citations" — call list_citations tool, do not type them from memory.

5. For save_citation always provide:
   - title: descriptive name of the fact or source (e.g. "Uganda Tech Jobs — Average Salary")
   - source_type: "document" for uploaded files, "web" for web search results
   - year: current year if unknown, leave authors blank if unknown

NEVER skip save_citation when the user explicitly asks to cite something. Calling save_citation is not optional — it is required behaviour.
"""

    return search_rules + citation_rules


# Graph Builder 

def build_graph(
    model_name: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    enabled_tools: list[str] | None = None,
    personality: str = "professional",
    document_sources: list | None = None,
) -> object:
    """Build and compile the LangGraph ReAct agent, with caching.

    The graph structure is cached by (model, temperature, tools, personality).
    The system prompt is ALWAYS updated with the latest document_sources
    so the agent knows which documents are currently indexed.

    Args:
        model_name: Model name string (e.g. 'gpt-4o-mini', 'claude-sonnet-4-5').
        temperature: Sampling temperature between 0.0 and 1.0.
        enabled_tools: List of tool name strings to bind to the LLM.
        personality: Agent tone — 'professional', 'concise', or 'academic'.
        document_sources: List of indexed document filenames for the system prompt.

    Returns:
        A compiled LangGraph StateGraph ready to invoke.

    Raises:
        ValueError: If the required API key is not set.
        KeyError: If an unknown tool name is passed in enabled_tools.
        ImportError: If a required provider package is not installed.
    """
    if enabled_tools is None:
        enabled_tools = CORE_TOOLS

    unknown = [n for n in enabled_tools if n not in ALL_TOOLS]
    if unknown:
        raise KeyError(
            f"Unknown tool names: {unknown}. "
            f"Valid tools: {list(ALL_TOOLS.keys())}"
        )

    cache_key = (model_name, temperature, tuple(sorted(enabled_tools)), personality, _PROMPT_VERSION)

    # Always build the latest system prompt
    # explicit instructions to NOT call search_documents.
    latest_prompt = get_system_prompt(
        personality=personality,
        document_sources=document_sources,
    )
    latest_prompt += _build_prompt_suffix(document_sources)

    if cache_key in _GRAPH_CACHE:
        # Graph is cached — just update the prompt container in-place
        _GRAPH_CACHE[cache_key]._system_prompt_container[0] = latest_prompt
        return _GRAPH_CACHE[cache_key]

    # First-time build 
    tools = [ALL_TOOLS[name] for name in enabled_tools]

    # Mutable container so call_agent closure always reads the latest prompt
    # without needing to rebuild the graph when documents change.
    system_prompt_container = [latest_prompt]

    llm = _build_llm(model_name, temperature)
    llm_with_tools = llm.bind_tools(tools)
    tool_node = ToolNode(tools)

    def call_agent(state: AgentState) -> dict:
        """Invoke the LLM with the current message history.

        Always prepends the current system prompt so document_sources
        changes are reflected on every invocation.
        """
        messages = list(state["messages"])
        current_prompt = system_prompt_container[0]

        # Remove any existing SystemMessage so we always use the latest
        messages = [m for m in messages if not isinstance(m, SystemMessage)]
        messages = [SystemMessage(content=current_prompt)] + messages

        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    def should_continue(state: AgentState) -> str:
        """Route to tools node or end depending on whether tool calls exist."""
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "tools"
        return END

    graph = StateGraph(AgentState)
    graph.add_node("agent", call_agent)
    graph.add_node("tools", tool_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", END: END},
    )
    graph.add_edge("tools", "agent")

    compiled = graph.compile()
    compiled._system_prompt_container = system_prompt_container
    _GRAPH_CACHE[cache_key] = compiled
    return compiled


# Token Extraction 

def _extract_token_usage(
    result_messages: list[BaseMessage],
    response_text: str,
) -> dict:
    """Extract real token counts from provider response metadata.

    Handles OpenAI, Anthropic, and Groq metadata formats.
    Falls back to character-count estimate if metadata is unavailable.

    Args:
        result_messages: All messages returned from the graph invocation.
        response_text: The final assistant response string (for fallback).

    Returns:
        Dict with keys: input_tokens, output_tokens, total_tokens, source.
    """
    for msg in reversed(result_messages):
        if not isinstance(msg, AIMessage):
            continue
        meta  = getattr(msg, "response_metadata", {}) or {}
        usage = meta.get("token_usage") or meta.get("usage") or {}

        input_t = (
            usage.get("prompt_tokens")          # OpenAI
            or usage.get("input_tokens")         # Anthropic
            or usage.get("prompt_token_count")   # Gemini
            or 0
        )
        output_t = (
            usage.get("completion_tokens")           # OpenAI
            or usage.get("output_tokens")            # Anthropic
            or usage.get("candidates_token_count")   # Gemini
            or 0
        )

        if input_t > 0 or output_t > 0:
            return {
                "input_tokens":  input_t,
                "output_tokens": output_t,
                "total_tokens":  input_t + output_t,
                "source": "openai_metadata",
            }

    # Fallback: character-count estimate
    total_chars = sum(len(str(m.content)) for m in result_messages)
    return {
        "input_tokens":  total_chars // CHARS_PER_TOKEN,
        "output_tokens": len(response_text) // CHARS_PER_TOKEN,
        "total_tokens":  (total_chars + len(response_text)) // CHARS_PER_TOKEN,
        "source": "character_estimate",
    }


# Public Interface 

def run_agent(
    user_message: str,
    history: list[dict],
    model_name: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    enabled_tools: list[str] | None = None,
    personality: str = "professional",
    document_sources: list | None = None,
) -> tuple[str, dict]:
    """Run one turn of the Analyst AI agent.

    Converts chat history to LangChain messages, invokes the compiled graph,
    and returns the final AI response text with token usage metadata.

    The system prompt is always updated with the latest document_sources
    before invocation — no stale document list will ever reach the agent.

    Args:
        user_message: The latest message from the user.
        history: Previous messages as list of dicts with 'role' and 'content'.
        model_name: LLM model string. Defaults to DEFAULT_MODEL.
        temperature: Sampling temperature. Defaults to DEFAULT_TEMPERATURE.
        enabled_tools: Tool names to enable. Defaults to CORE_TOOLS.
        personality: Agent response style. Defaults to 'professional'.
        document_sources: Currently indexed document filenames.

    Returns:
        Tuple of (response_text, usage_dict).

    Raises:
        ValueError: If the required API key is missing.
        RuntimeError: If the graph produces no valid AI response.
    """
    if enabled_tools is None:
        enabled_tools = CORE_TOOLS

    # build_graph() handles both cache lookup AND system prompt update —
    # we no longer call _update_system_prompt_in_cache() separately here
    # because that was the source of the race condition.
    graph = build_graph(
        model_name=model_name,
        temperature=temperature,
        enabled_tools=enabled_tools,
        personality=personality,
        document_sources=document_sources,  #  always fresh from app.py
    )

    # Convert history dicts → LangChain message objects
    lc_messages: list[BaseMessage] = []
    for msg in history:
        if msg["role"] == "user":
            lc_messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            lc_messages.append(AIMessage(content=msg["content"]))
    lc_messages.append(HumanMessage(content=user_message))

    result = graph.invoke({"messages": lc_messages})

    # Extract the last non-empty AI response
    response_text = ""
    for msg in reversed(result["messages"]):
        if isinstance(msg, AIMessage) and msg.content:
            response_text = msg.content
            break

    if not response_text:
        raise RuntimeError("Agent produced no response. Please try again.")

    usage = _extract_token_usage(result["messages"], response_text)
    return response_text, usage
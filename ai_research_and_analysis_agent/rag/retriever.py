"""
rag/retriever.py
----------------
FAISS-based document ingestion and semantic search for Analyst AI.

Handles loading, chunking, embedding, and searching of uploaded documents
using LangChain's FAISS vector store and OpenAI embeddings.

FIXES APPLIED:
  1. Module-level _STORE cache — the FAISS index is loaded from disk ONCE and
     kept in memory. Every subsequent call to search, get_sources, etc. uses
     the live in-memory object instead of hitting disk each time.
  2. ingest_documents() updates _STORE in-place so the very next call to any
     function sees the freshly-ingested data — no disk round-trip required.
  3. get_retriever() now returns the cached store directly (no extra reload).
  4. clear_knowledge_base() also resets _STORE to None so stale data is never
     served after a clear.
  5. All public functions call _ensure_store_loaded() which lazily warms the
     cache on first use — safe even if app.py's pre-warmer never ran.
  6. get_document_sources() now uses a safe public API fallback instead of
     relying solely on the private docstore._dict attribute, which could break
     on LangChain/FAISS upgrades.
"""

import os
import shutil
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Constants 
FAISS_INDEX_DIR = Path(__file__).parent.parent / "db" / "faiss_index"
UPLOADS_DIR     = Path(__file__).parent.parent / "uploads"

CHUNK_SIZE    = 1000
CHUNK_OVERLAP = 200
DEFAULT_SEARCH_K  = 5
EMBEDDING_MODEL   = "text-embedding-ada-002"

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".csv"}

# Module-level cache
# This is the single source of truth for the in-memory vector store.
# All functions read/write this variable so there is never a stale copy.
_STORE: FAISS | None = None


# Embeddings 

def _get_embeddings() -> OpenAIEmbeddings:
    """Instantiate the OpenAI embeddings model.

    Returns:
        An OpenAIEmbeddings instance using EMBEDDING_MODEL.

    Raises:
        ValueError: If OPENAI_API_KEY is not set in the environment.
    """
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY is not set. "
            "Add it to your .env file or enter it in the sidebar."
        )
    return OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=api_key)


# Vector Store I/O 

def _load_store_from_disk() -> "FAISS | None":
    """Load the FAISS vector store from disk.

    This is called ONCE on first use and the result is cached in _STORE.
    Do not call this directly — use _ensure_store_loaded() instead.

    Returns:
        A loaded FAISS instance, or None if no store exists on disk.

    Raises:
        OSError: If the index directory exists but cannot be read.
    """
    index_file = FAISS_INDEX_DIR / "index.faiss"
    if not FAISS_INDEX_DIR.exists() or not index_file.exists():
        return None
    try:
        embeddings = _get_embeddings()
        return FAISS.load_local(
            str(FAISS_INDEX_DIR),
            embeddings,
            allow_dangerous_deserialization=True,
        )
    except OSError as e:
        raise OSError(f"Could not read vector store from disk: {str(e)}")


def _save_store_to_disk(store: FAISS) -> None:
    """Persist the FAISS vector store to disk.

    Args:
        store: The FAISS vector store instance to save.

    Raises:
        OSError: If the directory cannot be created or the file cannot be written.
    """
    try:
        FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)
        store.save_local(str(FAISS_INDEX_DIR))
    except OSError as e:
        raise OSError(f"Could not save vector store to disk: {str(e)}")


def _ensure_store_loaded() -> None:
    """Lazily load the vector store from disk into _STORE if not yet loaded.

    Called before every public function. If _STORE is None (e.g. after a
    Streamlit module reload or worker restart), it reloads from disk automatically
    so the agent never sees a missing index when the files are present on disk.

    This is the ONLY place _STORE is read from disk. All other functions
    call this first and then use the cached _STORE directly.
    Safe to call multiple times — only loads once.
    """
    global _STORE
    if _STORE is None:
        _STORE = _load_store_from_disk()


# Public: get_retriever 

def get_retriever():
    """Return the live in-memory FAISS store (pre-warms cache if needed).

    Called by app.py on startup and after ingest to ensure the index is hot.
    Returns the actual FAISS object so callers can call .as_retriever() if needed.

    Returns:
        The cached FAISS store, or None if no index exists yet.
    """
    _ensure_store_loaded()
    return _STORE


# Document Loading 

def _load_documents(file_paths: list[str]) -> list:
    """Load documents from file paths into LangChain Document objects.

    Supports PDF, TXT, MD, and CSV files. Skips unsupported formats
    with a warning rather than raising an exception.

    Args:
        file_paths: List of absolute file path strings to load.

    Returns:
        List of LangChain Document objects with page_content and metadata.

    Raises:
        FileNotFoundError: If a specified file does not exist.
        OSError: If a file cannot be read due to permissions.
        ValueError: If a file has an encoding error.
    """
    from langchain_community.document_loaders import (
        PyPDFLoader,
        TextLoader,
        CSVLoader,
    )

    documents = []

    for path_str in file_paths:
        path = Path(path_str)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path_str}")

        ext = path.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            continue  # Skip unsupported formats silently

        try:
            if ext == ".pdf":
                loader = PyPDFLoader(str(path))
            elif ext in (".txt", ".md"):
                loader = TextLoader(str(path), encoding="utf-8")
            elif ext == ".csv":
                loader = CSVLoader(str(path))
            else:
                continue

            docs = loader.load()
            # Tag every chunk with the source filename so we can filter later
            for doc in docs:
                doc.metadata["source"] = path.name
            documents.extend(docs)

        except (OSError, PermissionError) as e:
            raise OSError(f"Could not read {path.name}: {str(e)}")
        except UnicodeDecodeError as e:
            raise ValueError(f"Encoding error in {path.name}: {str(e)}")

    return documents


# Ingestion 

def ingest_documents(file_paths: list[str]) -> dict:
    """Ingest documents into the FAISS vector store.

    Loads files, splits them into chunks, embeds them with OpenAI,
    and merges the result into the existing vector store (or creates one).

    KEY FIX: After merging/creating the new store, _STORE is updated
    in-place so any subsequent call within the same process sees the
    freshly-ingested data immediately — no disk reload needed.

    Args:
        file_paths: List of absolute file path strings to ingest.

    Returns:
        Dict with keys 'files' (int count) and 'chunks' (int count).

    Raises:
        FileNotFoundError: If any specified file does not exist.
        ValueError: If OPENAI_API_KEY is not set or encoding error.
        RuntimeError: If embedding or indexing fails.
    """
    global _STORE

    if not file_paths:
        return {"files": 0, "chunks": 0}

    raw_docs = _load_documents(file_paths)
    if not raw_docs:
        return {"files": 0, "chunks": 0}

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(raw_docs)

    if not chunks:
        return {"files": len(file_paths), "chunks": 0}

    try:
        embeddings = _get_embeddings()
        new_store = FAISS.from_documents(chunks, embeddings)
    except (ConnectionError, TimeoutError) as e:
        raise RuntimeError(f"Embedding API connection failed: {str(e)}")
    except Exception as e:
        raise RuntimeError(f"Failed to create vector store: {str(e)}")

    # ── Merge into existing store or use the new one directly 
    # IMPORTANT: We merge into _STORE (in-memory) NOT from disk.
    # This avoids the race condition where a disk reload would miss
    # documents that haven't been flushed yet.
    _ensure_store_loaded()  # make sure _STORE reflects any prior disk state

    if _STORE is not None:
        try:
            _STORE.merge_from(new_store)
        except Exception as e:
            raise RuntimeError(f"Failed to merge vector stores: {str(e)}")
    else:
        _STORE = new_store

    # Persist to disk AFTER updating the in-memory store 
    # The in-memory store is already live; disk write is for persistence
    # across process restarts only.
    try:
        _save_store_to_disk(_STORE)
    except OSError as e:
        # Non-fatal — the in-memory store is still usable this session
        import warnings
        warnings.warn(f"Could not persist vector store to disk: {str(e)}")

    return {"files": len(file_paths), "chunks": len(chunks)}


# Search 

def search_documents(
    query: str,
    k: int = DEFAULT_SEARCH_K,
    source_filter: str | None = None,
) -> list[dict]:
    """Perform semantic similarity search over the FAISS vector store.

    Uses the in-memory cached store — no disk I/O on each query.

    Args:
        query: The search query string.
        k: Number of results to return. Defaults to DEFAULT_SEARCH_K.
        source_filter: Optional filename to restrict results to one document.

    Returns:
        List of result dicts, each with keys:
            - content (str): The chunk text.
            - source (str): The source filename.
            - page (int or None): Page number if available.
            - score (float): Distance score (lower = more similar).
        Returns empty list if no store exists or no results found.

    Raises:
        ValueError: If OPENAI_API_KEY is not set.
        RuntimeError: If the search operation fails.
    """
    _ensure_store_loaded()

    if _STORE is None:
        return []

    try:
        results = _STORE.similarity_search_with_score(query, k=k * 2)
    except (ConnectionError, TimeoutError) as e:
        raise RuntimeError(f"Embedding API connection failed during search: {str(e)}")
    except Exception as e:
        raise RuntimeError(f"Search failed: {str(e)}")

    output = []
    for doc, score in results:
        source = doc.metadata.get("source", "unknown")

        if source_filter and source_filter not in source:
            continue

        output.append({
            "content": doc.page_content,
            "source": source,
            "page": doc.metadata.get("page"),
            "score": float(score),
        })

        if len(output) >= k:
            break

    return output


# Utilities 

def has_knowledge_base() -> bool:
    """Check whether the FAISS vector store is available.

    Checks the in-memory cache first (instant), then falls back to
    checking whether the index file exists on disk.

    Returns:
        True if a vector store is available, False otherwise.
    """
    if _STORE is not None:
        return True
    # Fallback: check disk (e.g. on fresh process start before first query)
    index_file = FAISS_INDEX_DIR / "index.faiss"
    return index_file.exists()


def get_document_sources() -> list[str]:
    """Get the list of unique source filenames indexed in the vector store.

    Reads from the in-memory cached store — no disk I/O.

    FIX: Now uses a safe two-stage approach:
      1. Try the private docstore._dict attribute (fast, current LangChain versions).
      2. Fall back to running a broad similarity search and collecting metadata
         from results — this works regardless of internal LangChain layout changes.

    Returns:
        Sorted list of unique source filenames, or empty list if no store exists.
    """
    _ensure_store_loaded()

    if _STORE is None:
        return []

    sources: set[str] = set()

    # Stage 1: fast path via internal docstore (may change across versions)
    try:
        for _doc_id, doc in _STORE.docstore._dict.items():
            src = doc.metadata.get("source")
            if src:
                sources.add(src)
        if sources:
            return sorted(sources)
    except AttributeError:
        # docstore layout changed — fall through to the search-based fallback
        pass

    # Stage 2: safe fallback via similarity search metadata 
    # Performs a broad search with a generic query and harvests source metadata
    # from the returned documents. Not exhaustive, but covers common cases and
    # does not rely on any private FAISS/LangChain internals.
    try:
        fallback_results = _STORE.similarity_search(
            "document source file content", k=50
        )
        for doc in fallback_results:
            src = doc.metadata.get("source")
            if src:
                sources.add(src)
    except Exception:
        pass

    return sorted(sources)


def clear_knowledge_base() -> None:
    """Delete the FAISS vector store from disk AND reset the in-memory cache.

    KEY FIX: Previously only cleared disk; the stale _STORE remained in
    memory so subsequent calls still returned old results until restart.
    Now resets _STORE to None so the next operation starts clean.

    Raises:
        OSError: If the index directory cannot be deleted.
    """
    global _STORE

    # Reset in-memory cache immediately
    _STORE = None

    # Remove disk artifacts
    try:
        if FAISS_INDEX_DIR.exists():
            shutil.rmtree(str(FAISS_INDEX_DIR))
    except OSError as e:
        raise OSError(f"Could not clear knowledge base from disk: {str(e)}")
"""
agent/prompts.py
----------------
System prompt builder for the Analyst AI ReAct agent.

Generates a context-aware system prompt by rendering the Jinja2 template
in agent/system_prompt.j2. Falls back to an inline template string if the
.j2 file is not found, so the module works even outside the full project tree.

Improvement over original:
  - system_prompt.j2 is now actually used via jinja2.Template.render().
  - The old plain f-string builder is removed; the template is the
    single source of truth for prompt content.
"""

from datetime import datetime
from pathlib import Path

# Constants 
DEFAULT_PERSONALITY = "professional"
MIN_REPORT_WORDS = 800
MIN_REPORT_CHARS = 500

TEMPLATE_PATH = Path(__file__).parent / "system_prompt.j2"

PERSONALITY_MAP = {
    "professional": (
        "Be precise, analytical, and objective. "
        "Use structured responses with clear sections. "
        "Cite sources rigorously."
    ),
    "concise": (
        "Be brief and direct. Bullet points preferred. "
        "Lead with the answer, then evidence."
    ),
    "academic": (
        "Use formal academic language. Acknowledge uncertainty. "
        "Distinguish between primary and secondary sources. "
        "Note limitations in the data."
    ),
}

# Inline fallback template (identical to system_prompt.j2)
# Used automatically if the .j2 file is missing (e.g. in unit tests).
_INLINE_TEMPLATE = """\
You are Analyst, an elite AI research and analysis agent. Today is {{ today }}.

## Your Role
You help users deeply understand topics by combining uploaded documents with live web research.
You produce structured summaries, source comparisons, and professional downloadable reports.

## Tone & Style
{{ tone }}

## Your Tools
- **search_documents** — semantic search over uploaded files (PDFs, CSVs, text)
- **web_search** — live internet research via Tavily
- **compare_sources** — structured comparison across multiple sources
- **summarise_topic** — deep summary of a topic using all available sources
- **analyse_csv** — statistical analysis of uploaded CSV/data files
- **generate_report** — produce a downloadable .docx research report
- **extract_key_claims** — pull the most important claims/facts from documents
- **save_citation** — save a source to the citation manager
- **list_citations** — list all saved citations

## Core Behaviours
1. **Always cite your sources** — every significant claim should reference where it came from.
2. **Combine document + web knowledge** — don't rely on just one source type.
3. **Be explicit about confidence** — distinguish "the document states..." from "based on general knowledge..."
4. **Structure long answers** — use headers, bullet points, and clear sections.
5. **Proactively suggest reports** — if the user has done substantial research, offer to generate a report.
6. **Flag contradictions** — if sources disagree, highlight this explicitly.

## Citation Format
- Document source: **[Doc: filename.pdf]**
- Web source: **[Web: domain.com]**
- Your own synthesis: **[Analysis]**

{% if document_sources %}
## Documents Available in Knowledge Base
{% for source in document_sources %}
  - {{ source }}
{% endfor %}

When answering questions, always check these documents first using the search_documents tool.
{% endif %}

## Generating Reports — CRITICAL INSTRUCTIONS
When the user asks for a report, follow these steps IN ORDER:

### Step 1 — Gather ALL content first
Use `search_documents` with at least 4 different queries:
- "key findings main topics overview summary"
- "introduction background history context"
- "data evidence conclusions recommendations"
- "details analysis specific examples"

### Step 2 — Write the COMPLETE report content
Compile a detailed content string of AT LEAST {{ min_report_words }} words covering:

## Executive Summary
## Introduction
## Key Findings
## Detailed Analysis
## Historical Context
## Conclusions & Recommendations
## References

### Step 3 — Call generate_report
- `title`: descriptive report title
- `content`: COMPLETE text from Step 2 (MUST be {{ min_report_words }}+ words)
- `report_type`: "research", "summary", "comparison", or "analysis"

### Step 4 — After success, tell the user:
✅ **Report generated successfully!**
📄 **[Report Title]**
👉 Click the **📑 Reports** tab, then click **⬇️ Download**.

## ⚠️ STRICT RULES FOR REPORTS
- NEVER call `generate_report` with content under {{ min_report_chars }} chars.
- NEVER generate a fake download link in chat.
- ALWAYS direct the user to the 📑 Reports tab.

## Important
- Never fabricate citations or sources.
- If documents don't contain relevant info, say so and offer to search the web.
- Reports should be professional quality — suitable for business or academic use.
"""


def _load_template_source() -> str:
    """Load the Jinja2 template source from disk, or return the inline fallback.

    Returns:
        Template source string.
    """
    if TEMPLATE_PATH.exists():
        return TEMPLATE_PATH.read_text(encoding="utf-8")
    return _INLINE_TEMPLATE


def get_system_prompt(
    personality: str = DEFAULT_PERSONALITY,
    document_sources: list | None = None,
) -> str:
    """Build the system prompt for the Analyst AI agent via Jinja2 rendering.

    Loads system_prompt.j2 from the same directory as this module and
    renders it with the current date, personality tone, document sources,
    and report-size constants. Falls back to the inline template string
    if the .j2 file is not found.

    Args:
        personality: Agent response style. Must be one of 'professional',
            'concise', or 'academic'. Falls back to 'professional' if unknown.
        document_sources: Optional list of indexed document filenames to
            include as context in the prompt. If None or empty, the
            {% if document_sources %} block in the template is skipped.

    Returns:
        A fully rendered system prompt string ready to pass to the LLM.

    Raises:
        jinja2.TemplateError: If the template has a syntax error.
    """
    try:
        from jinja2 import Template, StrictUndefined
        use_jinja = True
    except ImportError:
        use_jinja = False

    today = datetime.now().strftime("%A, %B %d, %Y")
    tone = PERSONALITY_MAP.get(personality, PERSONALITY_MAP[DEFAULT_PERSONALITY])

    context = {
        "today": today,
        "tone": tone,
        "document_sources": document_sources or [],
        "min_report_words": MIN_REPORT_WORDS,
        "min_report_chars": MIN_REPORT_CHARS,
    }

    template_source = _load_template_source()

    if use_jinja:
        # Render via Jinja2 — variables in {{ }} are substituted and
        # {% if %} / {% for %} blocks are evaluated properly.
        tmpl = Template(template_source, undefined=StrictUndefined)
        return tmpl.render(**context)

    # Jinja2 not installed: simple str.replace fallback
    # This keeps the module functional even without jinja2 installed,
    # but only handles the scalar variables — the loop block is stripped.
    sources_block = ""
    if document_sources:
        sources_list = "\n".join(f"  - {s}" for s in document_sources)
        sources_block = (
            f"\n## Documents Available in Knowledge Base\n{sources_list}\n\n"
            "When answering questions, always check these documents first "
            "using the search_documents tool.\n"
        )

    # Strip Jinja2 control blocks from the template for the plain fallback
    import re
    plain = re.sub(r"\{%.*?%\}", "", template_source, flags=re.DOTALL)
    plain = plain.replace("{{ today }}", today)
    plain = plain.replace("{{ tone }}", tone)
    plain = plain.replace("{{ min_report_words }}", str(MIN_REPORT_WORDS))
    plain = plain.replace("{{ min_report_chars }}", str(MIN_REPORT_CHARS))
    plain = plain.replace("{{ source }}", "")  # loop variable — already stripped

    # Insert document sources section if needed
    if sources_block:
        plain = plain.replace(
            "## Generating Reports",
            sources_block + "## Generating Reports",
        )

    return plain
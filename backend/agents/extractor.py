"""
Agent 1: Extractor

Converts a single raw source report into structured, atomic claims.
CRITICAL DESIGN RULE: this agent only ever sees ONE source at a time.
It must never be given other sources' text, or it could start silently
resolving contradictions itself -- that job belongs to the Verifier
agent in Phase 3.
"""

import json
from google import genai

EXTRACTOR_SYSTEM_PROMPT = """You are a Disaster Information Extraction Agent. Your job is to read a raw
report from a single source during an active disaster and extract
structured, atomic claims from it.

Rules:
- Extract only factual claims (location, casualty/injury counts,
  infrastructure status, resource availability, requests for help).
- Do NOT extract opinions, emotions, or general commentary.
- Each claim must be atomic (one fact per claim) and specific
  (include location/entity names when present).
- If the source is vague or lacks specifics, extract what little is
  there and mark it as low_specificity: true.
- Do not infer or add information not present in the text.

Output ONLY valid JSON in this exact format, nothing else:
{
  "source_id": "<id passed in>",
  "claims": [
    {
      "claim_id": "<short unique id>",
      "text": "<atomic claim in plain English>",
      "category": "casualties | infrastructure | resources | shelter | hazard | other",
      "location": "<specific location if mentioned, else null>",
      "low_specificity": false
    }
  ]
}"""


def _build_user_prompt(source: dict) -> str:
    return f"""Source ID: {source['id']}
Source name: {source['source_name']}
Timestamp: {source['timestamp']}
Raw report text:
"{source['raw_text']}"

Extract claims from this report following your instructions."""


def _strip_code_fences(text: str) -> str:
    """Gemini sometimes wraps JSON in ```json ... ``` even when told not to. Strip defensively."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return text.strip()


def extract_claims(client: genai.Client, model: str, source: dict) -> dict:
    """
    Runs the Extractor agent on a single source.
    Returns the parsed claims dict: {"source_id": ..., "claims": [...]}
    Raises ValueError if Gemini's output isn't valid JSON.
    """
    user_prompt = _build_user_prompt(source)

    response = client.models.generate_content(
        model=model,
        contents=user_prompt,
        config={
            "system_instruction": EXTRACTOR_SYSTEM_PROMPT,
            "temperature": 0.2,  # low temp: we want consistent, literal extraction, not creativity
            "response_mime_type": "application/json",
        },
    )

    raw_text = _strip_code_fences(response.text)

    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Extractor returned invalid JSON for source {source['id']}: {e}\nRaw output: {raw_text}"
        )

    # Defensive: make sure source_id always matches what we passed in,
    # even if the model mangles it.
    parsed["source_id"] = source["id"]
    return parsed


def extract_all(client: genai.Client, model: str, sources: list[dict]) -> list[dict]:
    """
    Runs extraction across all sources, one isolated call per source.
    Returns a list of per-source claim dicts, in the same order as input.
    """
    results = []
    for source in sources:
        result = extract_claims(client, model, source)
        results.append(result)
    return results
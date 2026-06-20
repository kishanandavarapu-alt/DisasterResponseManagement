"""
Agent 2: Cross-Verifier

Takes the combined extracted claims from ALL sources (unlike the
Extractor, which only ever sees one source at a time) and groups them
by real-world topic, classifying each group as corroborated,
contradicted, unconfirmed, or low_confidence.

This is the one agent in the pipeline that is deliberately given
cross-source context, because cross-referencing claims against each
other is its entire purpose.
"""

import json
from google import genai

VERIFIER_SYSTEM_PROMPT = """You are a Disaster Claim Verification Agent. You receive a list of
atomic claims extracted from multiple independent sources during an
active disaster. Your job is to cross-reference these claims against
each other and assess their reliability.

For each distinct factual topic (e.g., a specific bridge's status, a
specific shelter's capacity), group claims that refer to the same
real-world fact, even if worded differently or from different sources.

Classify each group as one of:
- "corroborated": 2+ independent sources agree
- "contradicted": sources disagree on the same fact
- "unconfirmed": only 1 source mentions it, no corroboration or contradiction
- "low_confidence": source(s) flagged low_specificity or claim is vague

For "contradicted" groups, clearly state what each side claims.

Output ONLY valid JSON in this exact format, nothing else:
{
  "verified_groups": [
    {
      "topic": "<short description of the fact in question>",
      "status": "corroborated | contradicted | unconfirmed | low_confidence",
      "supporting_claims": [
        {"source_id": "...", "claim_id": "...", "text": "..."}
      ],
      "contradicting_claims": [
        {"source_id": "...", "claim_id": "...", "text": "..."}
      ],
      "confidence_note": "<one sentence explaining the verdict>"
    }
  ]
}"""


def _strip_code_fences(text: str) -> str:
    """Gemini sometimes wraps JSON in ```json ... ``` even when told not to. Strip defensively."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return text.strip()


def _build_user_prompt(extraction_results: list[dict]) -> str:
    n_sources = len(extraction_results)
    claims_json = json.dumps(extraction_results, indent=2)
    return f"""Here are all claims extracted from {n_sources} independent sources during this
disaster event. Cross-verify them according to your instructions.

{claims_json}"""


def verify_claims(client: genai.Client, model: str, extraction_results: list[dict]) -> dict:
    """
    Runs the Cross-Verifier agent on the combined claims from all sources.

    extraction_results: list of per-source claim dicts, e.g. the
    "extraction_results" list returned by /extract.

    Returns the parsed verification dict: {"verified_groups": [...]}
    Raises ValueError if Gemini's output isn't valid JSON.
    """
    user_prompt = _build_user_prompt(extraction_results)

    response = client.models.generate_content(
        model=model,
        contents=user_prompt,
        config={
            "system_instruction": VERIFIER_SYSTEM_PROMPT,
            # Low temperature: verification verdicts should be consistent
            # across re-runs, not creative. This matters a lot for demo
            # reliability -- you don't want the contradiction you designed
            # in Phase 0 to randomly stop being flagged as "contradicted".
            "temperature": 0.2,
            "response_mime_type": "application/json",
        },
    )

    raw_text = _strip_code_fences(response.text)

    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Verifier returned invalid JSON: {e}\nRaw output: {raw_text}")

    return parsed
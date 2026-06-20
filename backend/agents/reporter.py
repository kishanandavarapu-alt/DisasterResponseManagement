"""
Agent 3: Reporter / Synthesizer

Takes the Cross-Verifier's output (verified_groups, already classified
as corroborated / contradicted / unconfirmed / low_confidence) and
writes a concise, professional situation report for emergency
responders.

CRITICAL DESIGN RULE: this agent must never silently resolve a
contradiction or upgrade an unconfirmed claim to a fact. Its job is to
present the verified picture honestly, including its own uncertainty,
not to clean it up into false confidence.
"""

import json
from google import genai

REPORTER_SYSTEM_PROMPT = """You are a Disaster Situation Report Agent. You receive verified claim
groups (already classified as corroborated, contradicted, unconfirmed,
or low_confidence) about an ongoing disaster. Your job is to write a
concise, professional situation report for emergency responders.

Rules:
- Lead with corroborated, high-confidence facts.
- Clearly flag contradicted information as "DISPUTED" and briefly
  state both sides -- never silently pick one side.
- Clearly flag unconfirmed claims as "UNVERIFIED -- requires follow-up"
  -- do not present them as fact.
- Treat low_confidence claims with caution: mention them only if
  operationally relevant (e.g. they involve casualties or urgent
  resource needs), and always caveat them explicitly. Do not let a
  low_confidence claim drive the headline.
- Do not exaggerate, do not add information not present in the input.
- End with a short "Recommended next verification steps" list for
  unconfirmed/contradicted items.

Output ONLY valid JSON in this exact format, nothing else:
{
  "headline": "<one-line summary of the situation>",
  "confirmed_facts": ["..."],
  "disputed_items": [{"topic": "...", "sides": ["...", "..."]}],
  "unverified_items": ["..."],
  "recommended_next_steps": ["..."]
}"""


def _strip_code_fences(text: str) -> str:
    """Gemini sometimes wraps JSON in ```json ... ``` even when told not to. Strip defensively."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return text.strip()


def _build_user_prompt(verification: dict) -> str:
    verified_groups_json = json.dumps(verification.get("verified_groups", []), indent=2)
    return f"""Here are the verified claim groups for this disaster event:

{verified_groups_json}

Generate the situation report following your instructions."""


def generate_report(client: genai.Client, model: str, verification: dict) -> dict:
    """
    Runs the Reporter agent on the Cross-Verifier's output.

    verification: the dict returned by verify_claims(), i.e.
    {"verified_groups": [...]}

    Returns the parsed situation report dict.
    Raises ValueError if Gemini's output isn't valid JSON.
    """
    user_prompt = _build_user_prompt(verification)

    response = client.models.generate_content(
        model=model,
        contents=user_prompt,
        config={
            "system_instruction": REPORTER_SYSTEM_PROMPT,
            # Low temperature: this is the final, judge-facing artifact.
            # We want it accurate and consistent, not creatively phrased
            # in a way that might drift from what the data actually shows.
            "temperature": 0.3,
            "response_mime_type": "application/json",
        },
    )

    raw_text = _strip_code_fences(response.text)

    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Reporter returned invalid JSON: {e}\nRaw output: {raw_text}")

    return parsed
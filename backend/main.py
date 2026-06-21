"""
Disaster Response Agent - Backend
Phase 1: Skeleton + Gemini connectivity check.
Phase 5: Firestore-backed run history (Google Cloud free tier integration).

Run with:
    uvicorn main:app --reload --port 8000

Then visit:
    http://localhost:8000/health
    http://localhost:8000/test-gemini
"""

import json
import os
from pathlib import Path

import firebase_admin
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from firebase_admin import credentials, firestore
from google import genai

from agents.extractor import extract_all, extract_claims
from agents.verifier import verify_claims
from agents.reporter import generate_report

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

load_dotenv()  # reads .env in this directory

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-2.5-flash-lite"  # higher free-tier daily quota than 2.5-flash

app = FastAPI(title="Disaster Response Agent API")

# Allow the frontend to call this API. Since the frontend is a plain HTML
# file (possibly opened directly via file://, or served from any local
# port/extension), we allow all origins here. This is fine for a local
# hackathon demo; would need tightening for any real deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# A single shared Gemini client. Created lazily so the app can still boot
# (and /health can still respond) even if the API key is missing.
_client = None

# Caches the last successful /report result in memory, since each run
# costs 7 Gemini calls and free-tier daily quotas are very limited.
_report_cache = None

# Lazily-initialized Firestore client (Firebase free Spark tier).
_firestore_db = None


def get_client() -> genai.Client:
    global _client
    if _client is None:
        if not GEMINI_API_KEY:
            raise HTTPException(
                status_code=500,
                detail="GEMINI_API_KEY is not set. Copy .env.example to .env and add your key.",
            )
        _client = genai.Client(api_key=GEMINI_API_KEY)
    return _client


def get_firestore_db():
    """
    Lazily initializes the Firebase Admin SDK and returns a Firestore client.
    Credentials come from FIREBASE_SERVICE_ACCOUNT_JSON, a single-line JSON
    string env var (see setup instructions) -- never a committed file.
    """
    global _firestore_db
    if _firestore_db is None:
        creds_json = os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON")
        if not creds_json:
            raise HTTPException(
                status_code=500,
                detail=(
                    "FIREBASE_SERVICE_ACCOUNT_JSON is not set. Generate a service "
                    "account key in Firebase Console > Project settings > Service "
                    "accounts, minify it to one line, and add it to .env."
                ),
            )
        try:
            cred_dict = json.loads(creds_json)
        except json.JSONDecodeError as e:
            raise HTTPException(
                status_code=500,
                detail=f"FIREBASE_SERVICE_ACCOUNT_JSON is not valid JSON: {e}",
            )

        if not firebase_admin._apps:
            cred = credentials.Certificate(cred_dict)
            firebase_admin.initialize_app(cred)

        _firestore_db = firestore.client()
    return _firestore_db


def save_report_to_history(scenario: dict, report: dict) -> None:
    """
    Saves a completed pipeline run to Firestore. Failures here are logged
    but never raised -- a Firestore hiccup should not break the actual
    verification pipeline response the user is waiting on.
    """
    try:
        db = get_firestore_db()
        db.collection("reports").add(
            {
                "timestamp": firestore.SERVER_TIMESTAMP,
                "scenario_name": scenario.get("name", "Unknown scenario"),
                "headline": report.get("headline", ""),
                "confirmed_count": len(report.get("confirmed_facts", [])),
                "disputed_count": len(report.get("disputed_items", [])),
                "unverified_count": len(report.get("unverified_items", [])),
                "full_report": report,
            }
        )
    except Exception as e:
        print(f"Warning: failed to save report to Firestore: {e}")


# Path to mock_data/sources.json, one level up from backend/
MOCK_DATA_PATH = Path(__file__).parent.parent / "mock_data" / "sources.json"


def load_mock_scenario() -> dict:
    if not MOCK_DATA_PATH.exists():
        raise HTTPException(
            status_code=500,
            detail=f"Mock scenario file not found at {MOCK_DATA_PATH}",
        )
    with open(MOCK_DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Phase 1 endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    """Confirms the server itself is up, independent of Gemini/Firestore."""
    return {
        "status": "ok",
        "gemini_key_loaded": bool(GEMINI_API_KEY),
        "firebase_configured": bool(os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON")),
    }


@app.get("/test-gemini")
def test_gemini():
    """
    Confirms end-to-end connectivity to the Gemini API before any
    agent logic is built on top of it.
    """
    client = get_client()
    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents='Respond with exactly: "Gemini connection OK"',
        )
        return {
            "status": "ok",
            "model": GEMINI_MODEL,
            "response_text": response.text.strip(),
        }
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Gemini call failed: {e}")


# ---------------------------------------------------------------------------
# Phase 2 endpoints: Extractor agent
# ---------------------------------------------------------------------------

@app.get("/scenario")
def get_scenario():
    """Returns the raw mock scenario (metadata + all 5 sources), unmodified."""
    return load_mock_scenario()


@app.post("/extract")
def run_extraction():
    """
    Runs the Extractor agent across all 5 mock sources, one isolated
    Gemini call per source. Returns a list of per-source claim sets.
    """
    client = get_client()
    scenario = load_mock_scenario()
    sources = scenario["sources"]

    try:
        results = extract_all(client, GEMINI_MODEL, sources)
    except ValueError as e:
        raise HTTPException(status_code=502, detail=str(e))

    return {"extraction_results": results}


@app.post("/extract/{source_id}")
def run_extraction_single(source_id: str):
    """Runs extraction on a single source by id -- useful for debugging one at a time."""
    client = get_client()
    scenario = load_mock_scenario()
    source = next((s for s in scenario["sources"] if s["id"] == source_id), None)
    if source is None:
        raise HTTPException(status_code=404, detail=f"Source '{source_id}' not found")

    try:
        result = extract_claims(client, GEMINI_MODEL, source)
    except ValueError as e:
        raise HTTPException(status_code=502, detail=str(e))

    return result


# ---------------------------------------------------------------------------
# Phase 3 endpoint: Cross-Verifier agent
# ---------------------------------------------------------------------------

@app.post("/verify")
def run_verification():
    """
    Full pipeline so far: runs extraction on all 5 sources, then feeds
    the combined claims into the Cross-Verifier agent in a single call.
    Returns both the raw extraction and the verification result, so the
    frontend can show the "claims -> verification" trail.
    """
    client = get_client()
    scenario = load_mock_scenario()
    sources = scenario["sources"]

    try:
        extraction_results = extract_all(client, GEMINI_MODEL, sources)
    except ValueError as e:
        raise HTTPException(status_code=502, detail=f"Extraction step failed: {e}")

    try:
        verification = verify_claims(client, GEMINI_MODEL, extraction_results)
    except ValueError as e:
        raise HTTPException(status_code=502, detail=f"Verification step failed: {e}")

    return {
        "extraction_results": extraction_results,
        "verification": verification,
    }


# ---------------------------------------------------------------------------
# Phase 4 endpoint: Reporter agent (full pipeline, all 3 agents)
# ---------------------------------------------------------------------------

@app.post("/report")
def run_full_pipeline(force_refresh: bool = False):
    """
    The complete pipeline: Extractor -> Cross-Verifier -> Reporter.
    Returns all three stages' output so the frontend can render the
    full trail (raw sources -> claims -> verification -> final report),
    not just the end result.

    Results are cached in memory after the first successful run, since
    each run costs 7 Gemini calls and free-tier daily quotas are very
    limited. Pass ?force_refresh=true to bypass the cache and re-run.

    Fresh (non-cached) runs are also persisted to Firestore so run
    history survives server restarts -- see GET /history.
    """
    global _report_cache
    if _report_cache is not None and not force_refresh:
        return {**_report_cache, "_cached": True}

    client = get_client()
    scenario = load_mock_scenario()
    sources = scenario["sources"]

    try:
        extraction_results = extract_all(client, GEMINI_MODEL, sources)
    except ValueError as e:
        raise HTTPException(status_code=502, detail=f"Extraction step failed: {e}")

    try:
        verification = verify_claims(client, GEMINI_MODEL, extraction_results)
    except ValueError as e:
        raise HTTPException(status_code=502, detail=f"Verification step failed: {e}")

    try:
        report = generate_report(client, GEMINI_MODEL, verification)
    except ValueError as e:
        raise HTTPException(status_code=502, detail=f"Report generation step failed: {e}")

    result = {
        "scenario": scenario["scenario"],
        "sources": sources,
        "extraction_results": extraction_results,
        "verification": verification,
        "report": report,
    }
    _report_cache = result

    save_report_to_history(scenario["scenario"], report)

    return {**result, "_cached": False}


# ---------------------------------------------------------------------------
# Phase 5 endpoint: Run history (Firestore)
# ---------------------------------------------------------------------------

@app.get("/history")
def get_history(limit: int = 10):
    """
    Returns past pipeline runs from Firestore, most recent first.
    This is the visible proof of the Google Cloud (Firestore) integration --
    surfaced in the frontend as a "Run History" section.
    """
    db = get_firestore_db()
    try:
        docs = (
            db.collection("reports")
            .order_by("timestamp", direction=firestore.Query.DESCENDING)
            .limit(limit)
            .stream()
        )
        history = []
        for doc in docs:
            data = doc.to_dict()
            ts = data.get("timestamp")
            history.append(
                {
                    "id": doc.id,
                    "timestamp": ts.isoformat() if ts else None,
                    "scenario_name": data.get("scenario_name"),
                    "headline": data.get("headline"),
                    "confirmed_count": data.get("confirmed_count"),
                    "disputed_count": data.get("disputed_count"),
                    "unverified_count": data.get("unverified_count"),
                }
            )
        return {"history": history}
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to fetch history: {e}")
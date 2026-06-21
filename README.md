# 🌀 Disaster Response Agent

**A multi-agent pipeline that turns chaotic, contradictory disaster reports into a verified, responder-ready situation report — in seconds.**

Built and demoed against real first-hours reporting from **Cyclone Amphan**, South 24 Parganas, West Bengal (20 May 2020).

---

## The Problem

In the first hours after a disaster, responders are flooded with information from wildly different sources — official alerts, news, citizen reports, volunteer radio chatter, and unverified social forwards — and much of it conflicts. Someone has to read all of it, figure out what's true, what's disputed, and what still needs verification, _fast_. This project automates that triage with a chain of LLM agents instead of a panicked human with five browser tabs open.

## How It Works

The pipeline runs as four visible stages, each handed off to the next:

| Stage                     | Agent                | What it does                                                                                                                                                   |
| ------------------------- | -------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **1. Raw Sources**        | —                    | Ingests 5 independent, conflicting reports (official alert, news/social, citizen report, volunteer radio, unverified forward)                                  |
| **2. Extraction**         | Extractor Agent      | Pulls discrete, categorized claims out of each raw source                                                                                                      |
| **3. Cross-Verification** | Cross-Verifier Agent | Groups claims by topic and tags each group as **corroborated**, **contradicted**, **unconfirmed**, or **low confidence**                                       |
| **4. Situation Report**   | Reporter Agent       | Synthesizes everything into a clean report: confirmed facts, disputed items (with both sides shown), unverified items, and recommended next verification steps |

Every completed run is then **persisted to Cloud Firestore** (Firebase free Spark tier), so the situation report and its verification trail survive server restarts and build a running history of past runs — not just a single in-memory result that vanishes on reload.

The result is a report that tells a responder exactly what they can act on immediately, what's still being disputed, and what someone needs to go check — plus a record of every past verification run for that scenario.

## Live Demo UI

The frontend (`docs/index.html`) is a single-file, dependency-light dashboard that visualizes the whole pipeline run in real time:

- **Animated header** with a Three.js pixel-noise background, character-by-character title reveal, and blur-fade subtitle
- **Stage 1** — source cards with a magnetic-line cursor effect
- **Stage 2** — claims rendered as chips per source, with a live claim-count badge and scroll-triggered reveal
- **Stage 3** — verification cards (color-coded by status) plus a sticky, scroll-animated sidebar listing every claim
- **Stage 4** — a typewriter-animated headline and a structured report with confirmed/disputed/unverified sections
- **Run History** — a Firestore-backed list of past completed runs (headline, timestamp, confirmed/disputed/unverified counts), loaded on page load and refreshed automatically after each fresh run

Everything is vanilla JS/CSS + Three.js + GSAP loaded via CDN — no build step, no framework, just open the HTML file.

## Architecture

```
┌─────────────┐     ┌────────────────┐     ┌─────────────────────┐     ┌────────────────┐
│ Raw Sources │ ──▶ │ Extractor Agent │ ──▶ │ Cross-Verifier Agent │ ──▶ │ Reporter Agent │
│  (5 inputs) │     │  (claims out)   │     │ (status per topic)  │     │  (final report) │
└─────────────┘     └────────────────┘     └─────────────────────┘     └───────┬────────┘
                                                                                 │
                                                                                 ▼
                                                                   FastAPI backend (localhost:8000)
                                                                                 │
                                                       ┌─────────────────────────┼─────────────────────────┐
                                                       ▼                                                     ▼
                                          Cloud Firestore (Firebase free tier)                  Static frontend (docs/index.html)
                                          — persists every completed run as run history —
```

The frontend calls a FastAPI backend at `POST /report` (optionally `?force_refresh=true` to bypass the in-memory cache and force a fresh round of LLM calls) and `GET /history` to load past runs from Firestore.

## Tech Stack

- **Gemini API** (`gemini-2.5-flash-lite`) — powers all three agents via structured JSON output (`responseSchema`)
- **FastAPI** — backend, with in-memory caching on top of the LLM pipeline
- **Cloud Firestore** (Firebase free Spark tier) — Google Cloud integration; persists run history across restarts
- **Vanilla JS/CSS + Three.js + GSAP** — frontend, no build step

## Getting Started

### Prerequisites

- Python 3.10+
- A Gemini API key (see backend `.env.example`)
- A Firebase project with Firestore enabled, and a service account key (see below)

### Set up Firestore

1. Create a project at [console.firebase.google.com](https://console.firebase.google.com) → **Build → Firestore Database → Create database** (production mode, any region).
2. **Project settings → Service accounts → Generate new private key** — downloads a JSON file. Keep this out of git.
3. Minify it to one line and add it to your backend `.env`:
   ```bash
   python -c "import json; print(json.dumps(json.load(open('your-key-file.json'))))"
   ```
   ```
   FIREBASE_SERVICE_ACCOUNT_JSON={"type":"service_account",...}
   ```

### Run the backend

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

`GET /health` will confirm both `gemini_key_loaded` and `firebase_configured` are `true` before you proceed.

### Open the frontend

Just open `docs/index.html` in a browser — or serve it statically:

```bash
cd docs
python -m http.server 5500
```

Click **▶ Run Verification** to load the cached demo run, or **↻ Force fresh run** to trigger new LLM calls end-to-end. Either way, the **Run History** section at the bottom pulls from Firestore and updates after every fresh run.

## Project Structure

```
disaster-response-agent/
├── backend/              # FastAPI app + agent pipeline (extractor, verifier, reporter)
│   ├── main.py           # API routes, Gemini client, Firestore client + history persistence
│   └── agents/           # extractor.py, verifier.py, reporter.py
├── docs/
│   ├── index.html        # Single-file animated dashboard (incl. Run History section)
│   └── ballpit.js         # (legacy) background animation, superseded by PixelBlast
└── README.md
```

## Why This Matters

Verification is the bottleneck in disaster response, not data collection — there's always plenty of reports coming in, the hard part is knowing which ones to trust. By splitting the job into extract → cross-verify → synthesize, each agent stays narrowly focused and auditable: you can see exactly which source said what, and why a claim was marked disputed instead of confirmed. Persisting every run to Firestore means that audit trail isn't lost the moment the server restarts, either.

## Roadmap

- [1] Swap the demo's static 5-source dataset for a live ingestion feed
- [2] Add geolocation tagging and a map view of claims
- [3] Support multi-disaster sessions with historical comparison
- [4] Add confidence scoring with source-reliability weighting over time
- [5] Surface Firestore history as a filterable, searchable archive rather than a flat recent-runs list

## License

MIT — use it, fork it, adapt it for the next disaster.

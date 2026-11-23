# 🧠 Conversational Digital Phenotyping Agent

An agentic AI system that runs a short, natural **daily check-in conversation**, extracts six behavioural and mood scores from what the user says, stores them over time, and shows the trends on a dashboard that flags sustained decline from the user's own baseline.

> **Disclaimer:** This is a research and education prototype. It is **not** a medical device, does not diagnose depression or any other condition, and must not replace professional care. Use only synthetic or consented data.

---

## Table of contents

1. [Objective](#1-objective)
2. [Key features](#2-key-features)
3. [How it works](#3-how-it-works)
4. [Tech stack](#4-tech-stack)
5. [Folder structure](#5-folder-structure)
6. [Setup](#6-setup)
7. [Usage](#7-usage)
8. [Data model and scoring](#8-data-model-and-scoring)
9. [Dashboard logic](#9-dashboard-logic)
10. [Configuration reference](#10-configuration-reference)
11. [Testing](#11-testing)
12. [Troubleshooting](#12-troubleshooting)
13. [Safety, privacy and ethics](#13-safety-privacy-and-ethics)
14. [Limitations](#14-limitations)
15. [Future work](#15-future-work)
16. [Project context and authors](#16-project-context-and-authors)

---

## 1. Objective

Passive sensing and questionnaires are the usual ways to monitor mental wellbeing, but questionnaires feel clinical and are easy to abandon. This project explores a third route: **conversational digital phenotyping**, where a friendly agent asks about the user's day and turns the free-text answers into structured, trackable signals.

The goals are to:

- Hold a **multi-turn, natural conversation** instead of a fixed form.
- Extract **six comparable daily scores** (mood, sleep, energy, anxiety, interest, social) from the whole conversation.
- Keep a **longitudinal record** so each day can be compared with the user's own baseline.
- Visualise trends and **alert on sustained deviation**, not on a single bad day.
- Demonstrate a clean **agentic architecture**: planner/interviewer, scorer, tool use, memory and a safety guard.

---

## 2. Key features

| Feature | Description |
|---|---|
| Conversational check-in | One short question per turn, reacting to what the user just said |
| Coverage tracking | The agent tracks which of the six areas it has learned about and steers toward gaps |
| Structured extraction | A separate scoring step produces validated 1–10 scores plus a one-line summary |
| Tool use | `save_checkin` writes a row to a local CSV or Google Sheets |
| Long-term memory | The last 3 days of check-ins are given to the agent for gentle follow-up |
| Safety guard | Rule-based detection of self-harm language, with a supportive message and helpline |
| Control logic in code | Stopping rules (min. coverage, `/done`, turn cap) live in code, not in the LLM |
| Dashboard | Wellbeing index, rolling average, baseline band, radar chart, alert and recent entries |
| Scheduler | Daily reminder via console, browser and optional Telegram |
| Offline test | Verifies the full pipeline with a stubbed LLM, so no API key is needed |

---

## 3. How it works

### 3.1 Agent graph

Every user message runs through a [LangGraph](https://langchain-ai.github.io/langgraph/) state machine:

```
                    ┌────────────┐
 user message ────► │   safety   │  keyword scan, count turns, detect /done
                    └─────┬──────┘
          /done or max turns │            │ otherwise
                    ┌────────▼───┐   ┌────▼───────┐
                    │   score    │   │ interview  │  LLM: next question + covered areas
                    └────────┬───┘   └────┬───────┘
                             │            │ enough coverage? ──no──► respond ──► END
                             │◄───────yes─┘
                    ┌────────▼───┐
                    │    save    │  tool: append row to CSV / Google Sheet
                    └────────┬───┘
                    ┌────────▼───┐
                    │   close    │  wrap-up message + saved confirmation
                    └────────┬───┘
                    ┌────────▼───┐
                    │  respond   │  prepend safety note if needed, append to chat
                    └────────┬───┘
                            END
```

### 3.2 Roles in the system

- **Interviewer agent** (`interview` node): returns structured output (`reply`, `covered_areas`, `ready_to_score`). It sees the transcript and the last 3 days of history.
- **Scoring agent** (`score` node): reads the full transcript and returns a validated schema with six integer scores, a summary and a risk flag. Values are clamped to 1–10 in code.
- **Tool** (`save` node → `save_checkin`): persists the result.
- **Guard** (`safety` node): deterministic regex check, independent of the LLM.
- **Memory:** short-term is the chat transcript held in the Streamlit session. Long-term is the stored history, injected into the interviewer's prompt.

### 3.3 Why the control logic is in code

The LLM proposes, the code decides. The interview only ends when at least 5 of 6 areas are covered, the user types `/done`, or the turn cap is reached. This prevents the agent from ending after one question or looping forever, and it makes behaviour predictable for a live demo.

---

## 4. Tech stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| Agent orchestration | LangGraph, LangChain Core |
| LLM | Google Gemini (default, free tier via `langchain-google-genai`) or Groq (`langchain-groq`) |
| Structured output | Pydantic schemas with `with_structured_output` |
| UI | Streamlit (chat and dashboard in one app) |
| Charts | Plotly |
| Data handling | Pandas |
| Storage | Local CSV (default) or Google Sheets via `gspread` and a service account |
| Scheduling | APScheduler |
| Notifications | Browser launch and optional Telegram Bot API (`requests`) |
| Config | `python-dotenv` |

---

## 5. Folder structure

```
checkin-agent/
├── app.py               # Streamlit entry point: chat page + dashboard page
├── agent.py             # LangGraph agent: schemas, prompts, nodes, tool, graph builder
├── storage.py           # CSV / Google Sheets storage, history loading, wellbeing index
├── dashboard.py         # Dashboard view rendered by app.py
├── scheduler.py         # Daily reminder (APScheduler, browser, optional Telegram)
├── test_offline.py      # End-to-end test with a stubbed LLM (no API key needed)
├── sample_data.csv      # 30 days of synthetic data (stable baseline, then decline)
├── requirements.txt     # Python dependencies
├── .env.example         # Template for configuration
├── README.md            # This file
│
│   ── created by you or at runtime ──
├── .env                 # Your real keys and settings (never commit this)
├── service_account.json # Google service account key (only for Sheets, never commit)
└── data/
    └── checkins.csv     # Local data store (created on first save)
```

Add this to `.gitignore` if you use Git:

```
.env
service_account.json
data/
venv/
__pycache__/
```

---

## 6. Setup

### 6.1 Prerequisites

- Python 3.10 or newer (`python --version`)
- A free Gemini API key from [Google AI Studio](https://aistudio.google.com) (or a Groq key)
- Optional: a Google account for Sheets storage, and a Telegram bot for reminders

### 6.2 Install

```bash
cd checkin-agent
python -m venv venv

# Windows
venv\Scripts\activate
# Mac / Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 6.3 Configure

```bash
cp .env.example .env        # Windows: copy .env.example .env
```

Edit `.env` and set at least:

```
LLM_PROVIDER=gemini
GOOGLE_API_KEY=your-gemini-key
GEMINI_MODEL=gemini-2.5-flash
```

Model names change over time. If you get a "model not found" error, set `GEMINI_MODEL` to another current Flash model. To use Groq, set `LLM_PROVIDER=groq` and `GROQ_API_KEY`.

### 6.4 Verify the wiring (no key needed)

```bash
python test_offline.py
```

Expected output ends with `ALL PASSED`.

### 6.5 Optional: Google Sheets storage

By default data is stored in `data/checkins.csv`. To use a Google Sheet instead:

1. In [Google Cloud Console](https://console.cloud.google.com), create a project and enable the **Google Sheets API** and **Google Drive API**.
2. Create a **Service Account**, then open it → **Keys → Add key → JSON**. Save the file as `service_account.json` in the project folder.
3. Create a Google Sheet and **share it with the service account's email** (the `client_email` value in the JSON) as **Editor**.
4. Add to `.env`:
   ```
   GOOGLE_SHEET_ID=<the long ID between /d/ and /edit in the sheet URL>
   GOOGLE_SERVICE_ACCOUNT_FILE=service_account.json
   ```

The header row is created automatically on the first write. The agent uses the first tab of the sheet.

### 6.6 Optional: Telegram reminders

1. Message **@BotFather** in Telegram, send `/newbot` and copy the token.
2. Send any message to your new bot, then open `https://api.telegram.org/bot<TOKEN>/getUpdates` and note the `chat.id`.
3. Add `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` to `.env`.

---

## 7. Usage

### 7.1 Start the app

```bash
streamlit run app.py
```

Open http://localhost:8501.

### 7.2 Do a check-in

1. Open the **💬 Check-in** page.
2. Reply naturally to the agent's questions. A typical check-in takes 5–7 messages.
3. When enough has been covered, the agent wraps up and shows **✅ Today's check-in is saved**.
4. Type `/done` at any time to finish early. Any areas not discussed are estimated cautiously toward the middle.
5. Click **🔄 New check-in** in the sidebar to start a fresh conversation.

The sidebar's **Agent internals** panel shows covered areas, turn count, risk flag and the extracted scores, which is useful when presenting.

### 7.3 View the dashboard

Open **📊 Dashboard**. If the store is empty, click **Load 30-day sample history** to load the synthetic data. Live check-ins then continue the same timeline.

### 7.4 Run the scheduler (optional)

```bash
python scheduler.py          # runs continuously, fires daily at CHECKIN_TIME (default 21:00)
python scheduler.py --now    # fire once immediately (good for demos)
```

When it fires, it prints a reminder, opens the app in your browser and sends a Telegram message if configured. The scheduler and the Streamlit app are separate processes, so run each in its own terminal.

### 7.5 Suggested live-demo flow

1. Start with the dashboard showing the sample history and the decline alert.
2. Run `python scheduler.py --now` to show the scheduled trigger.
3. Do a live check-in, for example: *"Pretty rough day, slept about 4 hours, felt drained and didn't feel like talking to anyone."*
4. Show the Agent internals panel and the new row in the CSV or Sheet.
5. Return to the dashboard and show the new point and trend.

---

## 8. Data model and scoring

### 8.1 Stored columns

| Column | Type | Meaning |
|---|---|---|
| `date` | `YYYY-MM-DD` | Day of the check-in |
| `time` | `HH:MM` | Time of the check-in |
| `mood_score` | 1–10 | 1 very low or hopeless, 5 neutral, 10 excellent |
| `sleep_score` | 1–10 | Duration and quality; 1 barely slept, 10 fully rested |
| `energy_score` | 1–10 | 1 exhausted, 10 energetic |
| `anxiety_score` | 1–10 | 1 calm, 10 severe. **Higher is worse** |
| `interest_score` | 1–10 | Enjoyment and motivation (anhedonia proxy); 1 none, 10 fully engaged |
| `social_score` | 1–10 | 1 isolated, 10 connected and active |
| `summary` | text | One neutral sentence about the day |
| `risk_flag` | `yes` / `no` | Self-harm or hopelessness mentioned |

The six areas map loosely to common screening concepts: depressed mood, sleep disturbance, fatigue, loss of interest, anxiety and social withdrawal. The mapping is illustrative and has not been clinically validated.

### 8.2 Wellbeing index

A single composite used for the main trend chart:

```
wellbeing = (mood + sleep + energy + interest + social + (11 − anxiety)) / 6
```

Anxiety is inverted so that higher is better for every term.

### 8.3 Duplicate days

If several check-ins exist for one date, the dashboard keeps the **latest** one.

---

## 9. Dashboard logic

- **Baseline:** the mean wellbeing of the first 14 logged days.
- **Alert threshold:** `baseline mean − max(baseline std, 0.5)`.
- **Alert:** shown when more than 14 days exist and the **last 7-day average** is below the threshold. A single bad day does not trigger it.
- **Views:** KPI cards, wellbeing trend with baseline and threshold lines, individual score lines (optional 7-day smoothing), a radar chart of baseline versus last 7 days, and the 10 most recent entries with summaries.

Thresholds are simple and intentionally explainable. They are a starting point, not a validated clinical rule.

---

## 10. Configuration reference

All settings are read from `.env` or the environment.

| Variable | Default | Purpose |
|---|---|---|
| `LLM_PROVIDER` | `gemini` | `gemini` or `groq` |
| `GOOGLE_API_KEY` | none | Gemini API key |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Gemini model name |
| `GROQ_API_KEY` | none | Groq API key |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Groq model name |
| `GOOGLE_SHEET_ID` | empty | Enables Sheets storage together with the next variable |
| `GOOGLE_SERVICE_ACCOUNT_FILE` | `service_account.json` in the template | Path to the service account key |
| `CHECKINS_CSV` | `data/checkins.csv` | Local CSV path when Sheets is not configured |
| `MAX_USER_TURNS` | `8` | Hard cap on user messages before scoring is forced |
| `CHECKIN_TIME` | `21:00` | Daily reminder time (24 h, local) |
| `TELEGRAM_BOT_TOKEN` | empty | Optional reminders |
| `TELEGRAM_CHAT_ID` | empty | Optional reminders |
| `APP_URL` | `http://localhost:8501` | Link included in reminders |

Other tunables in `agent.py`: `MIN_AREAS` (areas required before wrapping up, default 5), the interview and scoring prompts, and the risk keyword list.

---

## 11. Testing

```bash
python test_offline.py
```

This replaces the LLM with a stub and checks that:

1. A multi-turn conversation reaches the scoring step and saves exactly one row.
2. Out-of-range scores are clamped to 1–10.
3. A self-harm message triggers the safety response (including the helpline) and sets the risk flag.
4. `/done` forces scoring and saving.

It does **not** call Gemini or Google Sheets. Before a real demo, run one full check-in with your own key (and your Sheet, if used) to confirm those paths.

---

## 12. Troubleshooting

| Problem | Likely cause and fix |
|---|---|
| `model not found` / 404 | Change `GEMINI_MODEL` to another current model |
| 429 / quota exceeded | Wait a minute, or switch to Groq (`LLM_PROVIDER=groq`) |
| API key error | Check `.env` is in the project folder and the key has no quotes or spaces |
| Agent keeps asking and never saves | Type `/done`; it also stops automatically after `MAX_USER_TURNS` |
| Malformed or empty LLM output | Retry the message; free models occasionally fail structured output |
| Sheets: permission denied / not found | Sheet not shared with the service account email, wrong `GOOGLE_SHEET_ID`, or APIs not enabled |
| Dashboard empty | No rows yet; load the sample history or complete a check-in |
| Scheduler does nothing | It must stay running in its own terminal; use `--now` to test |
| `ModuleNotFoundError` | Virtual environment not activated, or `pip install -r requirements.txt` not run |

---

## 13. Safety, privacy and ethics

- **Not a diagnostic tool.** Scores are rough, LLM-derived estimates from short conversations.
- **Safety guard.** If the user's message matches self-harm patterns, the system shows a supportive message with **Tele-MANAS 14416** (India) and advises contacting local emergency services if in danger, and marks the day with `risk_flag = yes`. Change the helpline in `agent.py` (`SAFETY_MESSAGE`) if you deploy elsewhere. A keyword list is not exhaustive, so treat it as a minimum safeguard.
- **Privacy.** Conversation text is sent to the LLM provider you choose. Stored data contains scores and a short summary, not the full transcript. Use synthetic or explicitly consented data only.
- **Keys and credentials.** Never commit `.env` or `service_account.json`.
- **Transparency.** Users should know they are talking to an AI and what is stored.
- **Human oversight.** The dashboard alert is a prompt for a human follow-up, not an automated decision.

---

## 14. Limitations

- Scores come from a short self-report conversation and LLM judgement, so they vary between runs and models.
- The mapping to clinical constructs is not validated, and no clinical accuracy is claimed.
- The baseline uses the first 14 days only, which assumes those days are representative.
- Conversation memory lives in the Streamlit session and is lost on reload.
- Single-user design; there is no authentication or multi-user separation.
- English-language prompts only.
- Keyword-based risk detection will miss indirect or non-English expressions.

---

## 15. Future work

- Validate scores against PHQ-9 and GAD-7 collected alongside check-ins.
- Add a weekly trend-analysis agent that writes a short narrative report.
- Per-user accounts, consent flow and encrypted storage.
- Multilingual support (Hindi and other Indian languages).
- Combine with other signals, such as the EEG-based classifier from the wider project.
- Replace the keyword guard with an LLM-assisted risk classifier, kept behind the deterministic one.
- Automatic WhatsApp or Telegram two-way chat in addition to the web UI.

---

## 16. Project context and authors

This module is the **conversational digital phenotyping** component of the project *"EEG-Based Classification for Baseline Depression and Conversational Digital Phenotyping for Behavioral Monitoring"*.

- **Author:** Shivam Singh
- **Programme:** B.Tech in Artificial Intelligence and Machine Learning, Birla Institute of Technology, Patna (Off-Campus)
- **Mentor:** Dr. Gyanendra Tiwari Sir

Built with LangGraph, Google Gemini, Streamlit, Plotly and Pandas.
"""Conversational phenotyping agent built as a LangGraph state machine.

Graph:  safety -> (interview | score) ; interview -> (respond | score)
        score -> save -> close -> respond -> END
"""
import os
import re
from datetime import datetime
from functools import lru_cache
from typing import Optional, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

import storage

AREAS = ["mood", "sleep", "energy", "anxiety", "interest", "social"]
MAX_USER_TURNS = int(os.getenv("MAX_USER_TURNS", "8"))   # hard stop for the interview
MIN_AREAS = 5                                            # areas needed before wrapping up
OPENING = "Hi! 🌙 It's check-in time. How has your day been?"

RISK_RE = re.compile(
    r"(suicid|kill myself|end my life|want to die|don'?t want to (live|be here)|"
    r"self[- ]?harm|hurt myself|cut myself|no reason to live|better off dead)", re.I)
SAFETY_MESSAGE = (
    "I'm really glad you told me, and I'm taking it seriously. You don't have to carry this alone. "
    "Please reach out to someone you trust right now. In India you can call Tele-MANAS on 14416 at "
    "any time, and if you are in immediate danger please call your local emergency number. 💙")


# ----------------------------------------------------------------- schemas
class TurnDecision(BaseModel):
    reply: str = Field(description="Next message to the user")
    covered_areas: list[str] = Field(
        description="Areas among mood, sleep, energy, anxiety, interest, social that the user "
                    "has given enough detail about so far")
    ready_to_score: bool = Field(description="True only when all or nearly all six areas are covered")


class CheckinScores(BaseModel):
    mood_score: int = Field(description="1 very low .. 10 excellent")
    sleep_score: int = Field(description="1 very poor .. 10 fully rested")
    energy_score: int = Field(description="1 exhausted .. 10 energetic")
    anxiety_score: int = Field(description="1 calm .. 10 severe anxiety (higher is worse)")
    interest_score: int = Field(description="1 no interest .. 10 fully engaged")
    social_score: int = Field(description="1 isolated .. 10 very connected")
    summary: str = Field(description="One neutral sentence describing the day")
    risk_flag: bool = Field(description="True if self-harm, suicide or hopelessness about living was mentioned")


# ----------------------------------------------------------------- prompts
INTERVIEW_PROMPT = """You are the interviewer agent of a daily check-in app (a college research demo of conversational digital phenotyping). You are a warm, friendly companion. You are not a therapist and you never diagnose.
Today is <<TODAY>>. You opened the chat by asking how their day has been.

Earlier check-ins, for gentle continuity only (never re-ask what they already answered today):
<<HISTORY>>

Your job: through natural conversation, learn enough about six areas: mood, sleep (last night), energy, anxiety (stress or worry), interest (enjoyment and motivation), social (contact with people).

Rules:
- Ask exactly ONE short question per turn, in 1-3 casual sentences. React to what they just said first.
- Pick the area that is least covered and weave it in naturally. If an answer is vague, ask a gentle follow-up.
- If earlier check-ins show a notable change (for example poor sleep yesterday), you may follow up on it once.
- Never mention scores, scales or clinical terms. Never give medical advice.
- If they express hopelessness or thoughts of self-harm, respond with care and encourage them to reach out to someone they trust. A helpline message is added automatically by the system.

Output fields:
- reply: your next message to the user.
- covered_areas: every area (from mood, sleep, energy, anxiety, interest, social) the user has now given enough detail about, across the whole conversation.
- ready_to_score: true only when all or nearly all six areas are covered. In that case reply must be a warm 1-2 sentence wrap-up acknowledging what they shared, with NO question."""

SCORING_PROMPT = """You are the scoring agent for a daily wellbeing check-in. Read the conversation and score the USER's day. All scores are integers from 1 to 10.
- mood_score: 1 = very low or hopeless, 5 = neutral, 10 = excellent
- sleep_score: 1 = barely slept or very poor, 10 = fully rested. Consider duration (7-9 hours is ideal) and quality.
- energy_score: 1 = exhausted, 10 = full of energy
- anxiety_score: 1 = calm, 10 = severe anxiety or stress. HIGHER IS WORSE for this one.
- interest_score: 1 = no interest or enjoyment in anything, 10 = fully engaged and enjoying things
- social_score: 1 = isolated or avoiding people, 10 = connected and socially active
- summary: one neutral sentence about the day.
- risk_flag: true if self-harm, suicide or hopelessness about living was mentioned.
Base scores only on what the user said. If an area was not discussed, estimate cautiously toward the middle and say so in the summary."""


# ----------------------------------------------------------------- LLM access
@lru_cache(maxsize=1)
def get_llm():
    provider = os.getenv("LLM_PROVIDER", "gemini").lower()
    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"), temperature=0.4)
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"), temperature=0.4)


def call_structured(schema, system: str, messages: list[dict]):
    """Single place where the LLM is called (easy to mock in tests)."""
    msgs = [SystemMessage(content=system)]
    for m in messages:
        if m["role"] == "user":
            msgs.append(HumanMessage(content=m["content"]))
        elif len(msgs) > 1:               # Gemini wants the history to start with a user turn
            msgs.append(AIMessage(content=m["content"]))
    return get_llm().with_structured_output(schema).invoke(msgs)


# ----------------------------------------------------------------- state & tools
class State(TypedDict, total=False):
    messages: list
    history: str
    covered: list
    risk: bool
    done: bool
    reply: str
    scores: Optional[dict]
    turns: int
    force_score: bool
    ready: bool
    saved: bool
    save_error: str
    safety_note: str
    safety_shown: bool


def new_state() -> State:
    return State(messages=[{"role": "assistant", "content": OPENING}], history=storage.recent_context(),
                 covered=[], risk=False, done=False, reply="", scores=None, turns=0,
                 force_score=False, ready=False, saved=False, save_error="",
                 safety_note="", safety_shown=False)


def save_checkin(scores: dict) -> None:
    """TOOL: write one row to the data store (CSV or Google Sheet)."""
    now = datetime.now()
    row = {"date": now.strftime("%Y-%m-%d"), "time": now.strftime("%H:%M"), **scores}
    storage.append_rows([row])


def _clamp(x) -> int:
    try:
        return max(1, min(10, int(round(float(x)))))
    except (TypeError, ValueError):
        return 5


# ----------------------------------------------------------------- graph nodes
def safety(state: State) -> dict:
    last = state["messages"][-1]["content"]
    hit = bool(RISK_RE.search(last))
    note = SAFETY_MESSAGE if hit and not state.get("safety_shown") else ""
    return {"turns": state.get("turns", 0) + 1, "reply": "", "ready": False,
            "risk": state.get("risk", False) or hit, "safety_note": note,
            "safety_shown": state.get("safety_shown", False) or hit,
            "force_score": last.strip().lower() == "/done"}


def interview(state: State) -> dict:
    system = (INTERVIEW_PROMPT
              .replace("<<TODAY>>", datetime.now().strftime("%A, %d %B %Y"))
              .replace("<<HISTORY>>", state.get("history") or "No earlier check-ins."))
    d = call_structured(TurnDecision, system, state["messages"])
    covered = sorted(set(state.get("covered", [])) | {a.lower().strip() for a in d.covered_areas if a.lower().strip() in AREAS})
    ready = (d.ready_to_score and len(covered) >= MIN_AREAS) or len(covered) == len(AREAS)
    return {"covered": covered, "reply": d.reply.strip(), "ready": ready}


def score(state: State) -> dict:
    transcript = "\n".join(f"{'User' if m['role'] == 'user' else 'Companion'}: {m['content']}"
                           for m in state["messages"])
    r = call_structured(CheckinScores, SCORING_PROMPT, [{"role": "user", "content": transcript}])
    risk = state.get("risk", False) or bool(r.risk_flag)
    scores = {"mood_score": _clamp(r.mood_score), "sleep_score": _clamp(r.sleep_score),
              "energy_score": _clamp(r.energy_score), "anxiety_score": _clamp(r.anxiety_score),
              "interest_score": _clamp(r.interest_score), "social_score": _clamp(r.social_score),
              "summary": r.summary.strip(), "risk_flag": "yes" if risk else "no"}
    return {"scores": scores, "risk": risk}


def save(state: State) -> dict:
    try:
        save_checkin(state["scores"])
        return {"saved": True, "save_error": ""}
    except Exception as e:                      # keep the demo alive, show the reason
        return {"saved": False, "save_error": str(e)}


def close(state: State) -> dict:
    base = state.get("reply") or "Thank you for sharing all of that with me."
    tail = ("✅ Today's check-in is saved. See you tomorrow 🌙" if state.get("saved")
            else f"⚠️ I couldn't save today's check-in: {state.get('save_error')}")
    return {"reply": f"{base}\n\n{tail}", "done": True}


def respond(state: State) -> dict:
    text = state["reply"]
    if state.get("safety_note"):
        text = f"{state['safety_note']}\n\n{text}"
    return {"messages": state["messages"] + [{"role": "assistant", "content": text}], "reply": text}


def build_graph():
    g = StateGraph(State)
    for name, fn in [("safety", safety), ("interview", interview), ("score", score),
                     ("save", save), ("close", close), ("respond", respond)]:
        g.add_node(name, fn)
    g.add_edge(START, "safety")
    g.add_conditional_edges(
        "safety", lambda s: "score" if (s.get("force_score") or s["turns"] >= MAX_USER_TURNS) else "interview",
        {"score": "score", "interview": "interview"})
    g.add_conditional_edges("interview", lambda s: "score" if s.get("ready") else "respond",
                            {"score": "score", "respond": "respond"})
    g.add_edge("score", "save")
    g.add_edge("save", "close")
    g.add_edge("close", "respond")
    g.add_edge("respond", END)
    return g.compile()

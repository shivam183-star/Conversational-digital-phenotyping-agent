"""Storage layer: local CSV by default, Google Sheets if configured."""
import csv
import os
from pathlib import Path

import pandas as pd

COLUMNS = ["date", "time", "mood_score", "sleep_score", "energy_score",
           "anxiety_score", "interest_score", "social_score", "summary", "risk_flag"]
SCORE_COLS = COLUMNS[2:8]
DATA_FILE = Path(os.getenv("CHECKINS_CSV", "data/checkins.csv"))
SAMPLE_FILE = Path(__file__).parent / "sample_data.csv"


def using_sheets() -> bool:
    return bool(os.getenv("GOOGLE_SHEET_ID") and os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE"))


def _worksheet():
    import gspread
    gc = gspread.service_account(filename=os.environ["GOOGLE_SERVICE_ACCOUNT_FILE"])
    ws = gc.open_by_key(os.environ["GOOGLE_SHEET_ID"]).sheet1
    if not ws.get_all_values():
        ws.append_row(COLUMNS)
    return ws


def append_rows(rows: list[dict]) -> None:
    values = [[r.get(c, "") for c in COLUMNS] for r in rows]
    if using_sheets():
        _worksheet().append_rows(values, value_input_option="USER_ENTERED")
        return
    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    is_new = not DATA_FILE.exists() or DATA_FILE.stat().st_size == 0
    with open(DATA_FILE, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if is_new:
            w.writerow(COLUMNS)
        w.writerows(values)


def load_checkins() -> pd.DataFrame:
    """One row per day (latest wins) with a derived wellbeing index."""
    if using_sheets():
        df = pd.DataFrame(_worksheet().get_all_records())
    elif DATA_FILE.exists():
        df = pd.read_csv(DATA_FILE)
    else:
        df = pd.DataFrame(columns=COLUMNS)
    for c in COLUMNS:
        if c not in df.columns:
            df[c] = None
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for c in SCORE_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["date", *SCORE_COLS])
    df = df.sort_values(["date", "time"]).drop_duplicates("date", keep="last").reset_index(drop=True)
    df["wellbeing"] = (df["mood_score"] + df["sleep_score"] + df["energy_score"]
                       + df["interest_score"] + df["social_score"] + (11 - df["anxiety_score"])) / 6
    df["wellbeing_7d"] = df["wellbeing"].rolling(7, min_periods=1).mean()
    return df


def recent_context(n: int = 3) -> str:
    """Long-term memory: the last n days, as text for the agent's prompt."""
    df = load_checkins().tail(n)
    lines = []
    for _, r in df.iterrows():
        lines.append(f"{r['date']:%Y-%m-%d}: mood {r['mood_score']:.0f}, sleep {r['sleep_score']:.0f}, "
                     f"energy {r['energy_score']:.0f}, anxiety {r['anxiety_score']:.0f} (high=bad), "
                     f"interest {r['interest_score']:.0f}, social {r['social_score']:.0f}. {r['summary']}")
    return "\n".join(lines)


def seed_sample() -> int:
    """Load the 30-day demo history, only if the store is empty."""
    if not load_checkins().empty:
        return 0
    rows = pd.read_csv(SAMPLE_FILE).to_dict("records")
    append_rows(rows)
    return len(rows)

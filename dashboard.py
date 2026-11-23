"""Dashboard view (rendered by app.py)."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import storage

SCORES = {"mood_score": "Mood", "sleep_score": "Sleep", "energy_score": "Energy",
          "anxiety_score": "Anxiety (higher = worse)", "interest_score": "Interest",
          "social_score": "Social"}
BASELINE_DAYS = 14


def render():
    st.title("📊 Daily Check-in Dashboard")
    st.caption("Behavioural-monitoring prototype. Not a diagnostic tool.")
    st.sidebar.caption("Data: " + ("Google Sheet" if storage.using_sheets() else str(storage.DATA_FILE)))
    smooth = st.sidebar.checkbox("7-day rolling average", value=True)

    df = storage.load_checkins()
    if df.empty:
        st.info("No check-ins yet. Do one in the Check-in tab, or load the 30-day sample history.")
        if st.button("Load 30-day sample history"):
            storage.seed_sample()
            st.rerun()
        return
    if len(df) < 5 and st.sidebar.button("Load 30-day sample history"):
        st.sidebar.warning("Sample history loads only into an empty store.")

    base = df.head(BASELINE_DAYS)
    base_mean, base_std = base["wellbeing"].mean(), base["wellbeing"].std(ddof=0)
    last7 = df.tail(7)["wellbeing"].mean()
    prev7 = df.iloc[-14:-7]["wellbeing"].mean() if len(df) > 7 else float("nan")
    threshold = base_mean - max(base_std, 0.5)
    declining = len(df) > BASELINE_DAYS and last7 < threshold

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Latest wellbeing", f"{df['wellbeing'].iloc[-1]:.1f} / 10")
    c2.metric("7-day average", f"{last7:.1f}", None if pd.isna(prev7) else f"{last7 - prev7:+.1f} vs prior week")
    c3.metric("Days logged", len(df))
    c4.metric("Risk flags", int((df["risk_flag"].astype(str).str.lower() == "yes").sum()))

    if declining:
        st.error(f"⚠️ Sustained decline: 7-day average {last7:.1f} is below the personal baseline band "
                 f"({threshold:.1f}). Worth a human follow-up.")
    elif len(df) > BASELINE_DAYS:
        st.success("Within the personal baseline range.")
    else:
        st.info(f"Building baseline: {len(df)}/{BASELINE_DAYS} days.")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["date"], y=df["wellbeing"], mode="lines+markers", name="Daily", opacity=0.45))
    fig.add_trace(go.Scatter(x=df["date"], y=df["wellbeing_7d"], mode="lines", name="7-day avg", line=dict(width=4)))
    fig.add_hline(y=base_mean, line_dash="dash", annotation_text="Baseline")
    fig.add_hline(y=threshold, line_dash="dot", line_color="red", annotation_text="Alert threshold")
    fig.update_layout(title="Overall wellbeing index (mean of 6 scores, anxiety inverted)",
                      yaxis=dict(range=[0, 10.5]), height=380, margin=dict(t=50, b=10))
    st.plotly_chart(fig, width="stretch")

    left, right = st.columns([3, 2])
    with left:
        picked = st.multiselect("Scores to plot", list(SCORES), default=list(SCORES), format_func=SCORES.get)
        fig2 = go.Figure()
        for c in picked:
            y = df[c].rolling(7, min_periods=1).mean() if smooth else df[c]
            fig2.add_trace(go.Scatter(x=df["date"], y=y, mode="lines", name=SCORES[c]))
        fig2.update_layout(title="Individual scores over time", yaxis=dict(range=[0, 10.5]),
                           height=380, margin=dict(t=50, b=10))
        st.plotly_chart(fig2, width="stretch")
    with right:
        cats = list(SCORES.values())
        fig3 = go.Figure()
        fig3.add_trace(go.Scatterpolar(r=[base[c].mean() for c in SCORES], theta=cats, fill="toself", name="Baseline"))
        fig3.add_trace(go.Scatterpolar(r=[df.tail(7)[c].mean() for c in SCORES], theta=cats, fill="toself", name="Last 7 days"))
        fig3.update_layout(title="Baseline vs last 7 days", polar=dict(radialaxis=dict(range=[0, 10])),
                           height=380, margin=dict(t=50, b=10))
        st.plotly_chart(fig3, width="stretch")

    st.subheader("Recent check-ins")
    show = df.tail(10).iloc[::-1][["date", *SCORES, "summary", "risk_flag"]].copy()
    show["date"] = show["date"].dt.strftime("%d %b %Y")
    st.dataframe(show.rename(columns=SCORES), width="stretch", hide_index=True)

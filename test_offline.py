"""Checks the whole pipeline WITHOUT any API key (the LLM is replaced by a stub).
Run:  python test_offline.py"""
import os, tempfile
os.environ["CHECKINS_CSV"] = os.path.join(tempfile.mkdtemp(), "checkins.csv")
os.environ.pop("GOOGLE_SHEET_ID", None)

import agent, storage

calls = {"n": 0}
ORDER = ["mood", "sleep", "energy", "anxiety", "interest", "social"]


def fake(schema, system, messages):
    if schema is agent.TurnDecision:
        calls["n"] += 1
        k = min(calls["n"] + 1, 6)
        return agent.TurnDecision(reply="Thanks for sharing." if k == 6 else "Tell me more?",
                                  covered_areas=ORDER[:k], ready_to_score=(k == 6))
    return agent.CheckinScores(mood_score=4, sleep_score=3, energy_score=12, anxiety_score=7,
                               interest_score=4, social_score=3, summary="Rough day.", risk_flag=False)


agent.call_structured = fake
g = agent.build_graph()
st = agent.new_state()
for msg in ["bad day", "slept 4 hours", "so tired", "stressed", "no fun", "avoided friends"]:
    st = g.invoke({**st, "messages": st["messages"] + [{"role": "user", "content": msg}]})
    if st["done"]:
        break
df = storage.load_checkins()
assert st["done"] and st["saved"], st
assert len(df) == 1 and df.iloc[0]["energy_score"] == 10, df          # 12 was clamped to 10
print("conversation + scoring + save OK:", st["scores"])

# safety path
st2 = agent.new_state(); calls["n"] = 0
st2 = g.invoke({**st2, "messages": st2["messages"] + [{"role": "user", "content": "i want to die"}]})
assert "14416" in st2["messages"][-1]["content"] and st2["risk"]
print("safety branch OK")

# /done path
st3 = agent.new_state()
st3 = g.invoke({**st3, "messages": st3["messages"] + [{"role": "user", "content": "/done"}]})
assert st3["done"] and st3["saved"]
print("/done path OK")
print("ALL PASSED")

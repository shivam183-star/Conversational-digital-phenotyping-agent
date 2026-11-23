"""Daily reminder. Run:  python scheduler.py   (or  python scheduler.py --now  to fire once)."""
import os
import sys
import webbrowser

import requests
from apscheduler.schedulers.blocking import BlockingScheduler
from dotenv import load_dotenv

load_dotenv()
APP_URL = os.getenv("APP_URL", "http://localhost:8501")


def notify():
    msg = f"🌙 Time for your daily check-in! Open {APP_URL}"
    print(msg)
    token, chat = os.getenv("TELEGRAM_BOT_TOKEN"), os.getenv("TELEGRAM_CHAT_ID")
    if token and chat:
        try:
            requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                          json={"chat_id": chat, "text": msg}, timeout=10)
        except requests.RequestException as e:
            print("Telegram failed:", e)
    webbrowser.open(APP_URL)


if __name__ == "__main__":
    if "--now" in sys.argv:
        notify()
        sys.exit()
    hh, mm = os.getenv("CHECKIN_TIME", "21:00").split(":")
    sched = BlockingScheduler()
    sched.add_job(notify, "cron", hour=int(hh), minute=int(mm))
    print(f"Scheduler running. Daily check-in reminder at {hh}:{mm}. Ctrl+C to stop.")
    sched.start()

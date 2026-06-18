"""
Notifications. Supports Telegram bot and generic webhooks (Slack, Discord).

Telegram setup:
  1. Message @BotFather → /newbot → get token
  2. Message your bot → get chat_id via:
     curl https://api.telegram.org/bot<TOKEN>/getUpdates
  3. Set env vars: TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

Generic webhook (Slack / Discord):
  Set NOTIFY_WEBHOOK to the webhook URL.
"""
import json
import os
import urllib.request

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
WEBHOOK_URL = os.getenv("NOTIFY_WEBHOOK")


def send(message: str):
    """Send notification. Tries Telegram first, then webhook, then prints."""
    if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
        _telegram(message)
    elif WEBHOOK_URL:
        _webhook(message)
    else:
        print(f"[notify] {message}")
        print("[notify] tip: set TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID for phone alerts")


def _telegram(message: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = json.dumps({"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"}).encode()
    try:
        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=10)
        print(f"[notify] Telegram sent: {message[:60]}")
    except Exception as e:
        print(f"[notify] Telegram failed ({e}): {message}")


def _webhook(message: str):
    payload = json.dumps({"text": message, "content": message}).encode()
    try:
        req = urllib.request.Request(
            WEBHOOK_URL, data=payload,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        urllib.request.urlopen(req, timeout=10)
    except Exception as e:
        print(f"[notify] webhook failed ({e}): {message}")

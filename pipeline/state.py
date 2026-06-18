import json
from datetime import datetime
from pathlib import Path

ICONS = {"complete": "✅", "fallback": "⚡", "waiting": "⏳", "failed": "❌", "pending": "⬜"}


class State:
    def __init__(self, path: Path):
        self.path = path
        self.data = json.loads(path.read_text()) if path.exists() else {"version": 1, "phases": {}}

    def save(self):
        self.path.write_text(json.dumps(self.data, indent=2, ensure_ascii=False))

    def phase(self, name: str) -> dict:
        return self.data["phases"].get(name, {"status": "pending"})

    def is_done(self, name: str) -> bool:
        return self.phase(name).get("status") in ("complete", "fallback")

    def mark(self, name: str, status: str, **kwargs):
        self.data["phases"][name] = {"status": status, "ts": datetime.now().isoformat(), **kwargs}
        self.save()

    def summary(self) -> str:
        if not self.data["phases"]:
            return "  (nothing run yet)"
        lines = []
        for name, info in self.data["phases"].items():
            status = info.get("status", "pending")
            icon = ICONS.get(status, "?")
            reason = f" — {info['reason']}" if info.get("reason") else ""
            lines.append(f"  {icon} {name}: {status}{reason}")
        return "\n".join(lines)

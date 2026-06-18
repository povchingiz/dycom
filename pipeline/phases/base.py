from abc import ABC, abstractmethod
from pathlib import Path


class Phase(ABC):
    name: str

    def is_available(self) -> tuple[bool, str]:
        """Check required tools. Returns (ok, reason_if_not)."""
        return True, ""

    @abstractmethod
    def artifacts_exist(self, data_dir: Path) -> bool:
        """True if this phase output already exists on disk."""

    @abstractmethod
    def run(self, state, data_dir: Path) -> dict:
        """Run phase. Returns artifacts dict."""

    def run_fallback(self, state, data_dir: Path) -> dict:
        raise NotImplementedError

    def execute(self, state, data_dir: Path) -> bool:
        """Run with checkpoint / fallback / notification logic. Returns True if done."""
        from pipeline import notify

        if state.is_done(self.name):
            print(f"[{self.name}] already done, skipping")
            return True

        if self.artifacts_exist(data_dir):
            print(f"[{self.name}] artifacts detected, marking complete")
            state.mark(self.name, "complete", source="auto-detected")
            notify.send(f"✅ {self.name}: detected existing artifacts")
            return True

        ok, reason = self.is_available()

        if not ok:
            try:
                print(f"[{self.name}] {reason} — trying fallback")
                artifacts = self.run_fallback(state, data_dir)
                state.mark(self.name, "fallback", reason=reason, artifacts=artifacts)
                notify.send(f"⚡ {self.name}: fallback used\n{reason}")
                return True
            except NotImplementedError:
                print(f"[{self.name}] waiting: {reason}")
                state.mark(self.name, "waiting", reason=reason)
                notify.send(f"⏳ {self.name}: waiting\n{reason}")
                return False

        try:
            print(f"[{self.name}] running...")
            artifacts = self.run(state, data_dir)
            state.mark(self.name, "complete", artifacts=artifacts)
            notify.send(f"✅ {self.name}: complete")
            return True
        except Exception as e:
            try:
                print(f"[{self.name}] failed ({e}) — trying fallback")
                artifacts = self.run_fallback(state, data_dir)
                state.mark(self.name, "fallback", reason=str(e), artifacts=artifacts)
                notify.send(f"⚡ {self.name}: fallback after error\n{str(e)[:200]}")
                return True
            except (NotImplementedError, Exception) as fe:
                print(f"[{self.name}] fallback also failed: {fe}")
                state.mark(self.name, "failed", reason=str(e))
                notify.send(f"❌ {self.name}: FAILED\n{str(e)[:200]}")
                return False

#!/usr/bin/env python
"""
queue.py — ночная очередь под L40. Последовательно гоняет команды из очереди,
пишет статус на диск, не падает целиком если один эксперимент сломался.
Цель: GPU не простаивает ночью, пока ты спишь / идёшь по глубинным экспериментам днём.

Использование:
    python -m labkit.queue queue.txt
где queue.txt — по одной shell-команде на строку (пустые и # игнорируются).
"""
from __future__ import annotations
import json, subprocess, sys, time
from datetime import datetime
from pathlib import Path


def run_queue(queue_file: str, status_path: str = "logs/queue_status.jsonl"):
    Path(status_path).parent.mkdir(exist_ok=True)
    lines = [
        l.strip() for l in Path(queue_file).read_text().splitlines()
        if l.strip() and not l.strip().startswith("#")
    ]
    print(f"очередь: {len(lines)} задач")
    status = open(status_path, "a")

    for i, cmd in enumerate(lines, 1):
        rec = {"i": i, "cmd": cmd, "start": datetime.now().isoformat()}
        print(f"\n[{i}/{len(lines)}] {cmd}")
        t0 = time.time()
        try:
            r = subprocess.run(cmd, shell=True)
            rec["returncode"] = r.returncode
            rec["ok"] = r.returncode == 0
        except Exception as e:
            rec["returncode"] = -1
            rec["ok"] = False
            rec["error"] = str(e)
        rec["minutes"] = round((time.time() - t0) / 60, 1)
        rec["end"] = datetime.now().isoformat()
        status.write(json.dumps(rec) + "\n")
        status.flush()
        # один упавший эксперимент не должен убивать всю ночь
        print(f"  -> {'OK' if rec['ok'] else 'FAIL'} ({rec['minutes']} мин)")

    status.close()
    print(f"\nочередь завершена. статус: {status_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("укажи файл очереди, напр.: python -m labkit.queue queue.txt")
        sys.exit(1)
    run_queue(sys.argv[1])

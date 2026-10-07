#!/usr/bin/env python3
"""Stop hook: register a run's final RESULT block when the host runner does not (harnessreg).

The loop contract says "when your RESULT block says done, the harness checks the line and
appends the TASKLOG entry". This project's host runner registered 0 blocks (see
state/tickets/harnessreg.md), so the sandbox does it at stop time: parse the last assistant
message of the transcript, and on `RESULT: done` check the AGENT_TASK line in TODO.md
(exact text, falling back to the `(id: ...)` tag), append a TASKLOG.md entry, and commit
both files. Idempotent; always exits 0 so it can never block a session end.
"""

import json
import os
import re
import subprocess
import sys
import time

TASKLOG = "TASKLOG.md"
FIELDS = ("RESULT", "SUMMARY", "APPROACH", "BLOCKERS", "WHY", "NEEDED")


def main() -> int:
    try:
        data = json.load(sys.stdin)
    except (ValueError, OSError):
        return 0
    task = os.environ.get("AGENT_TASK", "").strip()
    if not task or data.get("stop_hook_active"):
        return 0
    cwd = data.get("cwd") or "/work"
    transcript = data.get("transcript_path") or ""
    if not os.path.exists(transcript):
        sid = data.get("session_id") or ""
        cand = os.path.expanduser(f"~/.claude/projects/-work/{sid}.jsonl")
        transcript = cand if os.path.exists(cand) else ""
    if not transcript:
        return 0
    block = final_block(transcript)
    audit(cwd, data.get("session_id", ""), (block or {}).get("RESULT", "none"), task)
    if not block or block["RESULT"] != "done":
        return 0
    src = os.environ.get("AGENT_TASK_SRC", "TODO.md")
    todo = os.path.join(cwd, src)
    tasklog = os.path.join(cwd, TASKLOG)
    changed = check_todo_line(todo, task) + append_tasklog(tasklog, task, block)
    if changed:
        commit(cwd, src, task)
    return 0


def audit(cwd: str, session_id: str, verdict: str, task: str) -> None:
    try:
        os.makedirs(os.path.join(cwd, ".agent-evidence"), exist_ok=True)
        import datetime

        line = f"{datetime.datetime.now().isoformat(timespec='seconds')} sid={session_id[:8]} "
        line += f"verdict={verdict} task={task_id(task) or task_title(task)}\n"
        with open(os.path.join(cwd, ".agent-evidence", "harnessreg-stops.log"), "a") as fh:
            fh.write(line)
    except OSError:
        pass


def final_block(path: str) -> dict | None:
    """Last assistant RESULT block. The final message is flushed to the transcript at
    stop time (observed: 9dcf5bf0/5b940b1d mtimes == Stop-hook fire time, both verdicts
    `none` then, block present afterwards), so when the first read has none, poll until
    it appears or the file settles for HARNESSREG_SETTLE s (cap HARNESSREG_WAIT s)."""
    settle = float(os.environ.get("HARNESSREG_SETTLE", "4"))
    deadline = time.monotonic() + float(os.environ.get("HARNESSREG_WAIT", "20"))
    prev, stable = -1, None
    while True:
        block = parse_block(last_assistant_text(path))
        if block:
            return block
        try:
            size = os.path.getsize(path)
        except OSError:
            size = -1
        if size == prev:
            if stable is None:
                stable = time.monotonic()
            elif time.monotonic() - stable >= settle:
                return None
        else:
            prev, stable = size, None
        if time.monotonic() >= deadline:
            return None
        time.sleep(1.0)


def last_assistant_text(path: str) -> str:
    last = ""
    with open(path, errors="replace") as fh:
        for line in fh:
            try:
                m = json.loads(line)
            except ValueError:
                continue
            if m.get("type") != "assistant":
                continue
            content = m.get("message", {}).get("content") or []
            text = " ".join(
                b.get("text", "")
                for b in content
                if isinstance(b, dict) and b.get("type") == "text"
            )
            if text.strip():
                last = text
    return last


def parse_block(text: str) -> dict | None:
    pat = re.compile(rf"^({'|'.join(FIELDS)}):\s*(.*)$")
    fields: dict[str, str] = {}
    for raw in text.splitlines():
        if m := pat.match(raw.strip()):
            fields[m.group(1)] = m.group(2).strip()
    if re.match(r"^(done|not done)$", fields.get("RESULT", "")):
        return fields
    return None


def task_id(task: str) -> str:
    m = re.search(r"\(id:\s*([\w-]+)\)", task)
    return m.group(1) if m else ""


def task_title(task: str) -> str:
    m = re.match(r"\*\*(.+?)\*\*", task)
    return m.group(1) if m else task[:60]


def check_todo_line(todo: str, task: str) -> int:
    if not os.path.exists(todo):
        return 0
    lines = open(todo, errors="replace").read().splitlines(keepends=True)
    want = f"- [ ] {task}".strip().lower()
    tid = task_id(task)
    for i, line in enumerate(lines):
        stripped = line.strip()
        by_text = stripped.lower() == want
        by_id = bool(tid) and stripped.startswith("- [ ]") and f"(id: {tid})" in line
        if by_text or by_id:
            lines[i] = line.replace("[ ]", "[x]", 1)
            open(todo, "w").writelines(lines)
            return 1
    return 0


def append_tasklog(tasklog: str, task: str, block: dict[str, str]) -> int:
    title, tid = task_title(task), task_id(task)
    header = f"## {tid or title} — {title}"
    if os.path.exists(tasklog) and header in open(tasklog, errors="replace").read():
        return 0
    goal = task.split("**", 2)[-1].strip().lstrip("-— ").strip()
    goal = re.split(r"\.\s| (?:Done when|id:)", goal)[0].strip()
    approach = " ".join(x for x in (block.get("SUMMARY"), block.get("APPROACH")) if x)
    entry = (
        f"{header}\n"
        f"- Goal: {goal[:300]}\n"
        f"- Approach: {approach[:600]}\n"
        f"- Blockers: {block.get('BLOCKERS') or 'none.'}\n"
    )
    with open(tasklog, "a") as fh:
        fh.write("\n" + entry)
    return 1


def commit(cwd: str, src: str, task: str) -> None:
    subprocess.run(["git", "-C", cwd, "add", "--", src, TASKLOG], capture_output=True)
    msg = f"register: {task_id(task) or task_title(task)} RESULT done (stop hook)"
    subprocess.run(["git", "-C", cwd, "commit", "-qm", msg], capture_output=True)


if __name__ == "__main__":
    sys.exit(main())

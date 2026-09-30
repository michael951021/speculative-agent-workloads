"""Stop-hook registration (harnessreg): parse a run's final RESULT block, check the TODO line,
append the TASKLOG entry, commit. Drives the real hook script with fixture repos and REAL
transcripts (hand-verified parses are the known answers)."""
import json
import os
import subprocess
from pathlib import Path

REPO = Path(__file__).parents[2]
HOOK = REPO / ".claude" / "hooks" / "register_result.py"
TRANSCRIPTS = Path(os.path.expanduser("~/.claude/projects/-work"))

DISTILL_TASK = (
    "**Distilled tiny drafter (JAX)** — draftlab/drafters/distill.py: cache T's top-64 logits on "
    "train blocks; train a 4-layer d=256 Qwen2-shaped student; val KL falls >= 30%. "
    "(id: distill) (after: alpha, jaxmodel)"
)
TASKLOG_HDR = "# Task log\n"


def make_repo(tmp_path: Path, task: str = DISTILL_TASK) -> Path:
    (tmp_path / "TODO.md").write_text(f"# Tasks\n\n- [ ] {task}\n")
    (tmp_path / "TASKLOG.md").write_text(TASKLOG_HDR)
    (tmp_path / ".gitignore").write_text(".agent-evidence/\n")  # as in /work
    for args in (["init", "-q"], ["config", "user.email", "t@t"],
                 ["config", "user.name", "t"], ["add", "-A"], ["commit", "-qm", "init"]):
        subprocess.run(["git", "-C", str(tmp_path), *args], check=True)
    return tmp_path


def run_hook(repo: Path, transcript: Path, task: str,
             active: bool = False) -> subprocess.CompletedProcess:
    payload = {"transcript_path": str(transcript), "session_id": "x", "cwd": str(repo),
               "hook_event_name": "Stop", "stop_hook_active": active}
    env = {**os.environ, "AGENT_TASK": task, "AGENT_TASK_SRC": "TODO.md"}
    return subprocess.run(["python3", str(HOOK)], input=json.dumps(payload), text=True,
                          env=env, capture_output=True, timeout=60)


def git_count(repo: Path) -> str:
    r = subprocess.run(["git", "-C", str(repo), "rev-list", "--count", "HEAD"],
                       capture_output=True, text=True)
    return r.stdout.strip()


def real_transcript(prefix: str) -> Path:
    return sorted(TRANSCRIPTS.glob(f"{prefix}*.jsonl"))[0]


def test_done_registers_line_tasklog_and_commit(tmp_path):
    repo = make_repo(tmp_path)
    r = run_hook(repo, real_transcript("db45b037"), DISTILL_TASK)
    assert r.returncode == 0, r.stderr
    todo = (repo / "TODO.md").read_text()
    assert "- [x] **Distilled tiny drafter" in todo
    log = (repo / "TASKLOG.md").read_text()
    assert "## distill — Distilled tiny drafter (JAX)" in log
    assert "30.6%" in log  # known: hand-verified from db45b037's SUMMARY
    # commit happened: tree clean, one new commit
    status = subprocess.run(["git", "-C", str(repo), "status", "--porcelain"],
                            capture_output=True, text=True)
    assert status.stdout.strip() == ""
    assert git_count(repo) == "2"


def test_reregister_is_noop(tmp_path):
    repo = make_repo(tmp_path)
    run_hook(repo, real_transcript("db45b037"), DISTILL_TASK)
    before = (repo / "TODO.md").read_text() + (repo / "TASKLOG.md").read_text()
    assert run_hook(repo, real_transcript("db45b037"), DISTILL_TASK).returncode == 0
    assert before == (repo / "TODO.md").read_text() + (repo / "TASKLOG.md").read_text()
    assert git_count(repo) == "2"


def test_title_edit_still_matches_by_id(tmp_path):
    task = DISTILL_TASK.replace("Distilled tiny drafter (JAX)", "Student drafter")
    repo = make_repo(tmp_path, task=task)
    r = run_hook(repo, real_transcript("db45b037"), DISTILL_TASK)
    assert r.returncode == 0, r.stderr
    assert "- [x] **Student drafter**" in (repo / "TODO.md").read_text()


def test_prose_prefixed_fenced_block_registers(tmp_path):
    # f2542893 ends with prose + fenced RESULT block; field values hand-verified this run
    repo = make_repo(tmp_path)
    r = run_hook(repo, real_transcript("f2542893"), DISTILL_TASK)
    assert r.returncode == 0, r.stderr
    assert "- [x] **Distilled tiny drafter" in (repo / "TODO.md").read_text()
    log = (repo / "TASKLOG.md").read_text()
    assert "0.0105" in log and "harnessreg" in log


def test_not_done_and_missing_block_are_noops(tmp_path):
    notdone = tmp_path / "notdone.jsonl"
    msg = {"type": "assistant", "message": {"content": [
        {"type": "text", "text": "RESULT: not done\nWHY: blocked on GPU"}]}}
    notdone.write_text(json.dumps(msg) + "\n")
    repo = make_repo(tmp_path)
    assert run_hook(repo, notdone, DISTILL_TASK).returncode == 0
    assert "- [ ] **Distilled tiny drafter" in (repo / "TODO.md").read_text()
    assert (repo / "TASKLOG.md").read_text() == TASKLOG_HDR

    none = tmp_path / "none.jsonl"
    msg = {"type": "assistant", "message": {"content": [
        {"type": "text", "text": "Just some prose."}]}}
    none.write_text(json.dumps(msg) + "\n")
    assert run_hook(repo, none, DISTILL_TASK).returncode == 0
    assert "- [ ] **Distilled tiny drafter" in (repo / "TODO.md").read_text()
    assert git_count(repo) == "1"


def test_chat_run_without_task_env_is_noop(tmp_path):
    repo = make_repo(tmp_path)
    payload = {"transcript_path": str(real_transcript("db45b037")), "session_id": "x",
               "cwd": str(repo), "hook_event_name": "Stop", "stop_hook_active": False}
    env = {**os.environ}
    env.pop("AGENT_TASK", None)
    r = subprocess.run(["python3", str(HOOK)], input=json.dumps(payload), text=True,
                       env=env, capture_output=True)
    assert r.returncode == 0
    assert "- [ ] **Distilled tiny drafter" in (repo / "TODO.md").read_text()

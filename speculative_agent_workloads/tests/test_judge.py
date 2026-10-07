"""Judge pipeline against a fake OpenAI server: verdict mapping, order bias
(always-A judge => disagreement 1.0), identical outputs => tie, caching,
call cap, invalid JSON retry, and Wilson CIs vs hand-computed values."""

import hashlib
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from speculative_agent_workloads.eval import judge as J


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        RULE["n"] += 1
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n))
        content = RULE["fn"](body)
        payload = {
            "id": "x",
            "object": "chat.completion",
            "created": 0,
            "model": body.get("model", "fake"),
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": content},
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        data = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


RULE: dict = {"fn": None, "n": 0}


def _extract(body):
    u = [m["content"] for m in body["messages"] if m["role"] == "user"][0]
    a = u.split("RESPONSE A:\n", 1)[1].split("\n\nRESPONSE B:", 1)[0]
    b = u.split("RESPONSE B:\n", 1)[1]
    return u, a, b


@pytest.fixture()
def server():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}/v1"
    srv.shutdown()
    RULE["n"] = 0


def _mk(n=2, n_drafter=None):
    n_drafter = n if n_drafter is None else n_drafter
    prompts = [{"run": f"r{i}", "seq": i, "kind": "text", "text": f"prompt {i}"} for i in range(n)]
    outputs = {
        f"{p['run']}:{p['seq']}:{m}": f"{m} output {i}"
        for i, p in enumerate(prompts)
        for m in ("target", "spec", "drafter_alone")
    }
    return prompts, outputs, J.build_judge_jobs(prompts, n_drafter)


def _run(server, tmp_path, rule, n=2, n_drafter=None, max_calls=100, prompts=None, outputs=None):
    RULE["fn"] = rule
    p, o, jobs = (
        _mk(n, n_drafter)
        if prompts is None
        else (prompts, outputs, J.build_judge_jobs(prompts, n_drafter))
    )
    client = J.JudgeClient(server, "fake", tmp_path / "cache", max_calls=max_calls)
    verdicts = J.judge_all(client, jobs, p, o)
    return client, J.summarize(jobs, verdicts)


def test_always_a_order_bias_one(server, tmp_path):
    """A judge that always picks A says 'spec wins' in one order and 'spec
    loses' in the other: disagreement exactly 1.0, wins == losses."""
    client, rep = _run(server, tmp_path, lambda b: '{"winner": "A", "reason": "r"}')
    assert client.calls == 8  # 2 prompts x 2 comparisons x 2 orders
    for comp in ("spec_vs_target", "spec_vs_drafter_alone"):
        r = rep[comp]
        assert r["order_bias_disagreement"] == 1.0
        assert r["wins"] == r["losses"] == 2 and r["ties"] == 0
        assert r["win_ci95"][0] < r["win_rate"] < r["win_ci95"][1]


def test_identical_outputs_tie(server, tmp_path):
    def rule(body):
        _, a, b = _extract(body)
        w = "tie" if a == b else "A"
        return json.dumps({"winner": w, "reason": "r"})

    prompts, outputs, jobs = _mk(2)
    for key in [k for k in outputs if k.endswith(":spec")]:
        outputs[key] = outputs[key.replace(":spec", ":target")]  # spec == target
    client, rep = _run(server, tmp_path, rule, prompts=prompts, outputs=outputs)
    r = rep["spec_vs_target"]
    assert r["wins"] == 0 and r["losses"] == 0 and r["ties"] == 4
    assert r["order_bias_disagreement"] == 0.0
    assert r["win_ci95"][0] == 0.0  # 0 wins in 4: lower bound exactly 0


def test_cache_second_run_zero_calls(server, tmp_path):
    def rule(body):
        u, _, _ = _extract(body)
        w = "A" if int(hashlib.sha1(u.encode()).hexdigest(), 16) % 2 else "B"
        return json.dumps({"winner": w, "reason": "r"})

    c1, rep1 = _run(server, tmp_path, rule)
    assert c1.calls > 0
    RULE["n"] = 0
    c2, rep2 = _run(server, tmp_path, rule)
    assert c2.calls == 0 and c2.cache_hits == 8
    assert rep1 == rep2


def test_call_cap_respected(server, tmp_path):
    c, rep = _run(
        server, tmp_path, lambda b: '{"winner": "B", "reason": "r"}', n=3, n_drafter=3, max_calls=5
    )
    assert RULE["n"] == 5 == c.calls  # server saw exactly 5 requests
    n_judged = sum(r["n_verdicts"] for k, r in rep.items() if k != "meta")
    assert n_judged == 5


def test_invalid_json_retries_once(server, tmp_path):
    c, _ = _run(server, tmp_path, lambda b: "not json at all", n=1, n_drafter=1)
    assert c.calls == 8  # 4 jobs x (1 + 1 retry)
    assert RULE["n"] == c.calls


def test_parse_verdict():
    assert J.parse_verdict('{"winner": "A", "reason": "x"}')["winner"] == "A"
    assert J.parse_verdict('```json\n{"winner": "tie", "reason": "x"}\n```')["winner"] == "TIE"
    assert J.parse_verdict('Sure: {"winner": "b", "reason": "x"}')["winner"] == "B"
    assert J.parse_verdict("no json here") is None
    assert J.parse_verdict('{"winner": "C", "reason": "x"}') is None


def test_wilson_known_values():
    lo, hi = J.wilson(0, 10)
    assert abs(lo - 0.0) < 1e-9 and abs(hi - 0.2776) < 1e-3
    lo, hi = J.wilson(10, 10)
    assert abs(lo - 0.7224) < 1e-3 and abs(hi - 1.0) < 1e-9
    lo, hi = J.wilson(5, 10)
    assert abs((lo + hi) / 2 - 0.5) < 1e-9

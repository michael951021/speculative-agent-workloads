"""LoRA drafter: fresh adapter is exactly identity; 2-rank FSDP (gloo/CPU) training
decreases loss and every rank saves bit-identical lora weights; --qlora exits on CPU."""

import json
import socket
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
LORA_PY = ROOT / "speculative_agent_workloads" / "drafters" / "lora.py"
TARGETS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def _tiny_model(seed: int = 0) -> torch.nn.Module:
    from transformers import Qwen2ForCausalLM

    from speculative_agent_workloads.models import tiny_config

    torch.manual_seed(seed)
    return Qwen2ForCausalLM(tiny_config()).eval()


def _lora_cfg():
    from peft import LoraConfig

    return LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.0,
        bias="none",
        target_modules=TARGETS,
    )


def _subprocess(env_extra: dict | None = None) -> dict:
    import os

    env = dict(os.environ)
    env.update(env_extra or {})
    return {"env": env, "cwd": str(ROOT)}


def test_lora_init_is_identity():
    """lora_B = 0 at init => base + adapter logits equal base logits exactly."""
    from peft import get_peft_model

    base = _tiny_model()
    ids = torch.randint(0, 512, (1, 64))
    with torch.no_grad():
        ref = base(input_ids=ids).logits
    pm = get_peft_model(base, _lora_cfg())
    pm.eval()
    with torch.no_grad():
        out = pm(input_ids=ids).logits
    assert (out - ref).abs().max().item() == 0.0
    b_norms = [p.abs().max().item() for n, p in pm.named_parameters() if "lora_B" in n]
    assert b_norms and all(v == 0.0 for v in b_norms)


def test_step_loss_matches_hf_labels_loss():
    """Known answer: on a fully-valid chunk, step_loss must equal the transformers
    `model(ids, labels=ids).loss` (HF shifts internally) — the alignment contract."""
    from speculative_agent_workloads.drafters.lora import step_loss

    base = _tiny_model()
    ids = torch.randint(0, 512, (1, 64))
    base.eval()
    with torch.no_grad():
        hf = base(input_ids=ids, labels=ids).loss
        ours = step_loss(base, ids[0], n_valid=64)
    assert (hf - ours).abs().item() < 1e-5, f"{hf} vs {ours}"


def _run_train(args: list[str], timeout: int = 600) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "torch.distributed.run", *args]
        if "--fsdp" in args
        else [sys.executable, "-m", "speculative_agent_workloads.drafters.lora", *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        **_subprocess(),
    )


def test_plain_tiny_train_saves_loadable_adapter(tmp_path):
    out = tmp_path / "lora"
    p = _run_train(["train", "--tiny", "--steps", "3", "--seq-len", "64", "--out", str(out)])
    assert p.returncode == 0, p.stderr[-4000:]
    assert (out / "adapter_config.json").exists()
    from peft import PeftModel

    m = PeftModel.from_pretrained(_tiny_model(), out)  # HFDrafter-style load: base + adapter
    ws = [w for n, w in m.named_parameters() if "lora_B" in n]
    assert any(w.abs().max().item() > 0 for w in ws), "adapter weights did not move"


def test_fsdp_two_ranks_identical_weights(tmp_path):
    out = tmp_path / "fsdp"
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    p = _run_train(
        [
            "--nproc_per_node=2",
            f"--master_port={port}",
            str(LORA_PY),
            "train",
            "--tiny",
            "--fsdp",
            "--steps",
            "5",
            "--seq-len",
            "64",
            "--lr",
            "3e-4",
            "--seed",
            "0",
            "--log-every",
            "1",
            "--out",
            str(out),
        ]
    )
    assert p.returncode == 0, p.stderr[-4000:]
    import re

    recs = [
        json.loads(m)
        for m in re.findall(r'\{"step": \d+, "rank": \d+, "loss": -?\d+\.\d+\}', p.stdout)
    ]
    per_rank: dict[int, list[float]] = {}
    for r in recs:
        per_rank.setdefault(int(r["rank"]), []).append(float(r["loss"]))
    assert set(per_rank) == {0, 1}, f"need per-rank loss lines, got {per_rank.keys()}"
    for losses in per_rank.values():
        assert len(losses) >= 5 and losses[-1] < losses[0], f"loss not decreasing: {losses}"
    assert per_rank[0] == per_rank[1], "ranks disagree on losses (should run identical data)"
    from safetensors.torch import load_file

    f0, f1 = out / "rank0_lora.safetensors", out / "rank1_lora.safetensors"
    assert f0.exists() and f1.exists()
    t0, t1 = load_file(str(f0)), load_file(str(f1))
    assert set(t0) == set(t1) and any("lora_A" in k for k in t0)
    for k in t0:
        assert torch.equal(t0[k], t1[k]), f"rank weights differ at {k}"


def test_qlora_cpu_exits_with_clear_message():
    p = subprocess.run(
        [
            sys.executable,
            "-m",
            "speculative_agent_workloads.drafters.lora",
            "train",
            "--tiny",
            "--qlora",
        ],
        capture_output=True,
        text=True,
        timeout=300,
        **_subprocess(),
    )
    text = p.stdout + p.stderr
    assert p.returncode != 0
    assert "GPU" in text and "CUDA" in text, text[-2000:]

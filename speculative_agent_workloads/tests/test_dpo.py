"""Acceptance-aware DPO: pair builder on a fake spec log (exact pairs), spec-step
marginal == target p (chi2, known property of speculative sampling), tiny DPO run
raises the chosen/rejected margin."""

import tempfile

import numpy as np
import torch


def test_pairs_from_fake_log():
    """Hand-computed log: rejects resample deterministically (single-mass residual),
    accept produces no pair."""
    from speculative_agent_workloads.drafters.dpo import pairs_from_spec_log

    def p_row(x):
        return np.asarray(x, dtype=np.float64)

    log = [
        # step 0: u*q[d]=0.45 >= p[0]=0.1 -> reject; residual mass only at 1 -> b=1
        {
            "ctx": [1, 2],
            "draft": 0,
            "u": 0.5,
            "p": p_row([0.1, 0.9, 0.0, 0.0, 0.0]),
            "q": p_row([0.9, 0.05, 0.02, 0.02, 0.01]),
        },
        # step 1: u*q[1]=0.08 < p[1]=0.6 -> accept, no pair
        {
            "ctx": [1, 2, 1],
            "draft": 1,
            "u": 0.1,
            "p": p_row([0.2, 0.6, 0.2, 0.0, 0.0]),
            "q": p_row([0.1, 0.8, 0.05, 0.02, 0.03]),
        },
        # step 2: u*q[2]=0.36 >= p[2]=0.2 -> reject; residual only at 1 -> b=1
        {
            "ctx": [1, 2, 1, 1],
            "draft": 2,
            "u": 0.9,
            "p": p_row([0.1, 0.7, 0.2, 0.0, 0.0]),
            "q": p_row([0.1, 0.5, 0.4, 0.0, 0.0]),
        },
    ]
    pairs, stats = pairs_from_spec_log(log, seed=0)
    assert stats["n_steps"] == 3
    assert stats["n_accepted"] == 1
    assert len(pairs) == 2
    assert pairs[0] == {"ctx": [1, 2], "chosen": 1, "rejected": 0}
    assert pairs[1] == {"ctx": [1, 2, 1, 1], "chosen": 1, "rejected": 2}


def test_spec_step_marginal_is_target():
    """Known property of speculative sampling: the committed token is exactly
    distributed as the target p. Toy case with hand-computable expectations:
    d=0 always, accept prob min(1,p0/q0); residual on reject is [0, 0.25, 0.05]/0.3.
    => P(out) = p = [0.6, 0.3, 0.1] exactly."""
    from speculative_agent_workloads.drafters.dpo import accept_step

    p = np.array([0.6, 0.3, 0.1])
    q = np.array([0.9, 0.05, 0.05])
    d = 0
    rng = np.random.default_rng(0)
    n = 30_000
    outs = np.array([accept_step(p, q, d, float(rng.random()), rng) for _ in range(n)])
    obs = np.bincount(outs, minlength=3) / n
    chi2 = float((((obs - p) ** 2) / p).sum())
    # dof=2, chi2=13.82 is p=0.001; 30k draws should be far inside
    assert chi2 < 13.82, f"chi2={chi2:.2f}, obs={obs}"


def _tiny_dpo_setup(tmp: str, seed: int = 0):
    from peft import LoraConfig, get_peft_model
    from tokenizers import Tokenizer
    from tokenizers.decoders import ByteLevel as ByteDec
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import ByteLevel as BytePre
    from transformers import PreTrainedTokenizerFast, Qwen2ForCausalLM

    from speculative_agent_workloads.models import tiny_config

    t = Tokenizer(BPE())
    t.pre_tokenizer = BytePre()
    t.decoder = ByteDec()
    t.add_tokens([f"tk{i:03d}" for i in range(512)])
    tok_file = f"{tmp}/tk.json"
    t.save(tok_file)
    tok = PreTrainedTokenizerFast(tokenizer_file=tok_file, pad_token="tk000", unk_token=None)
    torch.manual_seed(seed)
    base = Qwen2ForCausalLM(tiny_config())
    pm = get_peft_model(base, LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"]))
    g = torch.Generator().manual_seed(seed)
    ds = []
    for _ in range(32):
        ctx = torch.randint(0, 512, (16,), generator=g).tolist()
        c = int(torch.randint(0, 512, (1,), generator=g).item())
        r = (c + int(torch.randint(1, 512, (1,), generator=g).item())) % 512
        ds.append({"ctx": ctx, "chosen": c, "rejected": r})
    return pm, tok, ds


def test_tiny_dpo_raises_margin():
    """DPO's objective directly targets log pi(c|x) - log pi(r|x): after a short
    run the mean margin on the train pairs must increase."""
    from datasets import Dataset
    from trl import DPOConfig

    from speculative_agent_workloads.drafters.dpo import PreTokDPOTrainer, pair_margin

    with tempfile.TemporaryDirectory() as tmp:
        pm, tok, ds = _tiny_dpo_setup(tmp)
        pm.eval()

        def margins():
            with torch.no_grad():
                return [pair_margin(pm, ex["ctx"], ex["chosen"], ex["rejected"]) for ex in ds]

        before = float(np.mean(margins()))
        cfg = DPOConfig(
            output_dir=tmp,
            max_steps=40,
            per_device_train_batch_size=8,
            learning_rate=1e-2,
            beta=0.1,
            max_length=17,
            use_cpu=True,
            report_to=[],
            save_strategy="no",
            optim="adamw_torch",
        )
        tr = PreTokDPOTrainer(
            model=pm,
            ref_model=None,
            args=cfg,
            train_dataset=Dataset.from_list(ds),
            processing_class=tok,
        )
        tr.train()
        after = float(np.mean(margins()))
        assert after > before, f"margin did not increase: {before:.4f} -> {after:.4f}"

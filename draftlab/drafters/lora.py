"""LoRA drafter (PyTorch + PEFT): SFT of Qwen2.5-0.5B on the 27B agent's own output.

- LoRA r=16, alpha=32 on q/k/v/o + gate/up/down (attention+MLP), f32, AdamW.
- Sequence packing: train blocks shuffled (seed), token stream chunked into
  --seq-len, padding tail ignored via ignore_index. Chunk boundaries reset
  context and supervision (no cross-chunk labels); standard packing, noted here.
- --fsdp: torchrun N ranks, gloo on CPU / nccl on GPU, FSDP2 (fully_shard,
  DTensor; FSDP1 dropped CPU in torch>=2.14). All ranks train on the SAME chunk
  order (no sharding) so runs are bitwise identical across ranks; each rank saves
  gathered lora tensors to rank{i}_lora.safetensors, rank 0 additionally writes
  the PEFT adapter dir.
- --qlora: bitsandbytes 4-bit base + LoRA. GPU only: on CPU exits 1 with a clear
  message (bitsandbytes is CUDA-only; install on the host).

CLI: python -m draftlab.drafters.lora train [--steps 300] [--seq-len 512]
     [--lr 2e-4] [--model models/qwen2.5-0.5b] [--out ckpt/lora]
     [--fsdp] [--qlora] [--tiny] [--seed 0]
torchrun: torchrun --nproc_per_node 2 draftlab/drafters/lora.py train --fsdp ...
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

LORA_TARGETS = [
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
]


def rank_world() -> tuple[int, int]:
    return int(os.environ.get("RANK", "0")), int(os.environ.get("WORLD_SIZE", "1"))


def lora_config(r: int = 16, alpha: int = 32):
    from peft import LoraConfig

    return LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=0.0,
        bias="none",
        target_modules=LORA_TARGETS,
    )


def build_base(model: str | None, tiny: bool, qlora: bool, seed: int) -> torch.nn.Module:
    """Base causal LM in f32 (4-bit under --qlora). Deterministic tiny init."""
    if tiny:
        from transformers import Qwen2ForCausalLM

        from draftlab.models import tiny_config

        torch.manual_seed(seed)
        return Qwen2ForCausalLM(tiny_config())
    from transformers import AutoModelForCausalLM

    kwargs: dict = {"dtype": torch.float32}
    if qlora:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    return AutoModelForCausalLM.from_pretrained(model, **kwargs)


def packed_chunks(model: str | None, tiny: bool, seq_len: int, n_tokens: int,
                  seed: int) -> Iterator[tuple[torch.Tensor, int]]:
    """Yield (chunk (seq_len,), n_valid). Tiny: a repeated 8-token motif
    (deterministic, so next-token prediction is learnable — mechanism test,
    no corpus). Real: shuffled train blocks packed."""
    if tiny:
        motif = [11, 17, 23, 29, 31, 37, 41, 47]
        reps = (n_tokens + seq_len) // len(motif) + 1
        stream = (motif * reps)[: n_tokens + seq_len]
    else:
        import pyarrow.parquet as pq
        from transformers import AutoTokenizer

        t = pq.read_table(_ROOT / "data" / "corpus" / "train.parquet").to_pydict()
        idx = torch.randperm(len(t["run"]), generator=torch.Generator().manual_seed(seed)).tolist()
        tok = AutoTokenizer.from_pretrained(model)
        stream: list[int] = []
        for i in idx:
            if len(stream) >= n_tokens + seq_len:
                break
            stream.extend(tok.encode(t["text"][i], add_special_tokens=False))
    for s in range(0, len(stream) - seq_len + 1, seq_len):
        yield torch.tensor(stream[s : s + seq_len], dtype=torch.long), seq_len


def step_loss(model: torch.nn.Module, chunk: torch.Tensor, n_valid: int) -> torch.Tensor:
    """Next-token CE with HF alignment: logits row t predicts ids[t+1]
    (== model(ids, labels=ids).loss semantics), padded tail ignored."""
    ids = chunk.unsqueeze(0)
    labels = ids.clone()
    labels[0, n_valid:] = -100
    logits = model(input_ids=ids).logits
    return F.cross_entropy(
        logits[:, :-1].reshape(-1, logits.shape[-1]),
        labels[:, 1:].reshape(-1),
        ignore_index=-100,
    )


def fsdp_wrap(model: torch.nn.Module, world: int) -> torch.nn.Module:
    """FSDP2: shard each decoder layer, then the root. CPU(gloo)/GPU(nccl)."""
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mesh = init_device_mesh(dev, (world,))
    for m in list(model.modules()):
        if m.__class__.__name__ == "Qwen2DecoderLayer":
            fully_shard(m, mesh=mesh)
    fully_shard(model, mesh=mesh)
    return model


def gather_lora(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Full (FSDP2-gathered) lora tensors, CPU. All ranks must call it."""
    sd = {}
    for k, v in model.state_dict().items():
        if "lora_" in k:
            sd[k] = (v.full_tensor() if hasattr(v, "full_tensor") else v.detach()).cpu()
    return {k: sd[k] for k in sorted(sd)}


def train(args: argparse.Namespace) -> None:
    from peft import get_peft_model

    if args.qlora:
        if not torch.cuda.is_available():
            raise SystemExit(
                "--qlora requires a GPU: bitsandbytes 4-bit is CUDA-only and no CUDA "
                "device is available here. Run on the host GPU, or drop --qlora."
            )
        try:
            import bitsandbytes  # noqa: F401
        except ImportError as err:
            raise SystemExit("--qlora requires the 'bitsandbytes' package (CUDA).") from err

    torch.manual_seed(args.seed)
    rank, world = rank_world()
    is_fsdp = args.fsdp and world > 1
    if is_fsdp:
        dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cfg = lora_config(args.lora_r, args.lora_alpha)
    model = get_peft_model(build_base(args.model, args.tiny, args.qlora, args.seed), cfg)
    model.train()
    if is_fsdp:
        model = fsdp_wrap(model, world)
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0,
    )

    n_tokens = args.steps * args.seq_len + args.seq_len
    chunks = packed_chunks(args.model, args.tiny, args.seq_len, n_tokens, args.seed)
    t0 = time.perf_counter()
    for s in range(args.steps):
        chunk, n_valid = next(chunks)
        loss = step_loss(model, chunk, n_valid)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if s == 0 or s == args.steps - 1 or (s + 1) % args.log_every == 0:
            print(json.dumps({"step": s, "rank": rank, "loss": round(float(loss), 6)}),
                  flush=True)
    dt = time.perf_counter() - t0
    print(f"train: steps={args.steps} last_loss={float(loss):.4f} "
          f"dt={dt:.1f}s ({dt / args.steps:.2f}s/step)", file=sys.stderr, flush=True)

    # every rank: gathered lora tensors (bitwise identical across ranks)
    sd = gather_lora(model)
    from safetensors.torch import save_file

    save_file(sd, str(out / f"rank{rank}_lora.safetensors"))
    if not is_fsdp:
        model.peft_config["default"].base_model_name_or_path = (
            "tiny" if args.tiny else str(Path(args.model).resolve())
        )
        model.save_pretrained(out)
    else:
        dist.barrier()
        if rank == 0:
            # rebuild a plain PeftModel from base + gathered lora tensors, save dir
            fresh = get_peft_model(build_base(args.model, args.tiny, args.qlora, args.seed),
                                   lora_config(args.lora_r, args.lora_alpha))
            missing, unexpected = fresh.load_state_dict(sd, strict=False)
            assert not unexpected, unexpected
            fresh.peft_config["default"].base_model_name_or_path = (
                "tiny" if args.tiny else str(Path(args.model).resolve())
            )
            fresh.save_pretrained(out)
        dist.barrier()
        dist.destroy_process_group()
    print(f"saved adapter -> {out}", file=sys.stderr, flush=True)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m draftlab.drafters.lora")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("train")
    p.add_argument("--model", default="models/qwen2.5-0.5b")
    p.add_argument("--out", default="ckpt/lora")
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--fsdp", action="store_true", help="FSDP multi-rank (launch via torchrun)")
    p.add_argument("--qlora", action="store_true", help="4-bit base (GPU only)")
    p.add_argument("--tiny", action="store_true", help="tiny_config + random stream (tests)")
    args = ap.parse_args(argv)
    if args.cmd == "train":
        train(args)


if __name__ == "__main__":
    main()

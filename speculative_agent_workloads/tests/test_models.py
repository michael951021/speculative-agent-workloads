"""Models: tiny_config fields; shared-vocab invariant when checkpoints are present."""

import pytest

from speculative_agent_workloads.models import MODEL_DIR, MODELS, tiny_config


def test_tiny_config():
    c = tiny_config()
    assert c.num_hidden_layers == 2
    assert c.hidden_size == 64
    assert c.vocab_size == 512
    assert c.intermediate_size == 128
    assert c.num_attention_heads == 4
    assert c.num_key_value_heads == 2
    assert c.head_dim == 16


def test_models_map():
    assert set(MODELS) == {"qwen2.5-0.5b-instruct", "qwen2.5-1.5b-instruct", "qwen2.5-0.5b"}


@pytest.mark.skipif(
    not all((MODEL_DIR / k / "config.json").exists() for k in MODELS),
    reason="models not fetched",
)
def test_shared_vocab():
    from speculative_agent_workloads.models import info

    rows = info(MODEL_DIR)
    assert len(rows) == 3
    assert all(r["vocab"] == rows[0]["vocab"] for r in rows)

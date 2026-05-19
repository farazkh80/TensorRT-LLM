# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Reconstructed parity test from the DSV3-Lite spike's summary.md.
# Original file at `tests/unittest/_torch/attention/test_tokenspeed_mla.py`
# (host source-tree path) was not committed to git and was deleted between
# the spike and the K2.6 follow-up. This reconstruction covers the same
# matrix the spike documented:
#
#     num_heads ∈ {16, 32}
#     dtype     ∈ {bf16, fp16}
#     shape     ∈ {bs1_qlen1, bs4_qlen1_varlen, bs8_qlen4_spec}
#     plus 1 sinks-rejection negative test
#
# Spike-documented results (DSV3-Lite NVFP4 / B300 / sm_103 / flashinfer 0.6.9):
#   bs1_qlen1 × num_heads∈{16,32} × bf16        → PASS
#   bs4_qlen1_varlen × num_heads∈{16,32} × bf16 → PASS
#   bs8_qlen4_spec × num_heads∈{16,32} × bf16   → FAIL  (0.9% elements,
#                                                        max abs 0.33,
#                                                        max rel ~1166×)
#   all fp16 cases                              → SKIPPED (no flashinfer fp16
#                                                          MLA cubin on sm_103)
#   sinks rejection                              → PASS
#
# The spec-decode FAILures are the regime where TokenSpeed's fold_sq_factor
# reorders queries into the head axis; the divergence is the open question
# for Albert Di.
"""Parity tests for the TokenSpeed MLA decode wrapper vs FlashInfer."""

from __future__ import annotations

import math

import pytest
import torch
from parameterized import parameterized

from tensorrt_llm._torch.attention_backend.tokenspeed_mla import (
    is_tokenspeed_mla_available,
    tokenspeed_batch_decode_with_kv_cache_mla,
)


# --- gates ---------------------------------------------------------------

def _skip_if_unsupported() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    major, _ = torch.cuda.get_device_capability()
    if major != 10:
        pytest.skip("TokenSpeed MLA requires data-center Blackwell (sm_10x)")
    if not is_tokenspeed_mla_available():
        pytest.skip("tokenspeed_mla package not importable")
    try:
        import flashinfer  # noqa: F401
    except Exception:
        pytest.skip("flashinfer baseline unavailable")


# --- MLA dim constants (match DSV3-Lite / Kimi-K2-Thinking architecture) -

KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
QK_NOPE_HEAD_DIM = 128
D_QK = KV_LORA_RANK + QK_ROPE_HEAD_DIM  # 576
D_V = KV_LORA_RANK                      # 512


# --- helpers -------------------------------------------------------------

def _make_workspace(B: int, H: int, q_len: int, device, num_sms: int = 132):
    """Workspace sizing per tokenspeed_mla docstring:
       B * H * q_len * split_kv * (kv_lora_rank + 1) * 4 bytes.
       Use a generous over-allocation; spike used 32 MiB which is plenty."""
    return torch.empty((32 * 1024 * 1024,), dtype=torch.uint8, device=device)


def _make_kv_cache(
    *,
    num_blocks: int,
    page_size: int,
    device,
    dtype: torch.dtype,
):
    return torch.randn(
        (num_blocks, page_size, D_QK),
        dtype=dtype, device=device,
    ) * 0.1


def _make_block_tables(B: int, max_pages: int, num_blocks: int, device):
    perm = torch.randperm(num_blocks - 1, device=device)[: B * max_pages] + 1
    return perm.view(B, max_pages).to(torch.int32)


def _make_seq_lens(B: int, low: int, high: int, device):
    return torch.randint(
        low=low, high=high + 1, size=(B,),
        device=device, dtype=torch.int32,
    )


def _make_query(B: int, q_len: int, H: int, device, dtype: torch.dtype):
    return torch.randn(
        (B, q_len, H, D_QK), dtype=dtype, device=device,
    ) * 0.1


# --- parametrized parity test --------------------------------------------

SHAPES = [
    # (shape_name, batch_size, q_len, seq_low, seq_high)
    ("bs1_qlen1",         1, 1, 512,  512),
    ("bs4_qlen1_varlen",  4, 1, 256,  768),
    ("bs8_qlen4_spec",    8, 4, 256,  512),
]
NUM_HEADS = [16, 32]
DTYPES = [torch.bfloat16, torch.float16]


@parameterized.expand(
    [(shape[0], shape[1], shape[2], shape[3], shape[4], H, dt)
     for shape in SHAPES
     for H in NUM_HEADS
     for dt in DTYPES],
    name_func=lambda fn, n, p: (
        f"{fn.__name__}_{p.args[0]}_H{p.args[5]}_{str(p.args[6]).split('.')[-1]}"
    ),
)
def test_tokenspeed_mla_matches_flashinfer(
    shape_name: str,
    B: int,
    q_len: int,
    seq_low: int,
    seq_high: int,
    H: int,
    dtype: torch.dtype,
):
    """Parity vs flashinfer.mla.trtllm_batch_decode_with_kv_cache_mla."""
    _skip_if_unsupported()
    import flashinfer

    torch.manual_seed(0)
    device = torch.device("cuda:0")
    page_size = 64
    max_pages = max(8, (seq_high + page_size - 1) // page_size)
    num_blocks = max(B * max_pages + 16, 256)

    q       = _make_query(B, q_len, H, device, dtype)
    kv      = _make_kv_cache(
        num_blocks=num_blocks, page_size=page_size,
        device=device, dtype=dtype,
    )
    block_t = _make_block_tables(B, max_pages, num_blocks, device)
    s_lens  = _make_seq_lens(B, seq_low, seq_high, device)
    ws      = _make_workspace(B, H, q_len, device)

    bmm1_scale = 1.0 / math.sqrt(QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM)
    bmm2_scale = 1.0

    # --- baseline (FlashInfer) ---
    try:
        ref = flashinfer.mla.trtllm_batch_decode_with_kv_cache_mla(
            query=q.clone(), kv_cache=kv.clone(),
            workspace_buffer=ws.clone().view(-1, 4),
            qk_nope_head_dim=QK_NOPE_HEAD_DIM,
            kv_lora_rank=KV_LORA_RANK,
            qk_rope_head_dim=QK_ROPE_HEAD_DIM,
            block_tables=block_t, seq_lens=s_lens,
            max_seq_len=int(s_lens.max().item()),
            out=None, bmm1_scale=bmm1_scale, bmm2_scale=bmm2_scale,
            sinks=None,
        )
    except Exception as e:
        if "Missing TRTLLM-GEN kernel" in str(e):
            pytest.skip(
                f"flashinfer baseline cubin missing for "
                f"{shape_name}/H={H}/{dtype}: {e}")
        raise

    # --- variant (TokenSpeed) ---
    out = tokenspeed_batch_decode_with_kv_cache_mla(
        query=q.clone(), kv_cache=kv.clone(),
        workspace_buffer=ws.clone(),
        qk_nope_head_dim=QK_NOPE_HEAD_DIM,
        kv_lora_rank=KV_LORA_RANK,
        qk_rope_head_dim=QK_ROPE_HEAD_DIM,
        block_tables=block_t, seq_lens=s_lens,
        max_seq_len=int(s_lens.max().item()),
        out=None, bmm1_scale=bmm1_scale, bmm2_scale=bmm2_scale,
        sinks=None,
    )

    # Tolerances match spike step 5 (rtol=0.005, atol=0.05). bs8_qlen4_spec
    # was the only documented FAIL — keep the assertion strict so the
    # divergence is visible if it recurs on K2.6.
    torch.testing.assert_close(out, ref, rtol=0.005, atol=0.05)


def test_tokenspeed_mla_rejects_sinks():
    """Wrapper must NotImplementedError when sinks is non-None (spike step 5)."""
    _skip_if_unsupported()

    device = torch.device("cuda:0")
    q = _make_query(1, 1, 16, device, torch.bfloat16)
    kv = _make_kv_cache(
        num_blocks=64, page_size=64, device=device, dtype=torch.bfloat16,
    )
    ws = _make_workspace(1, 16, 1, device)

    with pytest.raises(NotImplementedError):
        tokenspeed_batch_decode_with_kv_cache_mla(
            query=q, kv_cache=kv, workspace_buffer=ws,
            qk_nope_head_dim=QK_NOPE_HEAD_DIM,
            kv_lora_rank=KV_LORA_RANK,
            qk_rope_head_dim=QK_ROPE_HEAD_DIM,
            block_tables=torch.zeros((1, 8), dtype=torch.int32, device=device),
            seq_lens=torch.tensor([128], dtype=torch.int32, device=device),
            max_seq_len=128,
            out=None,
            bmm1_scale=None, bmm2_scale=None,
            sinks=torch.zeros(16, device=device, dtype=torch.bfloat16),  # ← rejected
        )

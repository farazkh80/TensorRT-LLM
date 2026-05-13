# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Numerical parity check between TokenSpeed MLA decode and FlashInfer MLA decode.

Drives the TokenSpeed wrapper introduced in
``tensorrt_llm/_torch/attention_backend/tokenspeed_mla.py`` against
``flashinfer.mla.trtllm_batch_decode_with_kv_cache_mla`` (the kernel TRT-LLM
calls today inside ``run_mla_generation``) at DeepSeek-V2-Lite MLA dims.

Skipped unless running on data-center Blackwell (SM 10.0 / 10.3) with both
``tokenspeed_mla`` and ``flashinfer`` installed — there is no software fallback
for either kernel.
"""

import math

import pytest
import torch

from tensorrt_llm._torch.attention_backend.tokenspeed_mla import (
    is_tokenspeed_mla_available,
    tokenspeed_batch_decode_with_kv_cache_mla,
)
from tensorrt_llm._utils import get_sm_version, is_sm_100f

flashinfer = pytest.importorskip(
    "flashinfer",
    reason="flashinfer is required for the FlashInfer MLA baseline.",
)
pytest.importorskip(
    "tokenspeed_mla",
    reason="tokenspeed-mla is not installed in this environment.",
)


# DeepSeek-V2-Lite / V3-Lite MLA geometry. Same kv_lora_rank/qk_rope_head_dim as
# DSv3 and Kimi K2.5, so kernel selection is identical to the production targets.
# Tested at two num_heads values to cover DSv2-Lite (16) and DSv3-Lite (32);
# kv_lora_rank/qk_rope_head_dim/qk_nope_head_dim are identical across both.
DSV_LITE_KV_LORA_RANK = 512
DSV_LITE_QK_ROPE_HEAD_DIM = 64
DSV_LITE_QK_NOPE_HEAD_DIM = 128
DSV_LITE_V_HEAD_DIM = 128
DSV_LITE_NUM_HEADS_VALUES = (16, 32)  # DSv2-Lite, DSv3-Lite respectively

# FlashInfer's MLA decode reuses ``params.workspace`` (32 MB by default).
WORKSPACE_BYTES = 64 * 1024 * 1024


def _skip_if_unsupported() -> None:
    sm = get_sm_version()
    if not is_sm_100f(sm):
        pytest.skip(
            f"TokenSpeed MLA requires data-center Blackwell (SM 10.0 / 10.3); found SM {sm}."
        )
    if not is_tokenspeed_mla_available():
        pytest.skip(
            "tokenspeed-mla is not importable; install tokenspeed-mla to run this parity test."
        )


def _make_inputs(
    *,
    batch_size: int,
    q_len: int,
    num_heads: int,
    page_size: int,
    seq_lens_host: list[int],
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
):
    """Build synthetic MLA decode inputs in the layout both kernels expect.

    Returns a dict consumed by both ``flashinfer.mla.trtllm_batch_decode_with_
    kv_cache_mla`` and ``tokenspeed_batch_decode_with_kv_cache_mla`` — same
    tensors, no copies.
    """
    torch.manual_seed(seed)

    head_dim_qk = DSV_LITE_KV_LORA_RANK + DSV_LITE_QK_ROPE_HEAD_DIM
    max_seq_len = max(seq_lens_host)
    max_pages = (max_seq_len + page_size - 1) // page_size

    query = torch.randn(batch_size, q_len, num_heads, head_dim_qk, dtype=dtype, device=device)

    # Allocate one global page pool and a unique block table per batch entry.
    # Pages are not shared across batch entries — keeps the test deterministic.
    num_pages_total = batch_size * max_pages
    kv_cache = torch.randn(num_pages_total, page_size, head_dim_qk, dtype=dtype, device=device)
    block_tables = torch.arange(num_pages_total, dtype=torch.int32, device=device).view(
        batch_size, max_pages
    )

    seq_lens = torch.tensor(seq_lens_host, dtype=torch.int32, device=device)

    workspace_buffer = torch.zeros(WORKSPACE_BYTES, dtype=torch.int8, device=device)

    softmax_scale = 1.0 / math.sqrt(DSV_LITE_QK_NOPE_HEAD_DIM + DSV_LITE_QK_ROPE_HEAD_DIM)

    return dict(
        query=query,
        kv_cache=kv_cache,
        workspace_buffer=workspace_buffer,
        qk_nope_head_dim=DSV_LITE_QK_NOPE_HEAD_DIM,
        kv_lora_rank=DSV_LITE_KV_LORA_RANK,
        qk_rope_head_dim=DSV_LITE_QK_ROPE_HEAD_DIM,
        block_tables=block_tables,
        seq_lens=seq_lens,
        max_seq_len=max_seq_len,
        bmm1_scale=softmax_scale,
        bmm2_scale=1.0,
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "num_heads",
    DSV_LITE_NUM_HEADS_VALUES,
    ids=[f"H{h}" for h in DSV_LITE_NUM_HEADS_VALUES],
)
@pytest.mark.parametrize(
    "batch_size,q_len,page_size,seq_lens_host",
    [
        # Pure decode (BS=1) — smallest config that exercises split-KV.
        (1, 1, 64, [256]),
        # BS=4 mixed lengths — exercises variable-seq kernel path.
        (4, 1, 64, [128, 256, 384, 512]),
        # BS=8, q_len=4 — MTP-style spec decode (matches the
        # "halves decode latency" claim from the TokenSpeed blog).
        (8, 4, 64, [256, 384, 512, 256, 384, 512, 256, 384]),
    ],
    ids=["bs1_qlen1", "bs4_qlen1_varlen", "bs8_qlen4_spec"],
)
def test_tokenspeed_mla_decode_parity(
    dtype: torch.dtype,
    num_heads: int,
    batch_size: int,
    q_len: int,
    page_size: int,
    seq_lens_host: list[int],
) -> None:
    """TokenSpeed MLA output matches FlashInfer MLA on DSv2/v3-Lite-sized inputs."""
    _skip_if_unsupported()

    device = torch.device("cuda")
    inputs = _make_inputs(
        batch_size=batch_size,
        q_len=q_len,
        num_heads=num_heads,
        page_size=page_size,
        seq_lens_host=seq_lens_host,
        dtype=dtype,
        device=device,
        seed=1234,
    )

    # Run TokenSpeed first so we capture its behaviour even when the FlashInfer
    # baseline has no kernel for this shape on the current GPU. On B300 (sm_103)
    # FlashInfer 0.6.x ships a sparse trtllm-gen cubin set that omits some
    # DSv3-Lite MLA shape × multi-CTA combinations; we surface that gap as a
    # test skip rather than a hard failure.
    tokenspeed_out = tokenspeed_batch_decode_with_kv_cache_mla(
        query=inputs["query"],
        kv_cache=inputs["kv_cache"],
        workspace_buffer=inputs["workspace_buffer"],
        qk_nope_head_dim=inputs["qk_nope_head_dim"],
        kv_lora_rank=inputs["kv_lora_rank"],
        qk_rope_head_dim=inputs["qk_rope_head_dim"],
        block_tables=inputs["block_tables"],
        seq_lens=inputs["seq_lens"],
        max_seq_len=inputs["max_seq_len"],
        bmm1_scale=inputs["bmm1_scale"],
        bmm2_scale=inputs["bmm2_scale"],
        sinks=None,
    )

    # Basic sanity on TokenSpeed output regardless of FlashInfer availability.
    expected_shape = (batch_size, q_len, num_heads, inputs["kv_lora_rank"])
    assert tuple(tokenspeed_out.shape) == expected_shape, (
        f"tokenspeed shape {tuple(tokenspeed_out.shape)} != expected {expected_shape}"
    )
    assert torch.isfinite(tokenspeed_out).all(), "tokenspeed output has NaN/Inf"

    # FlashInfer baseline — may fail on B300 if the trtllm-gen cubin set is
    # incomplete for this shape. Skip parity in that case so the test still
    # exercises TokenSpeed end-to-end.
    try:
        flashinfer_out = flashinfer.mla.trtllm_batch_decode_with_kv_cache_mla(
            query=inputs["query"],
            kv_cache=inputs["kv_cache"],
            # FlashInfer expects a 2D int32 view (one row per 16 bytes); the
            # same storage is reinterpreted as int8 inside the TokenSpeed
            # wrapper.
            workspace_buffer=inputs["workspace_buffer"].view(torch.int32).view(-1, 4),
            qk_nope_head_dim=inputs["qk_nope_head_dim"],
            kv_lora_rank=inputs["kv_lora_rank"],
            qk_rope_head_dim=inputs["qk_rope_head_dim"],
            block_tables=inputs["block_tables"],
            seq_lens=inputs["seq_lens"],
            max_seq_len=inputs["max_seq_len"],
            bmm1_scale=inputs["bmm1_scale"],
            bmm2_scale=inputs["bmm2_scale"],
            sinks=None,
        )
    except RuntimeError as exc:
        if "Missing TRTLLM-GEN kernel" in str(exc):
            pytest.skip(
                f"FlashInfer trtllm-gen has no kernel for this shape on "
                f"sm_{torch.cuda.get_device_capability(device)[0]}"
                f"{torch.cuda.get_device_capability(device)[1]}; "
                f"TokenSpeed ran fine. Reason: {exc}"
            )
        raise

    # Both kernels succeeded — do the parity diff.
    assert flashinfer_out.shape == tokenspeed_out.shape, (
        f"shape mismatch: flashinfer={flashinfer_out.shape} tokenspeed={tokenspeed_out.shape}"
    )

    # Tolerances are conservative for a spike — both kernels use online
    # softmax with different reduction orders, so small bit-level drift is
    # expected. Tighten once we have real-model parity numbers.
    atol, rtol = (5e-2, 5e-3) if dtype == torch.bfloat16 else (1e-2, 1e-3)
    torch.testing.assert_close(
        tokenspeed_out.to(torch.float32),
        flashinfer_out.to(torch.float32),
        atol=atol,
        rtol=rtol,
    )


def test_tokenspeed_mla_rejects_sinks() -> None:
    """The wrapper must reject ``sinks`` rather than silently ignoring them."""
    _skip_if_unsupported()
    device = torch.device("cuda")
    inputs = _make_inputs(
        batch_size=1,
        q_len=1,
        num_heads=16,
        page_size=64,
        seq_lens_host=[64],
        dtype=torch.bfloat16,
        device=device,
        seed=0,
    )
    with pytest.raises(NotImplementedError, match="sinks"):
        tokenspeed_batch_decode_with_kv_cache_mla(
            query=inputs["query"],
            kv_cache=inputs["kv_cache"],
            workspace_buffer=inputs["workspace_buffer"],
            qk_nope_head_dim=inputs["qk_nope_head_dim"],
            kv_lora_rank=inputs["kv_lora_rank"],
            qk_rope_head_dim=inputs["qk_rope_head_dim"],
            block_tables=inputs["block_tables"],
            seq_lens=inputs["seq_lens"],
            max_seq_len=inputs["max_seq_len"],
            sinks=torch.zeros(1, device=device),
        )

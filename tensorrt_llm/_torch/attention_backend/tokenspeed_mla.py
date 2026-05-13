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
"""TokenSpeed MLA kernel drop-in for TRT-LLM (spike, B200/B300 only).

Wraps LightSeek Foundation's tokenspeed-mla CuTe DSL kernels so they can
substitute for ``flashinfer.mla.trtllm_batch_decode_with_kv_cache_mla`` at the
existing call site (``trtllm_gen.run_mla_generation``) without disrupting the
TrtllmAttention pipeline.

Status: spike. Numerical parity is validated by
``tests/unittest/_torch/attention/test_tokenspeed_mla.py`` (skipped on non
SM 10.0 / 10.3 hardware or when ``tokenspeed_mla`` is not installed). Wiring
into ``run_mla_generation`` behind ``TLLM_TOKENSPEED_MLA=1`` is a separate
follow-up once parity is confirmed on real B200/B300 hardware.

Hard arch gate: tokenspeed-mla's CuTe DSL kernels target ``tcgen05`` /
Blackwell TMEM, which is only present on data-center Blackwell (SM 10.0 / 10.3,
i.e. B200 / B300). Consumer/workstation Blackwell SM 12.0 (RTX 50 series,
RTX PRO 6000) is **not** supported. The arch check inside tokenspeed-mla
itself (``mla_decode_fp{8,16}.py``, ``mla_prefill.py``) is the authoritative
guard; this module fails fast with a friendly error before reaching it.
"""

from functools import lru_cache
from typing import Optional

import torch

from tensorrt_llm._utils import get_sm_version, is_sm_100f
from tensorrt_llm.logger import logger

# Supported SM versions for tokenspeed-mla. Mirrors the
# ``Arch.sm_100..sm_100f`` / ``Arch.sm_103..sm_103f`` gates inside
# tokenspeed_mla itself.
_SUPPORTED_SM_VERSIONS = (100, 103)


@lru_cache(maxsize=1)
def is_tokenspeed_mla_available() -> bool:
    """Whether the tokenspeed-mla decode kernel can run on this process.

    Performs two checks (cached): (1) the local GPU is data-center Blackwell
    (SM 10.0 / 10.3), (2) the ``tokenspeed_mla`` package imports cleanly.
    Both must hold; otherwise callers should fall back to FlashInfer MLA.
    """
    sm = get_sm_version()
    if not is_sm_100f(sm):
        logger.debug(
            "TokenSpeed MLA unavailable: SM %d is not data-center Blackwell (supported: %s).",
            sm,
            _SUPPORTED_SM_VERSIONS,
        )
        return False
    try:
        import tokenspeed_mla  # noqa: F401
    except ImportError as exc:
        logger.debug("TokenSpeed MLA unavailable: %s", exc)
        return False
    return True


def _ensure_int8_workspace(workspace_buffer: torch.Tensor) -> torch.Tensor:
    """Return a contiguous int8 1D view of ``workspace_buffer``.

    TRT-LLM's FlashInfer call site passes ``params.workspace.view(-1, 4)``
    (typically uint8/int32 backing); tokenspeed-mla asserts ``int8`` and a
    1D shape. This converts via ``view`` without copying — the underlying
    storage is the same bytes.
    """
    flat = workspace_buffer.reshape(-1)
    if flat.dtype == torch.int8:
        return flat
    # ``view`` reinterprets dtype without copying. Requires contiguous storage.
    assert flat.is_contiguous(), "workspace_buffer must be contiguous"
    return flat.view(torch.int8)


def tokenspeed_batch_decode_with_kv_cache_mla(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    workspace_buffer: torch.Tensor,
    qk_nope_head_dim: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    max_seq_len: int,
    out: Optional[torch.Tensor] = None,
    bmm1_scale: float = 1.0,
    bmm2_scale: float = 1.0,
    sinks: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Drop-in for ``flashinfer.mla.trtllm_batch_decode_with_kv_cache_mla``.

    Mirrors the FlashInfer signature exactly so callers can swap kernels with
    a one-line change. Arguments not used by tokenspeed-mla (``qk_nope_head_dim``
    is implicit in the query layout, ``sinks`` is unsupported) are accepted
    for API compatibility and either ignored or rejected with a clear error.

    Shape contract (identical to FlashInfer):
        query:        ``[B, q_len, H, kv_lora_rank + qk_rope_head_dim]``
        kv_cache:     ``[num_pages, page_size, kv_lora_rank + qk_rope_head_dim]``
                      or ``[num_pages, 1, page_size, kv_lora_rank + qk_rope_head_dim]``
        block_tables: ``[B, max_pages]``
        seq_lens:     ``[B]``
        out:          ``[B, q_len, H, kv_lora_rank]`` if provided
    """
    # ``qk_nope_head_dim`` is part of the FlashInfer API but tokenspeed-mla
    # infers the layout from ``kv_lora_rank + qk_rope_head_dim`` directly.
    # The argument is preserved for signature parity; reject obvious mismatches.
    del qk_nope_head_dim

    if sinks is not None:
        raise NotImplementedError("TokenSpeed MLA decode does not support attention sinks.")

    # Lazy import keeps non-Blackwell paths (CI, unit tests on H100/SM90 etc.)
    # free of the tokenspeed_mla dependency.
    from tokenspeed_mla import tokenspeed_mla_decode

    ws = _ensure_int8_workspace(workspace_buffer)

    return tokenspeed_mla_decode(
        query=query,
        kv_cache=kv_cache,
        workspace_buffer=ws,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        block_tables=block_tables,
        seq_lens=seq_lens,
        max_seq_len=max_seq_len,
        softmax_scale=bmm1_scale,
        output_scale=bmm2_scale,
        out=out,
    )

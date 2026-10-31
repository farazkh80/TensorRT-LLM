# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""kimi_k3_cute_gemm (TLLM_K3_CUTE_GEMM) vs cuBLAS at the Kimi K3 attention-core shapes, against an fp32 reference."""

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch import cute_dsl_utils
from tensorrt_llm._torch.modules import kimi_k3_cute_gemm as k3_gemm
from tensorrt_llm._utils import get_sm_version

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not cute_dsl_utils.IS_CUTLASS_DSL_RUBIN_AVAILABLE
    or get_sm_version() != 107,
    reason="the Kimi K3 CuTe DSL GEMM dispatch targets SM107",
)

# (N, K) of every attention-core Linear of one DEP rank.
GEMMS = {
    "kda.qkvg": (49152, 7168),
    "kda.bfa": (224, 7168),
    "kda.g": (12288, 7168),
    "kda.o": (7168, 12288),
    "mla.qkva": (2112, 7168),
    "mla.qb": (18432, 1536),
}
M_VALUES = [1, 7, 33, 64, 8192]


@pytest.fixture(autouse=True)
def _no_tf32():
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    yield
    torch.backends.cuda.matmul.allow_tf32 = prev


def _rel(a: torch.Tensor, ref: torch.Tensor) -> float:
    return ((a.float() - ref).norm() / ref.norm()).item()


def _assert_no_worse_than_cublas(
    out: torch.Tensor, cublas: torch.Tensor, ref: torch.Tensor
) -> None:
    assert bool(torch.isfinite(out).all()) and out.abs().max().item() > 0
    err, err_cublas = _rel(out, ref), _rel(cublas, ref)
    assert err <= 1.05 * err_cublas + 1e-5, f"rel-L2 CuTe {err:.3e} cuBLAS {err_cublas:.3e}"


@pytest.mark.parametrize("m", M_VALUES)
@pytest.mark.parametrize("name", list(GEMMS))
def test_linear(name: str, m: int) -> None:
    n, k = GEMMS[name]
    torch.manual_seed(m)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.05
    _assert_no_worse_than_cublas(k3_gemm.linear(x, w), F.linear(x, w), x.float() @ w.float().t())


@pytest.mark.parametrize("m", M_VALUES)
def test_linear_strided_a(m: int) -> None:
    # KDA f_b reads the [:, :128] column slice of the [M, 224] [f_a | b] output.
    torch.manual_seed(m)
    x = torch.randn(m, 224, device="cuda", dtype=torch.bfloat16)[:, :128]
    w = torch.randn(12288, 128, device="cuda", dtype=torch.bfloat16) * 0.05
    _assert_no_worse_than_cublas(
        k3_gemm.linear(x, w), torch.matmul(x, w.t()), x.float() @ w.float().t()
    )


@pytest.mark.parametrize("t", [65, 8192])
@pytest.mark.parametrize("row", [192, 256, 128], ids=["k_nope", "v_kv", "v_contiguous"])
def test_bmm_broadcast_a(row: int, t: int) -> None:
    # MLA prefill kv_b: the latent broadcast over 96 heads, written into one head-strided view of a [t, 96, row] buffer.
    torch.manual_seed(t)
    a = torch.randn(t, 512, device="cuda", dtype=torch.bfloat16).expand(96, t, 512)
    w = torch.randn(96, 128, 512, device="cuda", dtype=torch.bfloat16) * 0.05
    out = torch.zeros(t, 96, row, device="cuda", dtype=torch.bfloat16)[:, :, :128].transpose(0, 1)
    k3_gemm.bmm(a, w, out)
    ref = torch.bmm(a.float(), w.float().transpose(1, 2))
    _assert_no_worse_than_cublas(out, torch.bmm(a, w.transpose(1, 2)), ref)


@pytest.mark.parametrize("m", [1, 16, 64])
@pytest.mark.parametrize("n, k", [(512, 128), (128, 512)], ids=["uk", "uv"])
def test_bmm_absorbed(n: int, k: int, m: int) -> None:
    torch.manual_seed(m)
    x = torch.randn(96, m, k, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(96, n, k, device="cuda", dtype=torch.bfloat16) * 0.05
    ref = torch.bmm(x.float(), w.float().transpose(1, 2))
    _assert_no_worse_than_cublas(k3_gemm.bmm(x, w), torch.bmm(x, w.transpose(1, 2)), ref)


def test_empty_and_non_bf16() -> None:
    w = torch.randn(224, 7168, device="cuda", dtype=torch.bfloat16)
    assert k3_gemm.linear(torch.empty(0, 7168, device="cuda", dtype=torch.bfloat16), w).shape == (
        0,
        224,
    )
    x32, w32 = torch.randn(4, 64, device="cuda"), torch.randn(32, 64, device="cuda")
    torch.testing.assert_close(k3_gemm.linear(x32, w32), x32 @ w32.t())

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

import os
from typing import Optional, Tuple, Union

import torch

from tensorrt_llm._utils import get_sm_version, nvtx_range
from tensorrt_llm.logger import logger
from tensorrt_llm.models.modeling_utils import QuantAlgo

from ...utils import ActivationType, Fp4QuantizedTensor
from .fused_moe_cutlass import CutlassFusedMoE
from .interface import _warn_and_return

# Module-level shared workspace pool. lukealonso/b12x's
# ``b12x_moe_fp4(workspace=pool)`` accepts a stateful ``TPMoEWorkspacePool``
# that grows on demand; sharing one pool across all MoE layers in a model
# (which run sequentially on a single CUDA stream) saves the per-layer arena
# duplication cost without correctness risk — the call returns its routed
# output before the next layer dispatches, so workspace is consumed serially.
_SHARED_WORKSPACE_POOL = None

# Module-level shared output buffer pool, keyed by
# ``(max_num_tokens, hidden_size, dtype, device)``. ``b12x.integration.b12x_moe_fp4``
# raises ``ValueError("CUDA graph capture requires a caller-owned output
# buffer")`` if ``output=None`` during capture (upstream tp_moe.py L3334),
# so we allocate one ``(max_num_tokens, hidden_size)`` buffer per
# (shape, dtype, device) tuple and share it across all MoE layers. Layers
# run sequentially on a single CUDA stream, so reuse is correctness-safe;
# this also saves ``(num_moe_layers - 1) * max_num_tokens * hidden_size *
# sizeof(dtype)`` bytes of GPU memory vs. one buffer per layer (~156 MB on
# Nemotron-Super-120B with hidden=1024, max_num_tokens=2048, bf16, 40 MoE
# layers — same shape pattern as the existing FlashInferFusedMoE shared buf).
_SHARED_MOE_OUTPUT_BUF: dict = {}


# ActivationType -> b12x activation string. lukealonso b12x exposes "relu2"
# (Nemotron-style x = relu(x)^2) and "silu" (SwiGLU-style x * silu(gate)).
_ACTIVATION_MAP = {
    ActivationType.Relu2: "relu2",
    ActivationType.Swiglu: "silu",
}


class B12xLukeFusedMoE(CutlassFusedMoE):
    """NVFP4 fused-MoE backend wrapping lukealonso's standalone ``b12x``
    package (https://github.com/lukealonso/b12x), the upstream CuTe DSL
    SM120/SM121 kernel that flashinfer's ``B12xMoEWrapper`` was vendored
    from.

    This is a sibling of :class:`FlashInferFusedMoE`; both subclass
    :class:`CutlassFusedMoE` and override only the per-expert compute step.
    The two backends consume the *same* NVFP4 ModelOpt weight tensors but
    differ in the scale-factor convention they expect:

    +------------------+-------------------------------+----------------------------+
    |                  | flashinfer ``B12xMoEWrapper`` | lukealonso ``b12x`` 0.13.0 |
    +==================+===============================+============================+
    | block scale form | UN-normalized FP8 SF          | NORMALIZED FP8 SF (HF/MO)  |
    | ``w1_alpha(s)``  | ``1/input_scale`` (dual-use)  | ``input_scale * w_scale_2``|
    | ``a1_gscale``    | (folded into ``w1_alpha``)    | ``1/input_scale`` (per E)  |
    | swizzle helper   | ``convert_sf_to_mma_layout``  | ``swizzle_block_scale``    |
    +------------------+-------------------------------+----------------------------+

    The NORMALIZED-SF convention matches what the inherited
    :class:`CutlassFusedMoE` NVFP4 ``post_load_weights`` produces, so the
    weight conversion here is shorter than the FlashInfer one — no
    un-normalization step needed.

    Selected via ``moe_backend=B12X_LUKE``. Same SM gating as FlashInfer
    (SM120/SM121), same activation gating (relu2 / silu), same hybrid
    CUTLASS-prefill / b12x-decode dispatch via
    ``TRTLLM_FLASHINFER_PREFILL_VIA_CUTLASS_THRESHOLD``.
    """

    # SM versions on which lukealonso/b12x supports kernels. SM120 = desktop
    # Blackwell (RTX PRO 6000 / GB202); SM121 = GB10 / DGX Spark. The
    # upstream README is explicit: "It does not intend to target any other
    # GPU architectures, including SM100."
    _SUPPORTED_SM_VERSIONS = frozenset({120, 121})

    @classmethod
    def can_implement(
        cls,
        quant_algo: Optional[QuantAlgo],
        dtype_activation: torch.dtype = torch.bfloat16,
        swiglu_gptoss_style: bool = False,
    ) -> Tuple[bool, Optional[str]]:
        sm_version = get_sm_version()
        if sm_version not in cls._SUPPORTED_SM_VERSIONS:
            sm_list = "/".join(f"SM{v}" for v in sorted(cls._SUPPORTED_SM_VERSIONS))
            return _warn_and_return(f"B12xLukeFusedMoE requires {sm_list}, got SM{sm_version}")
        if quant_algo != QuantAlgo.NVFP4:
            return _warn_and_return(
                f"B12xLukeFusedMoE only supports NVFP4 quantization (got quant_algo={quant_algo})"
            )
        if dtype_activation not in {torch.float16, torch.bfloat16}:
            return _warn_and_return(
                f"B12xLukeFusedMoE NVFP4 requires float16 or bfloat16 "
                f"activation dtype (got {dtype_activation})"
            )
        if swiglu_gptoss_style:
            return _warn_and_return("B12xLukeFusedMoE does not support swiglu_gptoss_style")
        return True, None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.ep_size != 1:
            raise ValueError(
                f"B12xLukeFusedMoE requires ep_size == 1 "
                f"(got ep_size={self.ep_size}); use --moe_backend CUTLASS for EP."
            )
        if self.enable_alltoall:
            raise ValueError("B12xLukeFusedMoE does not support MoE alltoall communication.")
        if self.activation_type not in _ACTIVATION_MAP:
            supported = ", ".join(a.name for a in _ACTIVATION_MAP)
            raise ValueError(
                f"B12xLukeFusedMoE does not support activation "
                f"{ActivationType(self.activation_type).name}; "
                f"supported: {supported}."
            )

        self._b12x_experts = None
        self._b12x_activation = _ACTIVATION_MAP[self.activation_type]
        self._b12x_workspace_pool = None
        # Pre-allocated output buffer of shape ``(max_num_tokens, hidden_size)``;
        # populated lazily in ``post_load_weights`` once the inherited init
        # has resolved ``self.dtype`` and the device. ``run_moe`` slices to
        # ``[:m]`` per call to satisfy b12x's exact-shape ``output`` contract.
        self._b12x_output_buf = None

    @property
    def _prefill_via_cutlass_threshold(self) -> int:
        """Hybrid CUTLASS-prefill / b12x-decode dispatch threshold.

        Reuses the same env var as :class:`FlashInferFusedMoE` so the
        existing `bench_kvoff_*.yml` files / scripts work unchanged when
        swapping ``moe_config.backend`` from FLASHINFER to B12X_LUKE.

        ``x.shape[0] >= threshold`` => CUTLASS GroupGEMM (better at large m).
        ``x.shape[0] <  threshold`` => b12x (better at m=1 decode).
        ``0`` (default) disables hybrid mode and keeps pure-b12x behavior.
        """
        return int(os.environ.get("TRTLLM_FLASHINFER_PREFILL_VIA_CUTLASS_THRESHOLD", "0"))

    def _route_to_cutlass(self, x) -> bool:
        """Return True iff this call should fall back to the inherited
        CUTLASS path. ``Fp4QuantizedTensor`` inputs always stay on the b12x
        path (which rejects them) so the existing error message is
        preserved."""
        return (
            self._prefill_via_cutlass_threshold > 0
            and isinstance(x, torch.Tensor)
            and x.shape[0] >= self._prefill_via_cutlass_threshold
        )

    def post_load_weights(self):
        """Convert NVFP4 weights to lukealonso/b12x layout and cache them
        in a :class:`B12XFP4ExpertWeights` for per-call dispatch.

        Called by ``model_loader`` after ``load_weights`` finishes. The
        inherited NVFP4 ``process_weights_after_loading`` has already run,
        populating ``w3_w1_weight``, ``w2_weight``, ``*_weight_scale``,
        ``fc{31,2}_alpha`` and ``fc{31,2}_input_scale``.
        """
        super().post_load_weights()

        try:
            from b12x.cute.fp4 import swizzle_block_scale
            from b12x.integration import B12XFP4ExpertWeights
            from b12x.integration.tp_moe import TPMoEWorkspacePool
        except ImportError as e:
            raise RuntimeError(
                "B12xLukeFusedMoE requires the `b12x` package "
                "(b12x.integration.B12XFP4ExpertWeights, b12x.cute.fp4."
                "swizzle_block_scale). Install with "
                "`pip install git+https://github.com/lukealonso/b12x`. "
                f"Original import error: {e}"
            ) from e

        num_experts = self.num_experts

        # Reciprocal input quant global scales. ``fc31_input_scale`` /
        # ``fc2_input_scale`` are scalars in TRT-LLM's NVFP4 quant method
        # (see quantization.py:2012/4051). b12x's contract is "[E] OR
        # scalar", and the Nemotron-tuned ``_launch_exact_relu2_bs1_nemotron``
        # fast path explicitly requires ``a1_gscale.numel() == 1`` (see
        # tp_moe.py L2735). Keep the scalar form so we can hit that path
        # when it's enabled (in upstream master `1378cea7` it's gated off
        # via an unconditional ``return False`` — see B12X_LUKE_RESULTS.md
        # for details — but keeping scalar form costs nothing).
        a1_gscale = (1.0 / self.fc31_input_scale).to(torch.float32).contiguous()
        a2_gscale = (1.0 / self.fc2_input_scale).to(torch.float32).contiguous()

        # b12x expects HF/ModelOpt-NORMALIZED FP8 block scales (no
        # un-normalization needed, unlike flashinfer's b12x). The inherited
        # NVFP4 quant method already stored them in normalized form, so we
        # just view-as-FP8 and swizzle to MMA layout.
        w1_sf_fp8 = self.w3_w1_weight_scale.view(torch.float8_e4m3fn).contiguous()
        w2_sf_fp8 = self.w2_weight_scale.view(torch.float8_e4m3fn).contiguous()
        w1_blockscale = swizzle_block_scale(w1_sf_fp8)
        w2_blockscale = swizzle_block_scale(w2_sf_fp8)

        # Recover per-expert ``weight_scale_2`` from
        # ``fc31_alpha = (1/fc31_input_scale) * weight_scale_2`` (the
        # CutlassFusedMoE NVFP4 dual-use convention; see
        # fused_moe_flashinfer.py L191 for the same reverse-engineering).
        # Then ``w1_alphas = input_scale * weight_scale_2 = fc31_input_scale
        # * fc31_alpha * fc31_input_scale = fc31_input_scale^2 * fc31_alpha``.
        fc31_input_scale_f32 = self.fc31_input_scale.to(torch.float32)
        fc2_input_scale_f32 = self.fc2_input_scale.to(torch.float32)
        w1_alphas = (
            (fc31_input_scale_f32 * fc31_input_scale_f32 * self.fc31_alpha)
            .to(torch.float32)
            .contiguous()
        )
        w2_alphas = (
            (fc2_input_scale_f32 * fc2_input_scale_f32 * self.fc2_alpha)
            .to(torch.float32)
            .contiguous()
        )

        # FP4 weights: TRT-LLM packs 16 FP4 values per int64. b12x expects
        # uint8 storage with stride[-1] == 1 byte for its internal
        # ``view(torch.float4_e2m1fn_x2)`` reinterpret. ``view(torch.uint8)``
        # provides that without copying.
        w1_fp4 = self.w3_w1_weight.view(torch.uint8)
        w2_fp4 = self.w2_weight.view(torch.uint8)

        self._b12x_experts = B12XFP4ExpertWeights(
            a1_gscale=a1_gscale,
            w1_fp4=w1_fp4,
            w1_blockscale=w1_blockscale,
            w1_alphas=w1_alphas,
            a2_gscale=a2_gscale,
            w2_fp4=w2_fp4,
            w2_blockscale=w2_blockscale,
            w2_alphas=w2_alphas,
        )

        # Lazy-init a single shared workspace pool on first layer; subsequent
        # layers reuse it. ``TPMoEWorkspacePool`` is stateful and grows on
        # demand, so the same pool serves any (m, num_topk) shape we throw at
        # it.
        global _SHARED_WORKSPACE_POOL
        if _SHARED_WORKSPACE_POOL is None:
            _SHARED_WORKSPACE_POOL = TPMoEWorkspacePool()
        self._b12x_workspace_pool = _SHARED_WORKSPACE_POOL

        # Pre-allocate a shared ``(max_num_tokens, hidden_size)`` output buffer
        # to satisfy b12x's "caller-owned output during CUDA graph capture"
        # contract. ``self.dtype`` is the activation dtype set by the
        # inherited :class:`MoE` init; ``self.moe_max_num_tokens`` is the
        # padded upper bound on tokens routed through this MoE in any one
        # forward call.
        device = self.w3_w1_weight.device
        max_m = int(self.moe_max_num_tokens)
        hidden = int(self.hidden_size)
        buf_key = (max_m, hidden, self.dtype, str(device))
        global _SHARED_MOE_OUTPUT_BUF
        shared = _SHARED_MOE_OUTPUT_BUF.get(buf_key)
        if shared is None:
            shared = torch.empty(
                (max_m, hidden), dtype=self.dtype, device=device,
            ).contiguous()
            _SHARED_MOE_OUTPUT_BUF[buf_key] = shared
        self._b12x_output_buf = shared

        logger.info_once(
            f"B12xLukeFusedMoE active: hidden={self.hidden_size}, "
            f"intermediate={self.intermediate_size_per_partition}, "
            f"experts={self.num_experts}, top_k="
            f"{self.routing_method.experts_per_token}, "
            f"activation={self._b12x_activation}.",
            key="b12x_luke_moe_active",
        )

    @nvtx_range("[b12x_luke] quantize_input")
    def quantize_input(
        self,
        x: Union[torch.Tensor, Fp4QuantizedTensor],
        post_quant_comm: bool = True,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Passthrough: ``b12x_moe_fp4`` quantizes activations internally.

        ``CutlassFusedMoE.quantize_input`` would NVFP4-quantize ``x`` here
        and pass ``(x_quantized, x_sf)`` into the kernel; b12x instead
        consumes a bf16 / fp16 ``x`` and produces its own scale factors,
        so we forward the activation unchanged and emit no SF.

        With ``TRTLLM_FLASHINFER_PREFILL_VIA_CUTLASS_THRESHOLD > 0`` set,
        the inherited NVFP4 quant path is used for chunks at or above the
        threshold (prefill); ``run_moe`` performs the matching dispatch.
        """
        if self._route_to_cutlass(x):
            return CutlassFusedMoE.quantize_input(
                self, x, post_quant_comm=post_quant_comm, **kwargs
            )
        if isinstance(x, Fp4QuantizedTensor):
            raise ValueError(
                "B12xLukeFusedMoE does not accept Fp4QuantizedTensor input; "
                "b12x performs its own input quantization."
            )
        return x, None

    @nvtx_range("[b12x_luke] run_moe")
    def run_moe(
        self,
        x: torch.Tensor,
        token_selected_experts: torch.Tensor,
        token_final_scales: torch.Tensor,
        x_sf: Optional[torch.Tensor] = None,
        is_sf_swizzled: bool = True,
        output_dtype: Optional[torch.dtype] = None,
        tuner_num_tokens: Optional[int] = None,
        tuner_top_k: Optional[int] = None,
        moe_output: Optional[torch.Tensor] = None,
        enable_alltoall: Optional[bool] = None,
    ) -> torch.Tensor:
        if self._route_to_cutlass(x):
            return CutlassFusedMoE.run_moe(
                self,
                x,
                token_selected_experts=token_selected_experts,
                token_final_scales=token_final_scales,
                x_sf=x_sf,
                is_sf_swizzled=is_sf_swizzled,
                output_dtype=output_dtype,
                tuner_num_tokens=tuner_num_tokens,
                tuner_top_k=tuner_top_k,
                moe_output=moe_output,
                enable_alltoall=enable_alltoall,
            )
        if self._b12x_experts is None or self._b12x_workspace_pool is None:
            raise RuntimeError(
                "B12xLukeFusedMoE.run_moe called before post_load_weights completed."
            )
        if x_sf is not None:
            raise ValueError(
                "B12xLukeFusedMoE expects unquantized input (x_sf=None); "
                "got a precomputed scale factor."
            )

        # ``b12x.integration.b12x_moe_fp4`` is the lower-level entry-point:
        # it takes precomputed top-k arrays and runs router-output =>
        # routed-experts MoE, dispatching to the static / dynamic /
        # micro / "exact relu2 bs1 nemotron" kernel based on shape +
        # activation. For our hybrid mode at decode (m == 1, top_k = 22,
        # activation = "relu2") this routes through the optimized
        # _launch_exact_relu2_bs1_nemotron path that the recent commit
        # "Restore Nemotron micro MoE performance" was specifically about.
        #
        # b12x's topk arg names vs. TRT-LLM's:
        #   topk_weights <-- token_final_scales       [m, top_k] float
        #   topk_ids     <-- token_selected_experts   [m, top_k] int
        from b12x.integration import b12x_moe_fp4

        # b12x.integration.b12x_moe_fp4 requires output of EXACT shape
        # ``(m, k)`` matching ``a.shape``. ``moe_output`` from
        # CutlassFusedMoE.forward_chunk is None for non-alltoall paths,
        # so we slice the shared ``(max_num_tokens, hidden_size)`` buffer
        # to ``[:m]``. A dim-0 slice of a contiguous tensor is itself
        # contiguous (b12x asserts is_contiguous() at L3343).
        m = int(x.shape[0])
        if moe_output is not None:
            output_buf = moe_output
        else:
            output_buf = self._b12x_output_buf[:m]

        with nvtx_range("[b12x_luke] b12x_moe_fp4"):
            out = b12x_moe_fp4(
                x,
                self._b12x_experts.a1_gscale,
                self._b12x_experts.w1_fp4,
                self._b12x_experts.w1_blockscale,
                self._b12x_experts.w1_alphas,
                self._b12x_experts.a2_gscale,
                self._b12x_experts.w2_fp4,
                self._b12x_experts.w2_blockscale,
                self._b12x_experts.w2_alphas,
                token_final_scales,
                token_selected_experts,
                workspace=self._b12x_workspace_pool,
                output=output_buf,
                activation=self._b12x_activation,
                # The default "input_scales_are_reciprocal" contract in b12x
                # 0.13.0 raises if not set explicitly. We pass reciprocals
                # (1/input_scale) for both ``a1_gscale`` and ``a2_gscale``
                # above, so this is True.
                input_scales_are_reciprocal=True,
                # Both a*_gscale are weight-side constants in our case
                # (loaded once at post_load_weights, never re-derived per
                # call), so flag them as static to skip per-launch
                # re-expansion in the kernel.
                input_scales_static=True,
            )

        # ``b12x_moe_fp4`` writes into the caller-provided ``output_buf``
        # and returns the same tensor. Return that — it's the slice of our
        # shared buf, which is fine because the next MoE layer slices the
        # same shared buf with a (possibly different but always serial) m.
        return out

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
"""Negative-path tests for the FlashInferFusedMoE backend.

These checks run without a GPU: they verify the can_implement() gating
matrix and the hard-error policy in create_moe.get_moe_cls. Functional
correctness of the b12x kernel is covered by end-to-end model tests on
SM120/SM121 hardware.
"""

from unittest.mock import patch

import pytest
import torch

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.modules.fused_moe.create_moe import get_moe_cls
from tensorrt_llm._torch.modules.fused_moe.fused_moe_flashinfer import FlashInferFusedMoE
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig


@pytest.mark.parametrize("sm_version", [80, 89, 90, 100, 103])
def test_can_implement_rejects_unsupported_sm(sm_version):
    """can_implement returns False on every SM outside the supported set."""
    with patch(
        "tensorrt_llm._torch.modules.fused_moe.fused_moe_flashinfer.get_sm_version",
        return_value=sm_version,
    ):
        ok, reason = FlashInferFusedMoE.can_implement(QuantAlgo.NVFP4)
    assert not ok
    assert reason is not None and f"SM{sm_version}" in reason


@pytest.mark.parametrize("sm_version", sorted(FlashInferFusedMoE._SUPPORTED_SM_VERSIONS))
def test_can_implement_accepts_supported_sm_with_nvfp4(sm_version):
    with patch(
        "tensorrt_llm._torch.modules.fused_moe.fused_moe_flashinfer.get_sm_version",
        return_value=sm_version,
    ):
        ok, reason = FlashInferFusedMoE.can_implement(QuantAlgo.NVFP4)
    assert ok
    assert reason is None


@pytest.mark.parametrize(
    "quant_algo",
    [
        None,
        QuantAlgo.FP8,
        QuantAlgo.FP8_BLOCK_SCALES,
        QuantAlgo.W4A16_MXFP4,
        QuantAlgo.W4A8_MXFP4_FP8,
    ],
)
def test_can_implement_rejects_non_nvfp4(quant_algo):
    """Only NVFP4 is supported; everything else must be turned away."""
    with patch(
        "tensorrt_llm._torch.modules.fused_moe.fused_moe_flashinfer.get_sm_version",
        return_value=120,
    ):
        ok, reason = FlashInferFusedMoE.can_implement(quant_algo)
    assert not ok
    assert reason is not None and "NVFP4" in reason


def test_can_implement_rejects_swiglu_gptoss_style():
    with patch(
        "tensorrt_llm._torch.modules.fused_moe.fused_moe_flashinfer.get_sm_version",
        return_value=120,
    ):
        ok, reason = FlashInferFusedMoE.can_implement(QuantAlgo.NVFP4, swiglu_gptoss_style=True)
    assert not ok
    assert reason is not None and "swiglu_gptoss_style" in reason


@pytest.mark.parametrize("dtype", [torch.float32, torch.float8_e4m3fn])
def test_can_implement_rejects_unsupported_activation_dtype(dtype):
    with patch(
        "tensorrt_llm._torch.modules.fused_moe.fused_moe_flashinfer.get_sm_version",
        return_value=120,
    ):
        ok, reason = FlashInferFusedMoE.can_implement(QuantAlgo.NVFP4, dtype_activation=dtype)
    assert not ok
    assert reason is not None


def test_get_moe_cls_raises_on_non_nvfp4():
    """create_moe.get_moe_cls must hard-error rather than fall back silently."""
    cfg = ModelConfig()
    cfg.moe_backend = "FLASHINFER"
    cfg.quant_config = QuantConfig(quant_algo=QuantAlgo.FP8)
    with pytest.raises(ValueError, match="NVFP4"):
        get_moe_cls(cfg)


def test_get_moe_cls_raises_on_missing_quant():
    cfg = ModelConfig()
    cfg.moe_backend = "FLASHINFER"
    cfg.quant_config = None
    with pytest.raises(ValueError, match="NVFP4"):
        get_moe_cls(cfg)


def test_get_moe_cls_raises_on_unsupported_sm():
    cfg = ModelConfig()
    cfg.moe_backend = "FLASHINFER"
    cfg.quant_config = QuantConfig(quant_algo=QuantAlgo.NVFP4)
    with patch("tensorrt_llm._utils.get_sm_version", return_value=100):
        with pytest.raises(ValueError, match="SM"):
            get_moe_cls(cfg)


def test_get_moe_cls_returns_flashinfer_on_supported_sm():
    cfg = ModelConfig()
    cfg.moe_backend = "FLASHINFER"
    cfg.quant_config = QuantConfig(quant_algo=QuantAlgo.NVFP4)
    with patch("tensorrt_llm._utils.get_sm_version", return_value=120):
        cls = get_moe_cls(cfg)
    assert cls is FlashInferFusedMoE

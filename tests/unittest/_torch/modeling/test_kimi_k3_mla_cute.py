# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""K3 MLA with the CuTe DSL kernels (TLLM_K3_CUTE_MLA) against the Hugging Face reference, at one DEP rank's 96 heads.

Two MLA layers share one KVCacheManager V1 pool, so the latent cache is interleaved by layer and layer 1 lives at a
non-zero slot: the decode page table that the CuTe latent append hands to CuteDslMlaFmha (the
``_k3_cute_mla_page_table`` stash) must carry that slot. Ragged context lengths cross page boundaries.
"""

import math
import weakref

import pytest
import torch
from _torch.modeling.kimi_k3_mla_reference import KimiMLAAttention
from _torch.modeling.test_kimi_k3_mla import _assert_matches_reference, _checkpoint_weights

from tensorrt_llm._torch.attention import mla as mla_module
from tensorrt_llm._torch.attention.backends.fmha.cute_dsl_mla import CuteDslMlaFmha
from tensorrt_llm._torch.attention.backends.fmha.cute_dsl_mla_context import CuteDslMlaContextFmha
from tensorrt_llm._torch.attention.backends.fmha.fallback import FallbackFmha
from tensorrt_llm._torch.attention.backends.interface import KVCacheParams
from tensorrt_llm._torch.attention.backends.utils import get_attention_backend
from tensorrt_llm._torch.configs.kimi_linear import KimiLinearConfig
from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_kimi_linear import KimiLinearForCausalLM
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, SamplingConfig
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm._torch.utils import model_extra_attrs
from tensorrt_llm._utils import get_sm_version, str_dtype_to_binding, torch_dtype_to_str
from tensorrt_llm.bindings.internal.batch_manager import CacheType
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.sampling_params import SamplingParams

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not IS_CUTLASS_DSL_AVAILABLE
    or get_sm_version() not in (100, 103, 107),
    reason="Kimi K3 MLA CuTe DSL kernels run on SM100/SM103/SM107 only",
)

NUM_HEADS, NUM_LAYERS, PAGE = 96, 2, 64
LENS = [130, 64, 257]


def _config() -> KimiLinearConfig:
    config = KimiLinearConfig(
        vocab_size=128,
        hidden_size=256,
        intermediate_size=256,
        num_hidden_layers=NUM_LAYERS,
        attn_res_block_size=1,
        torch_dtype=torch.bfloat16,
        linear_attn_config={
            "kda_layers": [],
            "full_attn_layers": list(range(1, NUM_LAYERS + 1)),
            "num_heads": 32,
        },
        num_attention_heads=NUM_HEADS,
        num_key_value_heads=NUM_HEADS,
        q_lora_rank=1536,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        rms_norm_eps=1e-6,
        mla_use_nope=True,
        mla_use_output_gate=True,
        max_position_embeddings=512,
    )
    config._attn_implementation = "eager"
    return config


def _load(model: KimiLinearForCausalLM, projections: dict) -> None:
    """Zero checkpoint for the whole model, with the same MLA projections in every layer."""
    name_map, _, _ = model.checkpoint_name_plan("")
    config = model.model_config.pretrained_config
    params = dict(model.named_parameters())
    weights = {}
    for name, key in name_map.items():
        param = params[name]
        if name.endswith("gate_up_proj.weight"):
            for source in ("gate_proj", "up_proj"):
                weights[key.replace("gate_up_proj", source)] = torch.zeros(
                    config.intermediate_size, config.hidden_size, dtype=param.dtype
                )
            continue
        shape = param.shape
        if name == "lm_head.weight":
            shape = (config.vocab_size, config.hidden_size)
        elif name.endswith(".mlp.down_proj.weight"):
            shape = (config.hidden_size, config.intermediate_size)
        weights[key] = torch.zeros(shape, dtype=param.dtype)
    for layer in range(NUM_LAYERS):
        weights.update(
            {f"model.layers.{layer}.self_attn.{name}": value for name, value in projections.items()}
        )
    model.load_weights(weights)


def _kv_cache_manager(mapping: Mapping, head_dim: int) -> KVCacheManager:
    blocks = sum(math.ceil((n + 1) / PAGE) for n in LENS) + len(LENS)
    manager = KVCacheManager(
        KvCacheConfig(max_tokens=blocks * PAGE, enable_block_reuse=False),
        CacheType.SELFKONLY,
        num_layers=NUM_LAYERS,
        num_kv_heads=1,
        head_dim=head_dim,
        tokens_per_block=PAGE,
        max_seq_len=max(LENS) + PAGE,
        max_batch_size=len(LENS),
        mapping=mapping,
        dtype=str_dtype_to_binding(torch_dtype_to_str(torch.bfloat16)),
    )
    for request_id, length in enumerate(LENS):
        request = LlmRequest(
            request_id=request_id,
            max_new_tokens=2,
            input_tokens=[1] * length,
            sampling_config=SamplingConfig(SamplingParams()._get_sampling_config()),
            is_streaming=False,
        )
        request.paged_kv_block_ids = []
        manager.impl.add_sequence_batch([(request_id, length, 1)], [request])
    return manager


def _metadata(manager: KVCacheManager, mapping: Mapping, generation: bool):
    batch = len(LENS)
    return get_attention_backend("TRTLLM").Metadata(
        seq_lens=torch.tensor([1] * batch if generation else LENS, dtype=torch.int),
        request_ids=list(range(batch)),
        max_num_requests=batch,
        num_contexts=0 if generation else batch,
        prompt_lens=list(LENS),
        max_num_tokens=sum(LENS),
        kv_cache_manager=manager,
        kv_cache_params=KVCacheParams(
            use_cache=True,
            num_cached_tokens_per_seq=list(LENS) if generation else [0] * batch,
        ),
        mapping=mapping,
        enable_context_mla_with_cached_kv=generation,
    )


class _Spy:
    """Counts CuTe context runs and records the page-table stash around every CuteDslMlaFmha call."""

    def __init__(self, monkeypatch):
        self.context_runs = 0
        self.decode_stash = []
        run_context = CuteDslMlaContextFmha.run_mla_context
        decode_forward = CuteDslMlaFmha.forward

        def counted_context(fmha, params):
            self.context_runs += 1
            return run_context(fmha, params)

        def watched_decode(fmha, q, k, v, metadata, forward_args):
            before = getattr(metadata, "_k3_cute_mla_page_table", None)
            decode_forward(fmha, q, k, v, metadata, forward_args)
            after = getattr(metadata, "_k3_cute_mla_page_table", None)
            self.decode_stash.append(
                (fmha.attn.layer_idx, None if before is None else before[0], after)
            )

        monkeypatch.setattr(CuteDslMlaContextFmha, "run_mla_context", counted_context)
        monkeypatch.setattr(CuteDslMlaFmha, "forward", watched_decode)


def _build(monkeypatch, cute_mla: str, fmha_libs=None):
    monkeypatch.setenv("TLLM_K3_CUTE_MLA", cute_mla)
    if fmha_libs is None:
        monkeypatch.delenv("TLLM_FMHA_LIBS", raising=False)
    else:
        monkeypatch.setenv("TLLM_FMHA_LIBS", fmha_libs)
    mla_module.k3_cute_mla_enabled.cache_clear()
    torch.manual_seed(42)
    config = _config()
    mapping = Mapping(world_size=1, rank=0, tp_size=1)
    reference = KimiMLAAttention(config, layer_idx=0).bfloat16().eval()
    weights = _checkpoint_weights(reference, fp8=False)
    with torch.device("cuda"):
        model = KimiLinearForCausalLM(
            ModelConfig(pretrained_config=config, mapping=mapping, quant_config=QuantConfig())
        ).eval()
    projections = {
        f"{name}.{suffix}": value
        for name, pair in weights.items()
        for suffix, value in pair.items()
    }
    projections.update(
        {name: value for name, value in reference.named_parameters() if value.ndim == 1}
    )
    _load(model, projections)
    runtimes = [model.model.layers[i].self_attn for i in range(NUM_LAYERS)]
    for runtime in runtimes:
        for module in runtime.modules():
            if callable(getattr(module, "post_load_weights", None)):
                module.post_load_weights()
    return config, mapping, reference.cuda(), runtimes


def _inputs(config, reference):
    """Per layer: request token rows [len + 1, hidden] and the reference output of each request."""
    generator = torch.Generator(device="cuda").manual_seed(1234)
    per_layer = []
    for _ in range(NUM_LAYERS):
        hidden, expected = [], []
        for length in LENS:
            rows = torch.randn(
                length + 1,
                config.hidden_size,
                dtype=torch.bfloat16,
                device="cuda",
                generator=generator,
            )
            mask = torch.full((length + 1, length + 1), float("-inf"), device="cuda").triu(1)[
                None, None
            ]
            with torch.no_grad():
                expected.append(reference(rows[None], attention_mask=mask)[0])
            hidden.append(rows)
        per_layer.append((hidden, expected))
    return per_layer


def _run(config, mapping, reference, runtimes):
    """Context step then one decode step through every layer, each compared with the reference."""
    per_layer = _inputs(config, reference)
    manager = _kv_cache_manager(mapping, config.kv_lora_rank + config.qk_rope_head_dim)
    try:
        pool = manager.get_buffers(1)
        assert pool.stride(0) == NUM_LAYERS * pool[0].numel() and pool.storage_offset() > 0, (
            "expected layer 1 at a non-zero slot of an interleaved V1 pool"
        )
        layers = {rt.mixer.layer_idx_str: weakref.ref(rt.mixer) for rt in runtimes}
        metadata = _metadata(manager, mapping, generation=False)
        metadata.prepare()
        with model_extra_attrs({"attention_metadata": weakref.ref(metadata), "mla_layers": layers}):
            for runtime, (hidden, expected) in zip(runtimes, per_layer):
                context = torch.cat([rows[:-1] for rows in hidden]).contiguous()
                actual = runtime(context, metadata)
                _assert_matches_reference(actual, torch.cat([e[:-1] for e in expected]), False)

        for request_id in range(len(LENS)):
            manager.impl.add_token(request_id)
        metadata = _metadata(manager, mapping, generation=True)
        metadata.prepare()
        with model_extra_attrs({"attention_metadata": weakref.ref(metadata), "mla_layers": layers}):
            for runtime, (hidden, expected) in zip(runtimes, per_layer):
                last = torch.stack([rows[-1] for rows in hidden]).contiguous()
                actual = runtime(last, metadata)
                _assert_matches_reference(actual, torch.stack([e[-1] for e in expected]), False)
    finally:
        manager.shutdown()


@pytest.mark.parametrize(
    "cute_mla,fmha_libs",
    [("0", None), ("1", None), ("1", "-cute_dsl_mla_context")],
    ids=["stock", "cute", "cute_without_context_library"],
)
def test_k3_mla_cute_matches_huggingface_interleaved_pool(monkeypatch, cute_mla, fmha_libs) -> None:
    spy = _Spy(monkeypatch)
    try:
        _run(*_build(monkeypatch, cute_mla, fmha_libs))
    finally:
        mla_module.k3_cute_mla_enabled.cache_clear()
    assert [layer for layer, _, _ in spy.decode_stash] == list(range(NUM_LAYERS))
    assert all(after is None for _, _, after in spy.decode_stash)
    if cute_mla == "0":
        assert spy.context_runs == 0
        assert all(stashed is None for _, stashed, _ in spy.decode_stash)
    else:
        assert spy.context_runs == (NUM_LAYERS if fmha_libs is None else 0)
        assert all(stashed == layer for layer, stashed, _ in spy.decode_stash)


def test_k3_mla_cute_context_served_elsewhere_raises(monkeypatch) -> None:
    """With the concat skipped, a context call that cute_dsl_mla_context does not take must not pass silently."""
    config, mapping, reference, runtimes = _build(monkeypatch, "1", "cute_dsl_mla_context,fallback")
    monkeypatch.setattr(CuteDslMlaContextFmha, "_is_supported", lambda *args, **kwargs: False)
    # The fallback would read the contiguous V with the kv-shaped stride; only the guard after it is under test.
    served = []
    monkeypatch.setattr(FallbackFmha, "forward", lambda *args, **kwargs: served.append(True))
    manager = _kv_cache_manager(mapping, config.kv_lora_rank + config.qk_rope_head_dim)
    try:
        metadata = _metadata(manager, mapping, generation=False)
        metadata.prepare()
        hidden = torch.randn(sum(LENS), config.hidden_size, dtype=torch.bfloat16, device="cuda")
        layers = {rt.mixer.layer_idx_str: weakref.ref(rt.mixer) for rt in runtimes}
        with model_extra_attrs({"attention_metadata": weakref.ref(metadata), "mla_layers": layers}):
            with pytest.raises(RuntimeError, match="concat was skipped"):
                runtimes[0](hidden, metadata)
        assert served
    finally:
        manager.shutdown()
        mla_module.k3_cute_mla_enabled.cache_clear()

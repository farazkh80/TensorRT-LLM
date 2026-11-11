# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests of FineGrainedSyncConfig and the Kimi K3 recipe it installs (kimi_k3_fgs.recipe)."""

import os

import pydantic
import pytest
import yaml

from tensorrt_llm._torch.modules.kimi_k3_fgs import edges, mla_chain, moe_chain, recipe
from tensorrt_llm.llmapi.llm_args import FineGrainedSyncConfig, update_llm_args_with_extra_dict

RECIPE_YAML = """
fine_grained_sync_config:
  mode: chain_after
  reset_policy: step
  edges:
    kda.x_norm: [[8, 64]]
    kda.attn_o: [[8, 8], [32, 64]]
    mla.chain: [[2, 32]]
    moe.recv_norm: [[1, 64]]
    moe.fc1_fc2: [[1, 128]]
  kernel_switches:
    TLLM_K3_MOE: cute
"""


@pytest.fixture(autouse=True)
def clean_state():
    env = dict(os.environ)
    for names in recipe.FAMILY_ENVS.values():
        for name in names:
            os.environ.pop(name, None)
    for name in (recipe.MODE_ENV, recipe.RESET_ENV, "TLLM_K3_MOE"):
        os.environ.pop(name, None)
    saved = (
        edges.MIN_ROWS,
        edges.MAX_ROWS,
        dict(edges.DROPPED),
        mla_chain.MIN_ROWS,
        mla_chain.MAX_ROWS,
        moe_chain.FGS_MAX_ROWS,
        moe_chain.MIX_MAX_ROWS,
    )
    yield
    recipe.install(None)
    (
        edges.MIN_ROWS,
        edges.MAX_ROWS,
        edges.DROPPED,
        mla_chain.MIN_ROWS,
        mla_chain.MAX_ROWS,
        moe_chain.FGS_MAX_ROWS,
        moe_chain.MIX_MAX_ROWS,
    ) = saved
    os.environ.clear()
    os.environ.update(env)


def config(text=RECIPE_YAML):
    return FineGrainedSyncConfig(**yaml.safe_load(text)["fine_grained_sync_config"])


def test_parse_and_window_lookup():
    c = config()
    assert c.windows("kda.attn_o") == ((8, 8), (32, 64))
    assert c.edge_on("kda.attn_o", 8) and not c.edge_on("kda.attn_o", 16)
    assert c.edge_on("kda.attn_o", 64) and not c.edge_on("kda.attn_o", 128)
    assert c.windows("kda.qkvg") == ()
    assert not c.edge_on("kda.qkvg", 16)


@pytest.mark.parametrize(
    "edges_yaml",
    [
        "{kda.x_norm: [[8, 32], [16, 64]]}",  # overlap
        "{kda.x_norm: [[8, 32], [32, 64]]}",  # shared end row
        "{kda.x_norm: [[16, 8]]}",  # empty window
        "{kda.x_norm: [[0, 8]]}",  # rows start at 1
        "{kda.x_norm: []}",  # no window
    ],
)
def test_bad_windows_rejected(edges_yaml):
    with pytest.raises(pydantic.ValidationError):
        FineGrainedSyncConfig(edges=yaml.safe_load(edges_yaml))


def test_unknown_field_rejected():
    with pytest.raises(pydantic.ValidationError):
        FineGrainedSyncConfig(edge={"kda.x_norm": [[8, 64]]})


def test_extra_options_yaml_builds_config():
    args = update_llm_args_with_extra_dict({}, yaml.safe_load(RECIPE_YAML))
    assert isinstance(args["fine_grained_sync_config"], FineGrainedSyncConfig)
    assert args["fine_grained_sync_config"].windows("mla.chain") == ((2, 32),)


def test_unknown_edge_rejected_at_install():
    with pytest.raises(ValueError, match="unknown edge"):
        recipe.install(FineGrainedSyncConfig(edges={"kda.xnorm": [[8, 64]]}))


def test_non_moe_switch_rejected_at_install():
    with pytest.raises(ValueError, match="not fused-MoE switches"):
        recipe.install(FineGrainedSyncConfig(kernel_switches={"TLLM_K3_CUTE_NORM": "1"}))


def test_install_sets_windows_and_defaults():
    recipe.install(config())
    assert os.environ[recipe.MODE_ENV] == "chain_after"
    assert os.environ[recipe.RESET_ENV] == "step"
    assert os.environ["TLLM_K3_MOE"] == "cute"
    assert os.environ[recipe.INBOX_ENV] == "combine"
    assert (edges.MIN_ROWS, edges.MAX_ROWS, edges.DROPPED) == (8, 64, {})
    assert (mla_chain.MIN_ROWS, mla_chain.MAX_ROWS) == (2, 32)
    assert moe_chain.FGS_MAX_ROWS == 128
    assert recipe.slot_allowed("kda.attn_o", 8)
    assert not recipe.slot_allowed("kda.attn_o", 16)
    assert not recipe.slot_allowed("kda.qkvg", 16)
    assert recipe.slot_allowed("mla.q", 2) and not recipe.slot_allowed("mla.q", 1)
    assert recipe.slot_allowed("mla.gated", 16) and not recipe.slot_allowed("mla.gated", 64)
    assert recipe.moe_edge_names() == frozenset({"recv_norm", "fc1_fc2"})
    assert moe_chain.fgs_edges() == frozenset({"recv_norm", "fc1_fc2"})
    assert recipe.moe_on("fc1_fc2", 128) and not recipe.moe_on("recv_norm", 128)
    assert moe_chain._edge_rows("recv_norm") == (1, 64)


def test_environment_wins():
    os.environ[recipe.MODE_ENV] = "off"
    os.environ["TLLM_K3_MOE_FGS"] = "fc1_fc2"
    os.environ["TLLM_K3_FGS_MLA_MAX_ROWS"] = "16"
    recipe.install(config())
    assert os.environ[recipe.MODE_ENV] == "off"
    assert not recipe.owns("moe") and not recipe.owns("mla") and recipe.owns("kda")
    assert recipe.moe_edge_names() is None
    assert recipe.moe_on("recv_norm", 128)
    assert recipe.slot_allowed("mla.q", 1)
    assert recipe.INBOX_ENV not in os.environ


def test_empty_edges_is_fgs_off():
    recipe.install(FineGrainedSyncConfig(edges={}))
    assert os.environ[recipe.MODE_ENV] == "off"
    assert edges.MIN_ROWS > edges.MAX_ROWS
    assert mla_chain.MIN_ROWS > mla_chain.MAX_ROWS
    assert moe_chain.fgs_edges() == frozenset()


def test_absent_config_keeps_environment_behaviour():
    before = (edges.MIN_ROWS, edges.MAX_ROWS, mla_chain.MIN_ROWS, mla_chain.MAX_ROWS)
    recipe.install(None)
    assert recipe.MODE_ENV not in os.environ
    assert (edges.MIN_ROWS, edges.MAX_ROWS, mla_chain.MIN_ROWS, mla_chain.MAX_ROWS) == before
    assert recipe.slot_allowed("kda.attn_o", 16) and recipe.moe_on("recv_norm", 128)

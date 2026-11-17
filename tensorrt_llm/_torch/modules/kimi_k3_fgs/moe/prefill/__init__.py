# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 prefill MoE kernels (SM107): routing for any received-row count, token-M block-scaled grouped GEMMs for the
routed experts, dense GEMMs with fused epilogues (latent down + MXFP8 quantize, shared expert, latent up) and the
chunk-sized Lamport combine inbox with the fused reduce / latent RMSNorm."""

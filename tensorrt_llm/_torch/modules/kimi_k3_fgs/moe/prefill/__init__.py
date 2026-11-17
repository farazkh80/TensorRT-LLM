# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 prefill MoE kernels (SM107): token-M block-scaled grouped GEMMs for the routed experts and the chunk-sized
Lamport combine inbox with the fused reduce / latent RMSNorm."""

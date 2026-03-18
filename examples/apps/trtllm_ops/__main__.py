# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Entry point: python -m examples.apps.trtllm_ops [viz|ops]"""

import sys


def main():
    if len(sys.argv) > 1 and sys.argv[1] in ('viz', 'visualizer', 'top'):
        sys.argv = sys.argv[:1] + sys.argv[2:]
        from .visualizer import main as viz_main
        viz_main()
    else:
        # Default to ops agent; strip 'ops' subcommand if present
        if len(sys.argv) > 1 and sys.argv[1] == 'ops':
            sys.argv = sys.argv[:1] + sys.argv[2:]
        from .cli import main as cli_main
        cli_main()


if __name__ == "__main__":
    main()

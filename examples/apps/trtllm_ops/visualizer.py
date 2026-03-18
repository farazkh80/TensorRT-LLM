# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""trtllm-top: Live inference visualizer TUI for trtllm-serve."""

import argparse
import re
import subprocess
import time

import httpx
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


def parse_prometheus_text(text: str) -> dict:
    """Parse Prometheus text format into flat dict of metric_name -> value."""
    result = {}
    for line in text.strip().split('\n'):
        if line.startswith('#') or not line.strip():
            continue
        match = re.match(r'^(\w+)(?:\{[^}]*\})?\s+([\d.eE+-]+)', line)
        if match:
            name = match.group(1)
            try:
                val = float(match.group(2))
                # Sum duplicate metrics (e.g. counters with different labels)
                result[name] = result.get(name, 0) + val
            except ValueError:
                pass
    return result


def get_gpu_bars() -> list[dict]:
    """Get GPU info for visualization."""
    try:
        out = subprocess.check_output(
            ['nvidia-smi',
             '--query-gpu=name,utilization.gpu,memory.used,memory.total,'
             'temperature.gpu,power.draw',
             '--format=csv,noheader,nounits'],
            text=True, timeout=5)
        gpus = []
        for line in out.strip().split('\n'):
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 5:
                gpus.append({
                    'name': parts[0],
                    'util': int(parts[1]),
                    'mem_used': int(parts[2]),
                    'mem_total': int(parts[3]),
                    'temp': int(parts[4]),
                    'power': float(parts[5]) if len(parts) > 5 else 0,
                })
        return gpus
    except Exception:
        return []


def _bar(pct: float, width: int = 20) -> str:
    """Render a text progress bar with color."""
    filled = int(pct / 100 * width)
    empty = width - filled
    if pct < 70:
        color = "green"
    elif pct < 90:
        color = "yellow"
    else:
        color = "red"
    return f"[{color}]{'█' * filled}{'░' * empty}[/] {pct:.0f}%"


def build_dashboard(endpoint: str, gpus: list, prom: dict,
                    iter_stats: dict, elapsed: float) -> Layout:
    """Build the rich Layout dashboard."""
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="body"),
        Layout(name="footer", size=3),
    )
    layout["body"].split_row(
        Layout(name="left", ratio=1),
        Layout(name="right", ratio=1),
    )

    # --- Header ---
    model = "unknown"
    if isinstance(iter_stats, dict):
        model = iter_stats.get('model', iter_stats.get('model_name', 'unknown'))
    header = Text(
        f" trtllm-top — {model} @ {endpoint}", style="bold cyan")
    layout["header"].update(Panel(header, style="cyan"))

    # --- GPU panel (left) ---
    gpu_table = Table(expand=True, show_header=True, header_style="bold")
    gpu_table.add_column("GPU", width=4)
    gpu_table.add_column("Name", width=18)
    gpu_table.add_column("Util", width=28)
    gpu_table.add_column("Memory", width=28)
    gpu_table.add_column("Temp", width=6)
    gpu_table.add_column("Power", width=8)

    for i, g in enumerate(gpus):
        mem_pct = g['mem_used'] / max(g['mem_total'], 1) * 100
        gpu_table.add_row(
            str(i),
            g['name'][:18],
            _bar(g['util']),
            _bar(mem_pct),
            f"{g['temp']}°C",
            f"{g['power']:.0f}W" if g['power'] else "?",
        )

    layout["left"].update(Panel(gpu_table, title="[bold]GPUs[/]"))

    # --- Metrics panel (right) ---
    m_table = Table(expand=True, show_header=True, header_style="bold")
    m_table.add_column("Metric", width=22)
    m_table.add_column("Value", width=30)

    # KV cache
    kv_util = prom.get('trtllm_kv_cache_utilization', 0)
    kv_hit = prom.get('trtllm_kv_cache_hit_rate', 0)
    m_table.add_row("KV Cache Util", _bar(kv_util * 100))
    m_table.add_row("KV Cache Hit Rate", f"{kv_hit:.1%}")

    # Request counts
    total_req = prom.get('trtllm_request_success_total', 0)
    m_table.add_row("Requests (total)", f"{total_req:.0f}")

    # Latency
    ttft_count = prom.get('trtllm_time_to_first_token_seconds_count', 0)
    ttft_sum = prom.get('trtllm_time_to_first_token_seconds_sum', 0)
    if ttft_count > 0:
        m_table.add_row("Avg TTFT", f"{ttft_sum / ttft_count:.3f}s")

    e2e_count = prom.get('trtllm_e2e_request_latency_seconds_count', 0)
    e2e_sum = prom.get('trtllm_e2e_request_latency_seconds_sum', 0)
    if e2e_count > 0:
        m_table.add_row("Avg E2E Latency", f"{e2e_sum / e2e_count:.3f}s")

    # From /metrics JSON if available
    if isinstance(iter_stats, dict):
        for key in ['num_active_requests', 'num_queued_requests',
                    'max_num_active_requests', 'tokens_per_second',
                    'generation_requests']:
            val = iter_stats.get(key)
            if val is not None:
                label = key.replace('_', ' ').title()
                m_table.add_row(label, str(val))

    layout["right"].update(Panel(m_table, title="[bold]Inference[/]"))

    # --- Footer ---
    layout["footer"].update(Panel(
        Text(f" Polling every 1s  |  Uptime: {elapsed:.0f}s  |  "
             f"Ctrl+C to exit", style="dim"),
        style="dim"))

    return layout


def run_visualizer(endpoint: str, poll_interval: float = 1.0):
    """Main loop for the live TUI."""
    console = Console()
    start_time = time.time()
    console.print(f"[bold cyan]trtllm-top[/] connecting to {endpoint}...")

    with Live(console=console, refresh_per_second=2, screen=True) as live:
        while True:
            try:
                gpus = get_gpu_bars()

                try:
                    prom_resp = httpx.get(
                        f"{endpoint}/prometheus/metrics", timeout=3)
                    prom = parse_prometheus_text(
                        prom_resp.text) if prom_resp.status_code == 200 else {}
                except Exception:
                    prom = {}

                try:
                    iter_resp = httpx.get(f"{endpoint}/metrics", timeout=3)
                    iter_stats = (
                        iter_resp.json()
                        if iter_resp.status_code == 200 else {})
                except Exception:
                    iter_stats = {}

                elapsed = time.time() - start_time
                dashboard = build_dashboard(
                    endpoint, gpus, prom, iter_stats, elapsed)
                live.update(dashboard)

                time.sleep(poll_interval)

            except KeyboardInterrupt:
                break


def main():
    parser = argparse.ArgumentParser(
        description="Live TRT-LLM inference visualizer")
    parser.add_argument(
        '--endpoint', default='http://localhost:8000',
        help='TRT-LLM server endpoint')
    parser.add_argument(
        '--interval', type=float, default=1.0,
        help='Polling interval in seconds')
    args = parser.parse_args()
    run_visualizer(args.endpoint, args.interval)


if __name__ == "__main__":
    main()

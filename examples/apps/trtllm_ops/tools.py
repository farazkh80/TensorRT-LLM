# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Tool functions for the trtllm-ops agent. Each returns a dict or string."""

import json
import subprocess

import httpx


def get_gpu_status() -> list[dict]:
    """Read GPU status via nvidia-smi."""
    try:
        out = subprocess.check_output(
            ['nvidia-smi',
             '--query-gpu=name,utilization.gpu,memory.total,memory.used,'
             'temperature.gpu,power.draw',
             '--format=csv,noheader,nounits'],
            text=True, timeout=5)
        gpus = []
        for line in out.strip().split('\n'):
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 5:
                gpus.append({
                    'name': parts[0],
                    'utilization_pct': int(parts[1]),
                    'memory_total_mb': int(parts[2]),
                    'memory_used_mb': int(parts[3]),
                    'temperature_c': int(parts[4]),
                    'power_w': float(parts[5]) if len(parts) > 5 else None,
                })
        return gpus
    except Exception as e:
        return [{'error': str(e)}]


def get_server_metrics(endpoint: str) -> dict:
    """Read /metrics and /perf_metrics from trtllm-serve."""
    result = {}
    for path in ['/metrics', '/perf_metrics']:
        try:
            resp = httpx.get(f"{endpoint}{path}", timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, list) and data:
                    result[path] = data[-1]  # latest entry
                elif isinstance(data, dict):
                    result[path] = data
                else:
                    result[path] = "empty (no data yet)"
            else:
                result[path] = f"HTTP {resp.status_code}"
        except Exception as e:
            result[path] = f"error: {e}"
    # Also grab model info
    try:
        resp = httpx.get(f"{endpoint}/v1/models", timeout=5)
        if resp.status_code == 200:
            models = resp.json()
            if models.get('data'):
                result['model'] = models['data'][0]['id']
    except Exception:
        pass
    return result


def get_prometheus_metrics(endpoint: str) -> str:
    """Read /prometheus/metrics text from trtllm-serve."""
    try:
        resp = httpx.get(f"{endpoint}/prometheus/metrics", timeout=5)
        if resp.status_code == 200:
            return resp.text
        return ("UNAVAILABLE: Prometheus metrics endpoint returned HTTP "
                f"{resp.status_code}. This data is not available — do NOT "
                "fabricate numbers. Report that prometheus metrics are not "
                "enabled on this server.")
    except Exception as e:
        return f"UNAVAILABLE: {e}. Do NOT fabricate numbers."


def get_server_health(endpoint: str) -> dict:
    """Check /health endpoint."""
    try:
        resp = httpx.get(f"{endpoint}/health", timeout=5)
        return {
            'status': 'healthy' if resp.status_code == 200 else 'unhealthy',
            'code': resp.status_code,
        }
    except Exception as e:
        return {'status': 'unreachable', 'error': str(e)}


def estimate_capacity(
    gpu_memory_total_mb: int,
    gpu_memory_used_mb: int,
    avg_seq_len: int = 2048,
    kv_bytes_per_token: float = 0.5 * 1024 * 1024,
) -> dict:
    """Estimate how many more concurrent users can fit."""
    headroom_mb = gpu_memory_total_mb - gpu_memory_used_mb
    headroom_bytes = headroom_mb * 1024 * 1024
    kv_per_request = avg_seq_len * kv_bytes_per_token
    additional = int(headroom_bytes / kv_per_request) if kv_per_request > 0 else 0
    return {
        'gpu_headroom_mb': headroom_mb,
        'kv_per_request_mb': round(kv_per_request / (1024 * 1024), 1),
        'estimated_additional_users': additional,
        'avg_seq_len_assumed': avg_seq_len,
    }


def get_recent_server_logs(n_lines: int = 50, level: str = None) -> str:
    """Grab recent server logs using methods available in containers."""
    lines = []

    # Method 1: Our own captured logs
    for path in ['/tmp/trtllm-ops-logs/server.log',
                 '/tmp/trtllm-serve.log']:
        try:
            result = subprocess.check_output(
                ['tail', '-n', str(n_lines), path],
                text=True, timeout=5, stderr=subprocess.DEVNULL)
            if result.strip():
                lines = result.strip().split('\n')
                break
        except Exception:
            pass

    # Method 2: Find trtllm-serve PID, read /proc/PID/fd/1 and /2
    if not lines:
        try:
            pids = subprocess.check_output(
                ['bash', '-c',
                 'pgrep -f "trtllm.commands.serve|trtllm-serve|uvicorn" '
                 '| head -3'],
                text=True, timeout=5).strip()
            for pid in pids.split('\n'):
                pid = pid.strip()
                if not pid:
                    continue
                for fd in ['2', '1']:  # stderr first, then stdout
                    proc_path = f"/proc/{pid}/fd/{fd}"
                    try:
                        # /proc/pid/fd/1 and /2 are usually pipes or ptys
                        # We can't tail them directly, but we can check
                        # if they point to a file
                        import os
                        real_path = os.readlink(proc_path)
                        if real_path.startswith('/') and os.path.isfile(
                                real_path):
                            result = subprocess.check_output(
                                ['tail', '-n', str(n_lines), real_path],
                                text=True, timeout=5,
                                stderr=subprocess.DEVNULL)
                            if result.strip():
                                lines = result.strip().split('\n')
                                break
                    except Exception:
                        pass
                if lines:
                    break
        except Exception:
            pass

    # Method 3: Check dmesg for kernel-level issues (OOM, GPU errors)
    if not lines:
        try:
            result = subprocess.check_output(
                ['dmesg'], text=True, timeout=5,
                stderr=subprocess.DEVNULL)
            all_lines = result.strip().split('\n')
            relevant = [l for l in all_lines
                        if any(k in l.lower() for k in
                               ['oom', 'kill', 'gpu', 'cuda', 'nccl',
                                'error', 'fault'])]
            lines = relevant[-n_lines:] if relevant else all_lines[-n_lines:]
        except Exception:
            pass

    # Method 4: Check common container log locations
    if not lines:
        import glob
        for pattern in ['/var/log/*.log', '/tmp/*.log']:
            for path in sorted(glob.glob(pattern),
                               key=lambda p: os.path.getmtime(p),
                               reverse=True)[:3]:
                try:
                    result = subprocess.check_output(
                        ['tail', '-n', str(n_lines), path],
                        text=True, timeout=5, stderr=subprocess.DEVNULL)
                    if result.strip():
                        lines = result.strip().split('\n')
                        break
                except Exception:
                    pass
            if lines:
                break

    if not lines:
        return (
            "No server logs found via file, /proc, or dmesg. "
            "The server is likely writing to stdout inside this container. "
            "To capture logs, restart the server with:\n"
            "  trtllm-serve ... 2>&1 | tee /tmp/trtllm-serve.log\n\n"
            "Or use run_shell_command with:\n"
            "  'ps aux | grep trtllm' to find the process\n"
            "  'nvidia-smi' for GPU-level diagnostics"
        )

    if level:
        lines = [l for l in lines if level.upper() in l.upper()]

    return '\n'.join(lines[-n_lines:])


def run_shell_command(command: str) -> str:
    """Run a shell command and return output."""
    try:
        result = subprocess.check_output(
            command, shell=True, text=True, timeout=30,
            stderr=subprocess.STDOUT)
        return result[:4000]
    except subprocess.CalledProcessError as e:
        return f"Exit code {e.returncode}:\n{e.output[:2000]}"
    except Exception as e:
        return f"Error: {e}"


# --- OpenAI function calling definitions ---

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "get_gpu_status",
            "description": "Get GPU utilization, memory, temperature, and power "
                           "via nvidia-smi for all GPUs on this machine",
            "parameters": {"type": "object", "properties": {}, "required": []},
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_server_metrics",
            "description": "Get server iteration stats (active requests, "
                           "throughput, queue depth) from /metrics endpoint",
            "parameters": {"type": "object", "properties": {}, "required": []},
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_prometheus_metrics",
            "description": "Get detailed Prometheus metrics including latency "
                           "histograms, KV cache utilization/hit rate, and "
                           "request success/error counts",
            "parameters": {"type": "object", "properties": {}, "required": []},
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_server_health",
            "description": "Check if the TRT-LLM server is healthy",
            "parameters": {"type": "object", "properties": {}, "required": []},
        }
    },
    {
        "type": "function",
        "function": {
            "name": "estimate_capacity",
            "description": "Estimate how many additional concurrent users the "
                           "server can handle based on GPU memory headroom and "
                           "KV cache requirements per request",
            "parameters": {
                "type": "object",
                "properties": {
                    "avg_seq_len": {
                        "type": "integer",
                        "description": "Average sequence length per request "
                                       "(default: 2048)"
                    }
                },
                "required": []
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_logs",
            "description": "Search server logs with regex pattern, filter by "
                           "log level (ERROR/WARNING/INFO) and time range. "
                           "Returns matching lines with timestamps.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Regex pattern to search for "
                                       "(e.g., 'OOM|eviction|CUDA error')"
                    },
                    "level": {
                        "type": "string",
                        "description": "Filter by log level: ERROR, WARNING, "
                                       "INFO, etc."
                    },
                    "since": {
                        "type": "string",
                        "description": "Only return logs after this timestamp "
                                       "(ISO format or HH:MM:SS)"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max results (default: 50)"
                    },
                },
                "required": []
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_log_summary",
            "description": "Get a quick health summary: error/warning counts "
                           "for last 1m/5m, last 5 errors with timestamps, "
                           "log capture status",
            "parameters": {"type": "object", "properties": {}, "required": []},
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_log_context",
            "description": "Get all log lines within ±N seconds of a specific "
                           "timestamp. Use after finding an error to see what "
                           "happened around it.",
            "parameters": {
                "type": "object",
                "properties": {
                    "timestamp": {
                        "type": "string",
                        "description": "Center timestamp (ISO format or "
                                       "HH:MM:SS)"
                    },
                    "window_secs": {
                        "type": "number",
                        "description": "Seconds before and after to include "
                                       "(default: 10)"
                    },
                },
                "required": ["timestamp"]
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_recent_server_logs",
            "description": "Get recent TRT-LLM server log lines. Tries "
                           "multiple methods: captured logs, process stdout, "
                           "dmesg. Use this for 'show me the logs' requests.",
            "parameters": {
                "type": "object",
                "properties": {
                    "n_lines": {
                        "type": "integer",
                        "description": "Number of recent lines (default: 50)"
                    },
                    "level": {
                        "type": "string",
                        "description": "Filter by level: ERROR, WARNING, INFO"
                    },
                },
                "required": []
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_shell_command",
            "description": "Run a shell command for advanced diagnostics "
                           "(e.g., dmesg, ps, netstat, df). Use sparingly.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Shell command to execute"
                    }
                },
                "required": ["command"]
            },
        }
    },
]


MAX_TOOL_OUTPUT = 3000  # chars — prevent context overflow


def _cap(text: str) -> str:
    """Cap tool output to avoid blowing the context window."""
    if len(text) > MAX_TOOL_OUTPUT:
        return text[:MAX_TOOL_OUTPUT] + f"\n... (truncated, {len(text)} total chars)"
    return text


def execute_tool(name: str, args: dict, endpoint: str,
                 log_manager=None) -> str:
    """Dispatch a tool call and return the result as a string."""
    if name == "get_gpu_status":
        return _cap(json.dumps(get_gpu_status(), indent=2))
    elif name == "get_server_metrics":
        return _cap(json.dumps(get_server_metrics(endpoint), indent=2))
    elif name == "get_prometheus_metrics":
        return _cap(get_prometheus_metrics(endpoint))
    elif name == "get_server_health":
        return _cap(json.dumps(get_server_health(endpoint), indent=2))
    elif name == "estimate_capacity":
        gpus = get_gpu_status()
        if gpus and 'error' not in gpus[0]:
            total = sum(g['memory_total_mb'] for g in gpus)
            used = sum(g['memory_used_mb'] for g in gpus)
        else:
            total, used = 0, 0
        return _cap(json.dumps(
            estimate_capacity(total, used, **args), indent=2))
    elif name == "search_logs":
        if log_manager:
            return _cap(json.dumps(
                log_manager.search_logs(**args), indent=2))
        return json.dumps({"error": "Log capture not active"})
    elif name == "get_log_summary":
        if log_manager:
            return _cap(json.dumps(
                log_manager.get_log_summary(), indent=2))
        return json.dumps({"error": "Log capture not active"})
    elif name == "get_log_context":
        if log_manager:
            return _cap(json.dumps(
                log_manager.get_log_context(**args), indent=2))
        return json.dumps({"error": "Log capture not active"})
    elif name == "get_recent_server_logs":
        return _cap(get_recent_server_logs(**args))
    elif name == "run_shell_command":
        return _cap(run_shell_command(args.get("command", "echo 'no command'")))
    else:
        return json.dumps({"error": f"Unknown tool: {name}"})

# trtllm-ops: AI-Powered Operations Agent + Live Visualizer

Two AI tools for monitoring and managing TRT-LLM inference servers.

## trtllm-ops: Chat with Your Server

The model running on your TRT-LLM server becomes its own ops engineer.
It reads its own metrics, greps its own logs, and reasons about its own
performance — all through natural language.

```bash
# Start your server
trtllm-serve meta-llama/Llama-3.1-8B-Instruct --backend pytorch --port 8000

# Chat with it
python -m examples.apps.trtllm_ops --endpoint http://localhost:8000

# Example questions:
#   "How's the server doing?"
#   "Why is latency high?"
#   "Can I handle 100 more concurrent users?"
#   "Any errors in the last 5 minutes?"
#   "What happened around 14:32?"
```

### Log capture

The agent captures server logs in the background for intelligent search:

```bash
# Auto-detect (tries docker, then file, then prometheus)
python -m examples.apps.trtllm_ops --endpoint http://localhost:8000

# Explicit docker container
python -m examples.apps.trtllm_ops --log-source docker --container-id abc123

# Explicit log file
python -m examples.apps.trtllm_ops --log-source file --log-file /tmp/server.log
```

### Tools

| Tool | Description |
|------|-------------|
| `get_gpu_status` | GPU util, memory, temp, power via nvidia-smi |
| `get_server_metrics` | Active requests, queue depth from /metrics |
| `get_prometheus_metrics` | Latency histograms, KV cache from /prometheus/metrics |
| `estimate_capacity` | How many more users fit? (memory math) |
| `search_logs` | Regex grep over captured logs, filter by level/time |
| `get_log_summary` | Error/warning counts (1m/5m), last 5 errors |
| `get_log_context` | All logs ±N seconds around a timestamp |
| `run_shell_command` | Run any shell command for diagnostics |

## trtllm-top: Live Inference Visualizer

Real-time terminal dashboard showing GPU utilization, memory, KV cache,
request throughput, and latency — all updating live.

```bash
python -m examples.apps.trtllm_ops viz --endpoint http://localhost:8000
```

## Requirements

```bash
pip install openai httpx rich
```

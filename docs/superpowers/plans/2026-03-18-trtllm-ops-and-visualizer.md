# TRT-LLM Ops Agent + Live Inference Visualizer — Hackathon Plan

**Goal:** Build two AI tools: (1) `trtllm-ops` — chat with a running TRT-LLM server about its own performance, and (2) `trtllm-top` — live TUI dashboard.

**Architecture:** Purely client-side. Connects to existing `trtllm-serve` endpoints. The model already running on the server reasons about its own state via tool/function calling.

**Tech Stack:** Python, `openai` SDK, `rich`, `httpx`. No TRT-LLM core changes.

**Endpoints used:**
- `POST /v1/chat/completions` — chat API with tool calling
- `GET /metrics` — iteration stats JSON
- `GET /prometheus/metrics` — Prometheus counters/histograms/gauges
- `GET /health` — health check
- `GET /v1/models` — model name detection

---

## File Structure

```
examples/apps/trtllm_ops/
├── __init__.py
├── __main__.py         # entry: python -m examples.apps.trtllm_ops [ops|viz]
├── cli.py              # ops agent chat REPL
├── tools.py            # tool functions (metrics, gpu, capacity, shell)
├── log_manager.py      # intelligent log capture, rotation, grep, analysis
├── system_prompt.py    # dynamic system prompt
├── visualizer.py       # rich Live TUI dashboard
└── README.md
```

---

## Task 1: Tool Functions + System Prompt (30 min)

**Create:** `tools.py`, `system_prompt.py`, `__init__.py`

### tools.py

Tool functions + OpenAI function calling definitions + dispatcher:

| Tool | Implementation |
|------|---------------|
| `get_gpu_status()` | `nvidia-smi --query-gpu=... --format=csv` → parse to list of dicts |
| `get_server_metrics(endpoint)` | `GET /metrics` → return JSON |
| `get_prometheus_metrics(endpoint)` | `GET /prometheus/metrics` → return raw text |
| `get_server_health(endpoint)` | `GET /health` → status dict |
| `estimate_capacity(gpu_total, gpu_used, avg_seq_len)` | Math: headroom / kv_per_request |
| `search_logs(pattern, level, since, n_lines)` | Smart log grep — see log_manager.py below |
| `get_log_summary()` | Recent error/warning counts + last N critical events |
| `get_log_context(timestamp, window)` | Get ±N seconds of logs around a timestamp |
| `run_shell_command(cmd)` | `subprocess.check_output` with timeout + output cap |

Plus `TOOL_DEFINITIONS` list (OpenAI function calling JSON schema) and `execute_tool(name, args, endpoint)` dispatcher.

### log_manager.py — Intelligent Log Handling

**Create:** `log_manager.py`

This is the brains behind log tools. The ops CLI captures server logs continuously in the background and provides smart search over them.

**How log capture works:**

When `trtllm-ops` starts, it begins tailing the server's log output in a background thread. Logs are stored in a ring buffer (last 10K lines) + written to a rotated log file under `/tmp/trtllm-ops-logs/`.

```python
class LogManager:
    """Captures and indexes TRT-LLM server logs for intelligent querying."""

    def __init__(self, endpoint: str, log_dir: str = "/tmp/trtllm-ops-logs"):
        self.endpoint = endpoint
        self.log_dir = log_dir
        self.ring_buffer = deque(maxlen=10000)  # last 10K lines in memory
        self.error_index = []     # [(timestamp, level, message)] for errors/warnings
        self._running = False
```

**Three capture methods (tries in order):**

1. **Docker logs** — if server is in a container, `docker logs -f <container_id> --since <start_time>` in a background thread
2. **Log file tailing** — if `--log-file` is passed, tail it with `watchdog` or simple poll
3. **Prometheus scraping** — fallback: poll `/prometheus/metrics` every 2s, track deltas (no raw logs but still catches error count changes)

The CLI auto-detects which method works. User can override with `--log-source docker|file|prometheus`.

**Log tools the agent can call:**

| Tool | What it does | When the agent would use it |
|------|-------------|----------------------------|
| `search_logs(pattern, level, since, limit)` | Regex grep over ring buffer. Filter by level (ERROR/WARNING/INFO). Filter by timestamp. Returns matching lines with context. | "Why did latency spike at 14:32?" → agent searches for errors around that time |
| `get_log_summary()` | Returns: error count last 1m/5m/15m, warning count, last 5 errors with timestamps, log capture rate | "Any problems?" → agent checks summary first |
| `get_log_context(timestamp, window_secs)` | Returns all log lines within ±window_secs of a timestamp | After finding an error, agent gets surrounding context |

**Log parsing intelligence:**

```python
# Known TRT-LLM log patterns parsed into structured data
LOG_PATTERNS = {
    'oom': r'(OutOfMemoryError|CUDA out of memory|OOM|oom-kill)',
    'kv_eviction': r'(evict|eviction|cache.*(full|pressure))',
    'cuda_error': r'(CUDA error|cudaError|cuda.*failed)',
    'timeout': r'(timeout|timed out|deadline exceeded)',
    'nccl': r'(NCCL|nccl.*(error|timeout|failed))',
    'request_error': r'(request.*failed|error.*request|500|Internal Server Error)',
    'loading': r'(loading.*model|shard|weight|checkpoint)',
    'ready': r'(ready|started|listening|health.*ok)',
}

def search_logs(self, pattern=None, level=None, since=None, limit=50):
    """Smart log search with structured output."""
    results = []
    for line in self.ring_buffer:
        parsed = self._parse_line(line)
        if level and parsed['level'] != level:
            continue
        if since and parsed['timestamp'] < since:
            continue
        if pattern and not re.search(pattern, line, re.IGNORECASE):
            continue
        results.append(parsed)
        if len(results) >= limit:
            break
    return results

def get_log_summary(self):
    """Quick health summary from logs."""
    now = time.time()
    errors_1m = sum(1 for ts, lvl, _ in self.error_index if now - ts < 60)
    errors_5m = sum(1 for ts, lvl, _ in self.error_index if now - ts < 300)
    last_errors = self.error_index[-5:] if self.error_index else []
    return {
        'errors_last_1m': errors_1m,
        'errors_last_5m': errors_5m,
        'warnings_last_5m': sum(1 for ts, lvl, _ in self.error_index
                                 if now - ts < 300 and lvl == 'WARNING'),
        'last_errors': [{'time': ts, 'level': lvl, 'msg': msg}
                        for ts, lvl, msg in last_errors],
        'buffer_size': len(self.ring_buffer),
        'capture_method': self.capture_method,
    }
```

**Why this matters for the ops agent:**

Without this, the agent is blind to WHY things happen. Metrics tell you "latency spiked" but logs tell you "CUDA OOM at 14:32:07, KV cache eviction storm started at 14:32:05." The agent can now:

1. See a latency spike in metrics
2. Call `search_logs(since="14:32:00", level="ERROR")` to find the cause
3. Call `get_log_context(timestamp="14:32:05", window_secs=10)` to see what happened around it
4. Reason: "KV cache hit 95% at 14:32:03, eviction started at 14:32:05, 3 OOM retries at 14:32:07. Root cause: batch of 128K context requests arrived simultaneously."

### system_prompt.py

`build_system_prompt(model_name, gpu_info, endpoint)` → returns a string with:
- Server info (model, GPU names, memory, endpoint)
- Role: "You are the ops agent, running ON this server, analyzing yourself"
- Rules: always call tools before answering, be precise with numbers, show math

### Verify
- [ ] `python -c "from examples.apps.trtllm_ops.tools import get_gpu_status; print(get_gpu_status())"` returns GPU info

---

## Task 2: CLI Chat Loop (30 min)

**Create:** `cli.py`

Core loop:
1. Parse args (`--endpoint`, `--model`, `--log-source docker|file|prometheus`, `--log-file`, `--container-id`)
2. Create `OpenAI` client pointing at `{endpoint}/v1`
2b. Start `LogManager` in background thread (captures logs from chosen source)
3. Auto-detect model name via `GET /v1/models`
4. Build system prompt with detected model + GPU info
5. REPL loop:
   - Read user input
   - Send to `/v1/chat/completions` with `tools=TOOL_DEFINITIONS`
   - If response has `tool_calls`: execute each tool, append results, re-send
   - Loop up to 5 tool rounds per question
   - Print final assistant response
   - Color output: green for user, yellow for tool calls, cyan for ops response

### Verify
- [ ] Start `trtllm-serve` with any model
- [ ] Run `python -m examples.apps.trtllm_ops.cli --endpoint http://localhost:8000`
- [ ] Ask "How's the server doing?" → model calls tools → responds with real metrics

---

## Task 3: Live Visualizer TUI (40 min)

**Create:** `visualizer.py`

Uses `rich.live.Live` with `Layout` to show:

```
┌─────────────────────────────────────────────────────┐
│  trtllm-top — Llama-3.1-8B @ localhost:8000         │
├──────────────────────┬──────────────────────────────┤
│  GPUs                │  Server Metrics              │
│  GPU 0: B300         │  Requests: 1,893             │
│  Util: ████████░░ 82%│  KV Cache: ███████░░░ 72%    │
│  Mem:  198/275 GB    │  KV Hit Rate: 34%            │
│  Temp: 62°C          │  TTFT: 0.028s                │
│                      │  E2E Latency: 0.142s         │
│  GPU 1: B300         │  Queue Time: 0.003s          │
│  Util: ████████░░ 79%│  Active Requests: 42         │
│  Mem:  195/275 GB    │  Queued: 3                   │
│  Temp: 60°C          │  Tokens/s: 1,247             │
├──────────────────────┴──────────────────────────────┤
│  Polling 1s │ Uptime: 342s │ Ctrl+C to exit         │
└─────────────────────────────────────────────────────┘
```

Implementation:
1. `parse_prometheus_text(text)` — regex parse Prometheus format → dict
2. `get_gpu_bars()` — nvidia-smi → list of dicts
3. `build_dashboard(endpoint, gpus, prom, iter_stats, elapsed)` — builds `rich.layout.Layout`
4. `run_visualizer(endpoint, poll_interval)` — `Live` context, poll every 1s, update dashboard
5. Color coding: green < 80%, yellow 80-95%, red > 95% for utilization/memory/KV cache

### Verify
- [ ] `python -m examples.apps.trtllm_ops viz --endpoint http://localhost:8000` shows live dashboard
- [ ] Values update every second
- [ ] Colors change when KV cache / GPU util crosses thresholds

---

## Task 4: Entry Points + README (15 min)

**Create:** `__main__.py`, `README.md`

`__main__.py`: routes `python -m examples.apps.trtllm_ops` to cli, `python -m examples.apps.trtllm_ops viz` to visualizer.

README: usage examples for both tools, tool table, requirements (`pip install openai httpx rich`).

### Verify
- [ ] Both entry points work

---

## Task 5: Integration Demo (45 min)

### Setup

```bash
# Terminal 1: server
docker run --rm --gpus '"device=0"' --ipc=host -p 8000:8000 \
  nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc7 \
  trtllm-serve <model> --backend pytorch --port 8000
```

### Demo Script

```bash
# Terminal 2: generate load
python -c "
from openai import OpenAI
import concurrent.futures
client = OpenAI(base_url='http://localhost:8000/v1', api_key='x')
def send():
    return client.chat.completions.create(
        model='<model>',
        messages=[{'role':'user','content':'Write a poem about GPUs'}],
        max_tokens=100)
with concurrent.futures.ThreadPoolExecutor(max_workers=32) as pool:
    futs = [pool.submit(send) for _ in range(500)]
    for f in concurrent.futures.as_completed(futs): print('.', end='', flush=True)
"

# Terminal 3: visualizer
python -m examples.apps.trtllm_ops viz --endpoint http://localhost:8000

# Terminal 4: ops agent (THE MONEY SHOT)
python -m examples.apps.trtllm_ops --endpoint http://localhost:8000
```

### Demo Questions
1. "How's the server doing?" → reads metrics, summarizes
2. "Is the GPU saturated?" → checks utilization + queue depth
3. "Can I handle 100 more concurrent users?" → estimates capacity, shows math
4. "Why is latency increasing?" → checks KV cache util, queue depth, identifies bottleneck
5. "What config change would help?" → suggests KV cache FP8, batch size adjustment

---

## Timeline

| Time | Task | Output |
|------|------|--------|
| 0:00-0:30 | Task 1: tools + system prompt | Working tool functions |
| 0:30-1:00 | Task 1b: log_manager.py | Ring buffer, log capture, smart grep |
| 1:00-1:30 | Task 2: CLI chat loop | `trtllm-ops` chats with server |
| 1:30-2:10 | Task 3: Visualizer TUI | `trtllm-top` live dashboard |
| 2:10-2:20 | Task 4: Entry points + README | Clean package |
| 2:20-2:50 | Task 5: Integration + demo prep | Live demo working |
| 2:50-3:00 | Buffer / polish | |

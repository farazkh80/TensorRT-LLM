# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Intelligent log capture, indexing, and search for TRT-LLM server logs."""

import json
import os
import re
import subprocess
import threading
import time
from collections import deque
from datetime import datetime


# Known TRT-LLM log patterns for structured classification
LOG_PATTERNS = {
    'oom': re.compile(
        r'(OutOfMemoryError|CUDA out of memory|OOM|oom-kill)', re.IGNORECASE),
    'kv_eviction': re.compile(
        r'(evict|eviction|cache.*(full|pressure))', re.IGNORECASE),
    'cuda_error': re.compile(
        r'(CUDA error|cudaError|cuda.*failed)', re.IGNORECASE),
    'timeout': re.compile(
        r'(timeout|timed out|deadline exceeded)', re.IGNORECASE),
    'nccl': re.compile(
        r'(NCCL|nccl.*(error|timeout|failed))', re.IGNORECASE),
    'request_error': re.compile(
        r'(request.*failed|error.*request|500|Internal Server Error)',
        re.IGNORECASE),
    'loading': re.compile(
        r'(loading.*model|shard|weight|checkpoint)', re.IGNORECASE),
    'ready': re.compile(
        r'(ready|started|listening|health.*ok)', re.IGNORECASE),
}

# TRT-LLM log line format: [TensorRT-LLM][LEVEL] message  or  standard python logging
LOG_LINE_RE = re.compile(
    r'(?:\[(?:TensorRT-LLM|TRT-LLM)\]\s*)?'
    r'(?:\[([A-Z]+)\]\s*)?'  # optional [LEVEL]
    r'(?:(\d{4}-\d{2}-\d{2}[\sT]\d{2}:\d{2}:\d{2}(?:\.\d+)?)\s*)?'  # optional timestamp
    r'(.*)')  # message


class LogManager:
    """Captures and indexes TRT-LLM server logs for intelligent querying."""

    def __init__(self, log_source="auto", log_file=None, container_id=None,
                 log_dir="/tmp/trtllm-ops-logs", endpoint=None):
        self.log_source = log_source
        self.log_file = log_file
        self.container_id = container_id
        self.log_dir = log_dir
        self.endpoint = endpoint

        self.ring_buffer = deque(maxlen=10000)
        self.error_index = []  # [(unix_ts, level, message, categories)]
        self.capture_method = "none"
        self._running = False
        self._thread = None
        self._start_time = time.time()

        os.makedirs(log_dir, exist_ok=True)

    def start(self):
        """Start log capture in background thread."""
        if self._running:
            return

        if self.log_source == "auto":
            self.capture_method = self._detect_source()
        else:
            self.capture_method = self.log_source

        self._running = True

        if self.capture_method == "docker":
            self._thread = threading.Thread(
                target=self._capture_docker, daemon=True)
        elif self.capture_method in ("file", "procfd"):
            self._thread = threading.Thread(
                target=self._capture_file, daemon=True)
        elif self.capture_method == "prometheus":
            self._thread = threading.Thread(
                target=self._capture_prometheus, daemon=True)
        else:
            # Last resort: scrape server logs via run_shell_command
            # Pre-load recent logs from common locations
            self._try_bootstrap_logs()
            return

        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)

    def _detect_source(self) -> str:
        if self.container_id:
            return "docker"
        if self.log_file and os.path.exists(self.log_file):
            return "file"
        # Try to find trtllm-serve process and read its /proc/pid/fd/2
        try:
            out = subprocess.check_output(
                ['pgrep', '-f', 'trtllm-serve|trtllm.commands.serve'],
                text=True, timeout=5).strip()
            if out:
                pid = out.split('\n')[0].strip()
                stderr_path = f"/proc/{pid}/fd/2"
                if os.path.exists(stderr_path):
                    self.log_file = stderr_path
                    return "procfd"
        except Exception:
            pass
        # Try to find a docker container running trtllm-serve
        try:
            out = subprocess.check_output(
                ['docker', 'ps', '--filter', 'ancestor=nvcr.io/nvidia/tensorrt-llm',
                 '--format', '{{.ID}}'],
                text=True, timeout=5).strip()
            if out:
                self.container_id = out.split('\n')[0]
                return "docker"
        except Exception:
            pass
        # Try common log paths
        for path in ['/tmp/trtllm-serve.log', '/var/log/trtllm-serve.log']:
            if os.path.exists(path):
                self.log_file = path
                return "file"
        if self.endpoint:
            return "prometheus"
        return "none"

    def _ingest_line(self, raw_line: str):
        """Parse and index a single log line."""
        raw_line = raw_line.rstrip('\n\r')
        if not raw_line:
            return

        ts = time.time()
        level = "INFO"
        message = raw_line

        m = LOG_LINE_RE.match(raw_line)
        if m:
            if m.group(1):
                level = m.group(1).upper()
            if m.group(2):
                try:
                    dt = datetime.fromisoformat(m.group(2).replace(' ', 'T'))
                    ts = dt.timestamp()
                except ValueError:
                    pass
            message = m.group(3) or raw_line

        # Classify
        categories = []
        for cat_name, pattern in LOG_PATTERNS.items():
            if pattern.search(raw_line):
                categories.append(cat_name)

        entry = {
            'timestamp': ts,
            'level': level,
            'message': message,
            'raw': raw_line,
            'categories': categories,
        }

        self.ring_buffer.append(entry)

        # Index errors and warnings for fast lookup
        if level in ('ERROR', 'WARNING', 'CRITICAL', 'FATAL') or categories:
            self.error_index.append((ts, level, message, categories))

        # Write to disk
        self._write_to_disk(raw_line)

    def _write_to_disk(self, line: str):
        log_path = os.path.join(self.log_dir, "server.log")
        try:
            with open(log_path, 'a') as f:
                f.write(line + '\n')
        except OSError:
            pass

    # --- Capture methods ---

    def _capture_docker(self):
        try:
            proc = subprocess.Popen(
                ['docker', 'logs', '-f', '--since',
                 datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S'),
                 self.container_id],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            while self._running:
                line = proc.stdout.readline()
                if not line:
                    time.sleep(0.1)
                    continue
                self._ingest_line(line)
            proc.kill()
        except Exception:
            self.capture_method = "docker (failed)"

    def _capture_file(self):
        try:
            with open(self.log_file, 'r') as f:
                # First: load last 500 lines of existing content
                f.seek(0, 2)
                end_pos = f.tell()
                # Read up to 500KB from the end to get recent history
                read_size = min(end_pos, 500 * 1024)
                f.seek(end_pos - read_size)
                if read_size < end_pos:
                    f.readline()  # skip partial first line
                for line in f:
                    self._ingest_line(line)

                # Then: tail for new lines
                while self._running:
                    line = f.readline()
                    if line:
                        self._ingest_line(line)
                    else:
                        time.sleep(0.2)
        except Exception:
            self.capture_method = "file (failed)"

    def _try_bootstrap_logs(self):
        """Try to grab recent logs from the server process."""
        # Try to find trtllm-serve worker PID and read from /proc
        try:
            out = subprocess.check_output(
                ['bash', '-c',
                 'pgrep -f "trtllm-serve|trtllm.commands.serve|rpc_server" '
                 '| head -5'],
                text=True, timeout=5).strip()
            if out:
                for pid in out.split('\n'):
                    pid = pid.strip()
                    # Try to read from /proc/pid/fd/1 (stdout) and /2 (stderr)
                    for fd in ['1', '2']:
                        path = f"/proc/{pid}/fd/{fd}"
                        try:
                            # Can't reliably tail a proc fd, but we can try
                            result = subprocess.check_output(
                                ['timeout', '1', 'cat', path],
                                text=True, timeout=3,
                                stderr=subprocess.DEVNULL)
                            for line in result.split('\n'):
                                self._ingest_line(line)
                        except Exception:
                            pass
        except Exception:
            pass

        # Also try reading from any .log files in /tmp
        try:
            import glob
            for log_file in glob.glob('/tmp/*.log') + glob.glob('/tmp/trtllm*'):
                try:
                    with open(log_file) as f:
                        for line in f.readlines()[-200:]:
                            self._ingest_line(line)
                except Exception:
                    pass
        except Exception:
            pass

        if self.ring_buffer:
            self.capture_method = f"bootstrap ({len(self.ring_buffer)} lines)"

    def _capture_prometheus(self):
        """Fallback: poll prometheus metrics for error count changes."""
        import httpx
        last_errors = 0
        while self._running:
            try:
                resp = httpx.get(
                    f"{self.endpoint}/prometheus/metrics", timeout=3)
                if resp.status_code == 200:
                    for line in resp.text.split('\n'):
                        if 'error' in line.lower() and not line.startswith('#'):
                            self._ingest_line(
                                f"[PROMETHEUS] {line.strip()}")
            except Exception:
                pass
            time.sleep(2)

    # --- Query methods ---

    def search_logs(self, pattern=None, level=None, since=None,
                    limit=50) -> list[dict]:
        """Search logs with optional filters.

        Args:
            pattern: regex pattern to match against raw log lines
            level: filter by level (ERROR, WARNING, INFO, etc.)
            since: ISO timestamp string or unix timestamp — only return logs after this
            limit: max results to return
        """
        since_ts = None
        if since:
            if isinstance(since, (int, float)):
                since_ts = since
            else:
                try:
                    since_ts = datetime.fromisoformat(since).timestamp()
                except ValueError:
                    pass

        compiled = re.compile(pattern, re.IGNORECASE) if pattern else None
        results = []

        # Search backwards (newest first)
        for entry in reversed(self.ring_buffer):
            if level and entry['level'] != level.upper():
                continue
            if since_ts and entry['timestamp'] < since_ts:
                continue
            if compiled and not compiled.search(entry['raw']):
                continue
            results.append({
                'time': datetime.fromtimestamp(
                    entry['timestamp']).strftime('%H:%M:%S.%f')[:-3],
                'level': entry['level'],
                'message': entry['message'][:500],
                'categories': entry['categories'],
            })
            if len(results) >= limit:
                break

        results.reverse()
        return results

    def get_log_summary(self) -> dict:
        """Get a quick health summary from captured logs."""
        now = time.time()
        return {
            'errors_last_1m': sum(
                1 for ts, lvl, _, _ in self.error_index
                if now - ts < 60 and lvl in ('ERROR', 'CRITICAL', 'FATAL')),
            'errors_last_5m': sum(
                1 for ts, lvl, _, _ in self.error_index
                if now - ts < 300 and lvl in ('ERROR', 'CRITICAL', 'FATAL')),
            'warnings_last_5m': sum(
                1 for ts, lvl, _, _ in self.error_index
                if now - ts < 300 and lvl == 'WARNING'),
            'last_errors': [
                {
                    'time': datetime.fromtimestamp(ts).strftime('%H:%M:%S'),
                    'level': lvl,
                    'message': msg[:200],
                    'categories': cats,
                }
                for ts, lvl, msg, cats in self.error_index[-5:]
            ],
            'buffer_size': len(self.ring_buffer),
            'total_lines_captured': len(self.ring_buffer),
            'capture_method': self.capture_method,
            'capture_uptime_secs': round(time.time() - self._start_time),
        }

    def get_log_context(self, timestamp: str, window_secs: float = 10,
                        limit: int = 100) -> list[dict]:
        """Get all log lines within ±window_secs of a timestamp."""
        try:
            center_ts = datetime.fromisoformat(timestamp).timestamp()
        except ValueError:
            try:
                # Try HH:MM:SS format (assume today)
                today = datetime.now().strftime('%Y-%m-%d')
                center_ts = datetime.fromisoformat(
                    f"{today}T{timestamp}").timestamp()
            except ValueError:
                return [{"error": f"Cannot parse timestamp: {timestamp}"}]

        results = []
        for entry in self.ring_buffer:
            if abs(entry['timestamp'] - center_ts) <= window_secs:
                results.append({
                    'time': datetime.fromtimestamp(
                        entry['timestamp']).strftime('%H:%M:%S.%f')[:-3],
                    'level': entry['level'],
                    'message': entry['message'][:500],
                    'categories': entry['categories'],
                })
                if len(results) >= limit:
                    break

        return results

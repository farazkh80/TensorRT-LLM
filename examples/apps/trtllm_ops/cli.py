# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""trtllm-ops: Chat with your running TRT-LLM server about its own performance."""

import argparse
import json
import re
import sys

from openai import OpenAI


def _clean_output(text: str) -> str:
    """Strip base-model chat template artifacts and internal reasoning."""
    # Remove <|channel|>, <|end|>, <|return|>, etc.
    text = re.sub(r'<\|[^|]*\|>', '', text)
    # Remove internal reasoning/commentary blocks
    text = re.sub(r'\banalysis\b[^{]*(?=\{|$)', '', text)
    text = re.sub(r'\bcommentary\s*\{[^}]*\}', '', text)
    text = re.sub(r'\bfunctions\.\w+', '', text)
    # Remove "final" or "assistant" channel labels
    text = re.sub(r'\b(final|assistant)\b(?=\s*[A-Z*#])', '', text)
    # Remove ALL leaked JSON blobs — any { ... } that looks like tool
    # args or fake tool results (contains quoted keys like "pattern",
    # "error_counts", "warning_counts", "gpus", "healthy", etc.)
    text = re.sub(
        r'\{[^}]*"(?:pattern|level|limit|command|n_lines|avg_seq_len|'
        r'timestamp|window_secs|error_counts|warning_counts|last_errors|'
        r'healthy|gpus|last_1m|last_5m|lines|matches|count)"[^}]*\}',
        '', text, flags=re.DOTALL)
    # Remove bare "Need ..." reasoning lines
    text = re.sub(r'^Need\s+.*$', '', text, flags=re.MULTILINE)
    # If what's left is mostly JSON/garbage (starts with , or { or }),
    # treat as empty
    cleaned = text.strip().lstrip(',').strip()
    if cleaned and cleaned[0] in '{[' and not any(
            c in cleaned for c in ['#', '*', '|', 'GPU', 'Server',
                                    'The ', 'No ', 'Yes']):
        return ""
    return cleaned


def _is_garbage(text: str) -> bool:
    """Check if text is leaked chain-of-thought rather than real answer."""
    if not text:
        return True
    # Mostly JSON
    json_chars = sum(1 for c in text if c in '{}[]":,')
    if json_chars > len(text) * 0.4:
        return True
    # Starts with comma or brace
    stripped = text.strip().lstrip(',').strip()
    if stripped and stripped[0] in '{[':
        return True
    # Too short and no natural language
    if len(text) < 20 and not any(c.isalpha() for c in text):
        return True
    return False

from .log_manager import LogManager
from .system_prompt import build_system_prompt
from .tools import TOOL_DEFINITIONS, execute_tool, get_gpu_status


def get_model_name(client: OpenAI) -> str:
    """Auto-detect model name from the running server."""
    try:
        models = client.models.list()
        if models.data:
            return models.data[0].id
    except Exception:
        pass
    return "unknown-model"


def chat_loop(client: OpenAI, model: str, endpoint: str,
              system_prompt: str, log_manager: LogManager):
    """Main REPL loop with tool calling."""
    messages = [{"role": "system", "content": system_prompt}]

    print(f"\n\033[1;36m{'='*60}\033[0m")
    print(f"\033[1;36m  trtllm-ops\033[0m — AI Operations Agent")
    print(f"  Server: {endpoint}")
    print(f"  Model:  {model}")
    print(f"  Logs:   {log_manager.capture_method}")
    print(f"\033[1;36m{'='*60}\033[0m")
    print("  Type your question, or 'quit' to exit.\n")

    while True:
        try:
            user_input = input("\033[1;32mYou>\033[0m ")
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if user_input.strip().lower() in ('quit', 'exit', 'q'):
            break
        if not user_input.strip():
            continue

        # Keep conversation history manageable — retain system prompt
        # + last 6 exchanges (12 messages) to stay under token limits
        MAX_HISTORY = 12
        if len(messages) > 1 + MAX_HISTORY:
            messages = messages[:1] + messages[-(MAX_HISTORY):]

        # Snapshot message count so we can rollback on failure
        msg_snapshot = len(messages)
        messages.append({"role": "user", "content": user_input})

        # Tool calling loop
        max_rounds = 5
        success = False
        for round_num in range(max_rounds):
            try:
                # Sanitize: ensure no None content in any message
                for m in messages:
                    if m.get("content") is None:
                        m["content"] = ""

                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    tools=TOOL_DEFINITIONS,
                    tool_choice="auto",
                    temperature=0.1,
                    max_tokens=1024,
                )
            except Exception as e:
                err_msg = str(e)[:200]
                print(f"\033[1;31mError:\033[0m {err_msg}")
                break

            choice = response.choices[0]
            msg = choice.message

            if msg.tool_calls:
                messages.append({
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments or "{}",
                            }
                        }
                        for tc in msg.tool_calls
                    ],
                })

                for tc in msg.tool_calls:
                    fn_name = tc.function.name
                    fn_args = json.loads(tc.function.arguments or '{}')
                    print(f"  \033[1;33m[{fn_name}]\033[0m", end="", flush=True)

                    result = execute_tool(
                        fn_name, fn_args, endpoint,
                        log_manager=log_manager)

                    result_preview = result[:80].replace('\n', ' ')
                    print(f" \033[2m{result_preview}...\033[0m")

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result or "OK",
                    })
                continue
            else:
                content = _clean_output(msg.content or "")
                if (not content.strip() or _is_garbage(content)) \
                        and round_num > 0:
                    # Nudge: re-ask without tools to force text
                    try:
                        nudge = client.chat.completions.create(
                            model=model,
                            messages=messages + [{
                                "role": "user",
                                "content": "Based on the tool results above, "
                                           "give a concise status report.",
                            }],
                            temperature=0.1,
                            max_tokens=1024,
                        )
                        content = _clean_output(
                            nudge.choices[0].message.content or "")
                    except Exception:
                        pass
                if content:
                    print(f"\n\033[1;36mOps>\033[0m {content}\n")
                    messages.append({
                        "role": "assistant",
                        "content": content,
                    })
                    success = True
                else:
                    print("\n\033[1;36mOps>\033[0m "
                          "(model returned empty — try rephrasing)\n")
                break

        if not success:
            # Rollback messages to before this question
            del messages[msg_snapshot:]


def main():
    parser = argparse.ArgumentParser(
        description="Chat with your running TRT-LLM server about its "
                    "own performance")
    parser.add_argument(
        '--endpoint', default='http://localhost:8000',
        help='TRT-LLM server endpoint (default: http://localhost:8000)')
    parser.add_argument(
        '--model', default=None,
        help='Model name (auto-detected from server if not set)')
    parser.add_argument(
        '--log-source', default='auto',
        choices=['auto', 'docker', 'file', 'prometheus'],
        help='How to capture server logs (default: auto-detect)')
    parser.add_argument(
        '--log-file', default=None,
        help='Path to server log file (for --log-source file)')
    parser.add_argument(
        '--container-id', default=None,
        help='Docker container ID (for --log-source docker)')
    args = parser.parse_args()

    # Connect to server
    client = OpenAI(base_url=f"{args.endpoint}/v1", api_key="unused")

    # Auto-detect model
    model = args.model or get_model_name(client)

    # Start log capture
    log_manager = LogManager(
        log_source=args.log_source,
        log_file=args.log_file,
        container_id=args.container_id,
        endpoint=args.endpoint,
    )
    log_manager.start()

    # Build system prompt
    gpu_info = get_gpu_status()
    system_prompt = build_system_prompt(
        model_name=model,
        gpu_info=gpu_info,
        endpoint=args.endpoint,
    )

    try:
        chat_loop(client, model, args.endpoint, system_prompt, log_manager)
    finally:
        log_manager.stop()


if __name__ == "__main__":
    main()

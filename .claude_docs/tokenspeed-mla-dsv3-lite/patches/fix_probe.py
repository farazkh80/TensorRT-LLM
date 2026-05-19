#!/usr/bin/env python3
"""Replace the malformed stderr probe with a clean one + remove duplicate."""
F = "/usr/local/lib/python3.12/dist-packages/tensorrt_llm/_torch/attention_backend/trtllm_gen.py"
s = open(F).read()

# Find lines starting with "        import sys as _spike_sys" (the bad probe)
lines = s.split("\n")
out = []
i = 0
while i < len(lines):
    if "_spike_sys" in lines[i]:
        # drop this line and the SPIKE PROBE marker that follows
        i += 1
        if i < len(lines) and "SPIKE PROBE" in lines[i] and "stderr print" not in lines[i]:
            i += 1
        continue
    out.append(lines[i])
    i += 1
s = "\n".join(out)

# Now inject a clean probe right before "# SPIKE PATCH: TLLM_TOKENSPEED_MLA=1"
needle = "        # SPIKE PATCH: TLLM_TOKENSPEED_MLA=1 swaps in tokenspeed-mla decode kernel."
probe = (
    '        # SPIKE PROBE: stderr print to confirm this site runs\n'
    '        import sys as _ssys\n'
    '        _ssys.stderr.write("[SPIKE] run_mla_generation hit; env=" + str(os.environ.get("TLLM_TOKENSPEED_MLA")) + "\\n")\n'
    '        _ssys.stderr.flush()\n'
)
if probe.strip() not in s:
    s = s.replace(needle, probe + needle, 1)
open(F, "w").write(s)

# Verify
import subprocess
r = subprocess.run(["python3", "-m", "py_compile", F], capture_output=True, text=True)
print("py_compile rc:", r.returncode)
if r.returncode != 0:
    print(r.stderr)

# Print surrounding context
new = open(F).read().split("\n")
for j, line in enumerate(new):
    if "SPIKE" in line:
        print(f"  line {j+1}: {line}")

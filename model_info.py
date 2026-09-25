"""Print a GGUF model's MoE layout. Works on split models: pass the first part."""
import sys
from pathlib import Path

import gguf

first = Path(sys.argv[1])
r = gguf.GGUFReader(first)
arch = bytes(r.fields["general.architecture"].parts[-1]).decode()
for k in r.fields:
    if k.startswith(arch + ".") and any(s in k for s in ("block_count", "expert", "embedding_length")):
        print(f"{k:45s} {r.fields[k].parts[-1][0]}")

parts = sorted(first.parent.glob(first.name.replace("00001-of", "*-of"))) if "00001-of" in first.name else [first]
expert = other = 0
for p in parts:
    try:
        for t in gguf.GGUFReader(p).tensors:
            if "_exps." in t.name:
                expert += int(t.n_bytes)
            else:
                other += int(t.n_bytes)
    except Exception as e:  # part still downloading
        print(f"skipped {p.name}: {e.__class__.__name__}")
print(f"routed experts {expert/1e9:.1f} GB | always-needed {other/1e9:.1f} GB")

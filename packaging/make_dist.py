"""Assemble and zip one stowaway download: python packaging/make_dist.py <name> <engine dir> [<compat engine dir>]

<engine dir> holds llama-cli and llama-server (the .exe on Windows); the launcher is dist/stowaway[.exe] (PyInstaller).
Writes stowaway-<name>/ and stowaway-<name>.zip next to it. Unix file modes survive the zip, so the programs stay
executable after unzipping on Mac and Linux.
"""

import os
import shutil
import sys
from pathlib import Path

name, engine = sys.argv[1], Path(sys.argv[2])
compat = Path(sys.argv[3]) if len(sys.argv) > 3 else None
exe = ".exe" if os.name == "nt" else ""
root = Path(__file__).resolve().parent.parent
out = Path(f"stowaway-{name}")
shutil.rmtree(out, ignore_errors=True)
out.mkdir()
shutil.copy2(Path("dist") / f"stowaway{exe}", out)
for tool in ("llama-cli", "llama-server"):
    shutil.copy2(engine / f"{tool}{exe}", out)
if compat:
    (out / "compat").mkdir()
    for tool in ("llama-cli", "llama-server"):
        shutil.copy2(compat / f"{tool}{exe}", out / "compat")
newline = "\r\n" if name == "windows" else "\n"
readme = (root / "packaging" / "README.txt").read_text().replace("\n", newline)
(out / "README.txt").write_text(readme, newline="")
shutil.copy2(root / "LICENSE", out / "LICENSE.txt")
shutil.copy2(root / "llama.cpp" / "LICENSE", out / "LICENSE-llama.cpp.txt")
for f in out.rglob("*"):
    if f.is_file() and f.suffix not in (".txt",):
        f.chmod(0o755)
shutil.make_archive(str(out), "zip", ".", out.name)
print(f"wrote {out}.zip")

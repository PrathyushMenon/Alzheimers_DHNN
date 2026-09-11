from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "run_pipeline.sh"
    if not script.exists():
        raise SystemExit(f"Pipeline wrapper not found: {script}")
    if os.name == "nt":
        wsl_script = "/mnt/" + str(script.drive[0]).lower() + str(script).replace(script.drive, "").replace("\\", "/")
        cmd = ["wsl.exe", "bash", wsl_script]
    else:
        cmd = ["bash", str(script)]
    print("Running exact reproduction pipeline:")
    print(" ".join(cmd))
    return subprocess.call(cmd, cwd=str(root))


if __name__ == "__main__":
    raise SystemExit(main())

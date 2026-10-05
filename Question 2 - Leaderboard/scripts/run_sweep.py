"""Run named configurations sequentially; stop on failure and preserve earlier results."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", nargs="+", default=[f"configs/{name}.yaml" for name in
                        ("a_log_target", "b_log_external", "c_raw_target", "d_raw_external")])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--origins", nargs="+", type=int)
    parser.add_argument("--epochs", type=int)
    args = parser.parse_args()
    if len({Path(p).stem for p in args.configs}) != len(args.configs):
        parser.error("config filenames must have distinct stems")
    for config in args.configs:
        command = [sys.executable, "-u", "-m", "scripts.run_experiment", "--config", config,
                   "--mode", "validate", "--device", args.device]
        for name in ("seeds", "origins", "epochs"):
            value = getattr(args, name)
            if value is not None:
                command += [f"--{name}", *map(str, value if isinstance(value, list) else [value])]
        print(f"Running {config}", flush=True)
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()

from __future__ import annotations

import subprocess
import sys


SCRIPTS = [
    "scripts.run_data_validation",
    "scripts.run_training",
    "scripts.run_evaluation",
    "scripts.run_stream_simulation",
]


def main() -> None:
    for module in SCRIPTS:
        print(f"Running {module}...")
        result = subprocess.run([sys.executable, "-m", module], check=False)
        if result.returncode != 0:
            raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()

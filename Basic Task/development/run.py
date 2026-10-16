#!/usr/bin/env python3
"""项目入口：在 WSL 中运行。

    python3 run.py tasks
    python3 run.py serve baseline
    python3 run.py bench baseline bench_c1
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "src"))

from antlab.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())

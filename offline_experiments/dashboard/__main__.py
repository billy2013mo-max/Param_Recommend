"""CLI entry point: python -m dashboard."""

from __future__ import annotations

import argparse
import os

import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the read-only live experiment dashboard")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8501)
    parser.add_argument("--scan-interval", type=float, default=2.0)
    args = parser.parse_args()
    os.environ["DASHBOARD_SCAN_INTERVAL"] = str(args.scan_interval)
    uvicorn.run(
        "dashboard.app:app",
        host=args.host,
        port=args.port,
        env_file=None,
        log_level="info",
    )


if __name__ == "__main__":
    main()


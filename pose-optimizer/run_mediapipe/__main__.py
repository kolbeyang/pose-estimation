"""MediaPipe pipeline entrypoint.

Usage:
    uv run python -m run_mediapipe config.json
    uv run python -m run_mediapipe run_mediapipe/run_single_config.json
"""

import logging
import sys

from config import load_config
from run_mediapipe import run_pipeline


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    if len(sys.argv) < 2:
        print("Usage: uv run python -m run_mediapipe <config.json>")
        sys.exit(1)

    config = load_config(sys.argv[1])
    run_pipeline(config)


if __name__ == "__main__":
    main()

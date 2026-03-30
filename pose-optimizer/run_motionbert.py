"""MotionBERT pipeline entrypoint.

Usage:
    uv run python run_motionbert.py config.json
    uv run python run_motionbert.py run_motionbert/run_single_config.json
"""

import sys

from config import load_config
from run_motionbert import run_pipeline


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: uv run python run_motionbert.py <config.json>")
        sys.exit(1)

    config = load_config(sys.argv[1])
    run_pipeline(config)


if __name__ == "__main__":
    main()

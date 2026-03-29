"""MediaPipe pipeline entrypoint.

Usage:
    uv run python mediapipe.py config.json
    uv run python mediapipe.py mediapipe/run_single_config.json
"""

import sys

from config import load_config

# Import must use the full module path to avoid conflict with the mediapipe package
import mediapipe as mediapipe_pipeline


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: uv run python mediapipe.py <config.json>")
        sys.exit(1)

    config = load_config(sys.argv[1])
    mediapipe_pipeline.run_pipeline(config)


if __name__ == "__main__":
    main()

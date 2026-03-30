"""Download external model repos and pretrained weights for MotionBERT pipeline.

Stacked Hourglass: installed via pip (pytorch-stacked-hourglass package).
MotionBERT: cloned from GitHub, checkpoint from HuggingFace.
"""

import os
import subprocess
import urllib.request

SCRIPT_DIR: str = os.path.dirname(os.path.abspath(__file__))
EXTERNAL_DIR: str = os.path.join(SCRIPT_DIR, "external")
CHECKPOINTS_DIR: str = os.path.join(SCRIPT_DIR, "checkpoints")


def clone_repo(url: str, target_dir: str) -> None:
    """Clone a git repo if not already present."""
    if os.path.exists(target_dir):
        print(f"  Already exists: {target_dir}")
        return
    print(f"  Cloning {url} -> {target_dir}")
    subprocess.run(["git", "clone", "--depth", "1", url, target_dir], check=True)


def download_file(url: str, target_path: str) -> None:
    """Download a file if not already present."""
    if os.path.exists(target_path):
        print(f"  Already exists: {target_path}")
        return
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    print(f"  Downloading {url}")
    print(f"    -> {target_path}")
    urllib.request.urlretrieve(url, target_path)
    print(f"    Done ({os.path.getsize(target_path) / 1e6:.1f} MB)")


def ensure_models() -> None:
    """Download all required models and checkpoints if not already present."""
    os.makedirs(EXTERNAL_DIR, exist_ok=True)
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

    # 1. Clone MotionBERT
    print("\n=== MotionBERT ===")
    motionbert_dir = os.path.join(EXTERNAL_DIR, "MotionBERT")
    clone_repo(
        "https://github.com/Walter0807/MotionBERT.git",
        motionbert_dir,
    )

    # 2. Download MotionBERT-Lite H36M checkpoint
    motionbert_ckpt = os.path.join(CHECKPOINTS_DIR, "motionbert_lite_h36m.bin")
    hf_url = (
        "https://huggingface.co/walterzhu/MotionBERT/resolve/main/"
        "checkpoint/pose3d/FT_MB_lite_MB_ft_h36m_global_lite/best_epoch.bin"
    )
    try:
        download_file(hf_url, motionbert_ckpt)
    except Exception as e:
        print(f"  Auto-download failed: {e}")
        print("  Please download manually from:")
        print("  https://huggingface.co/walterzhu/MotionBERT/tree/main")
        print(f"  Place at: {motionbert_ckpt}")

    # Check for existing models in motionbert-pose (symlink for convenience)
    existing_external = os.path.normpath(
        os.path.join(SCRIPT_DIR, "..", "..", "motionbert-pose", "external", "MotionBERT")
    )
    existing_ckpt = os.path.normpath(
        os.path.join(SCRIPT_DIR, "..", "..", "motionbert-pose", "checkpoints", "motionbert_lite_h36m.bin")
    )
    if not os.path.exists(motionbert_dir) and os.path.exists(existing_external):
        print(f"  Symlinking from existing: {existing_external}")
        os.symlink(existing_external, motionbert_dir)
    if not os.path.exists(motionbert_ckpt) and os.path.exists(existing_ckpt):
        print(f"  Symlinking from existing: {existing_ckpt}")
        os.symlink(existing_ckpt, motionbert_ckpt)


def main() -> None:
    """Download all required models and checkpoints."""
    print("=== Stacked Hourglass ===")
    print("  Installed via pip (pytorch-stacked-hourglass).")
    print("  Pretrained weights auto-download on first use.")
    ensure_models()
    print("\n=== Setup Complete ===")
    print(f"External repos: {EXTERNAL_DIR}")
    print(f"Checkpoints: {CHECKPOINTS_DIR}")


if __name__ == "__main__":
    main()

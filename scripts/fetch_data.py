"""Download LayerGuard-RAG data from HuggingFace Hub.

Run this once after cloning the repo to populate data/:
    python scripts/fetch_data.py

If files already exist, HF's local cache avoids re-downloading unchanged files.
For private repos, authenticate first:
    huggingface-cli login
or set the HF_TOKEN environment variable.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from huggingface_hub import snapshot_download

from config import DATA_DIR, DATA_REPO_ID


def main():
    print(f"Fetching data from {DATA_REPO_ID}...")
    snapshot_download(
        repo_id=DATA_REPO_ID,
        repo_type="dataset",
        local_dir=DATA_DIR,
    )
    print(f"Data downloaded to {DATA_DIR}/")
    print("Next steps (if not already done):")
    print("  python scripts/train_stage1.py ...")
    print("  python scripts/tune_thresholds.py")


if __name__ == "__main__":
    main()

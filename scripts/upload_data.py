"""Upload LayerGuard-RAG data to HuggingFace Hub (one-time, run by data owner).

Requires a HF write token:
    huggingface-cli login
or set the HF_TOKEN environment variable.

Usage:
    python scripts/upload_data.py
    python scripts/upload_data.py --repo michchicken/layerguard-nq
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from huggingface_hub import HfApi, create_repo

from config import DATA_DIR, DATA_REPO_ID


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=DATA_REPO_ID, help="HF dataset repo id")
    parser.add_argument("--private", action="store_true", help="Make repo private")
    args = parser.parse_args()

    api = HfApi()

    create_repo(args.repo, repo_type="dataset", private=args.private, exist_ok=True)
    print(f"Uploading data/ to {args.repo} (this may take a few minutes)...")

    api.upload_folder(
        folder_path=DATA_DIR,
        repo_id=args.repo,
        repo_type="dataset",
        ignore_patterns=["*.pyc", "__pycache__"],
    )
    print(f"Done. Dataset available at: https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()

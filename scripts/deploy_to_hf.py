#!/usr/bin/env python3
"""Deploy DepthWizard to Hugging Face Spaces using huggingface_hub."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

IGNORE_PATTERNS = [
    "venv/*",
    ".venv/*",
    "*.pyc",
    "__pycache__/*",
    ".DS_Store",
    ".git/*",
    "data/*",
    "*.h5",
    "outputs/stage12/checkpoints/checkpoint_*.pt",
    "outputs/stage12/checkpoints/experiment_A_*.pt",
    "outputs/stage12/checkpoints/smoke_test_*.pt",
    "outputs/final/runs/*",
    "tests/*",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", help="Hugging Face Space repo ID (e.g. 'username/depthwizard')")
    parser.add_argument("--token", help="Hugging Face API write token (optional if logged in via `huggingface-cli login`)")
    parser.add_argument("--private", action="store_true", help="Create space as private (default is public)")
    args = parser.parse_args()

    try:
        from huggingface_hub import HfApi, login
    except ImportError:
        print("Error: huggingface_hub is not installed in the active environment.")
        print("Install it with: ./venv/bin/python -m pip install huggingface_hub")
        sys.exit(1)

    api = HfApi(token=args.token)

    # Determine token and user
    user_info = None
    try:
        user_info = api.whoami()
        username = user_info["name"]
        print(f"Authenticated as Hugging Face user: {username}")
    except Exception as exc:
        if not args.token:
            print("Authentication required. Please provide --token or run `huggingface-cli login`.")
            print("You can get a free token with write access at: https://huggingface.co/settings/tokens")
            token_input = input("Enter your Hugging Face write token: ").strip()
            if not token_input:
                sys.exit(1)
            api = HfApi(token=token_input)
            user_info = api.whoami()
            username = user_info["name"]
        else:
            print(f"Error authenticating with provided token: {exc}")
            sys.exit(1)

    repo_id = args.repo_id
    if not repo_id:
        default_repo = f"{username}/depthwizard"
        user_choice = input(f"Enter Space name [default: {default_repo}]: ").strip()
        repo_id = user_choice if user_choice else default_repo

    if "/" not in repo_id:
        repo_id = f"{username}/{repo_id}"

    print(f"\nTarget Space: https://huggingface.co/spaces/{repo_id}")
    print("Creating/verifying Space on Hugging Face...")

    try:
        api.create_repo(
            repo_id=repo_id,
            repo_type="space",
            space_sdk="docker",
            private=args.private,
            exist_ok=True,
        )
        print("Space is ready for deployment.")
    except Exception as exc:
        print(f"Notice: {exc}")

    print("\nUploading project files to Hugging Face Spaces (this may take a couple minutes for model weights)...")
    try:
        commit_info = api.upload_folder(
            folder_path=str(ROOT),
            repo_id=repo_id,
            repo_type="space",
            ignore_patterns=IGNORE_PATTERNS,
            commit_message="Deploy DepthWizard frontend and backend",
        )
        print("\nUpload complete!")
        print(f"Live Space URL: https://huggingface.co/spaces/{repo_id}")
        print("Hugging Face will now build the Docker container and start DepthWizard.")
        print("You can track build logs directly at the link above.")
    except Exception as exc:
        print(f"Upload failed: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()

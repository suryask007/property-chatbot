"""Module 7 - Deploy the chatbot to Hugging Face Spaces (free Docker hosting).

What it does
  1. creates the Space (a Docker Space) if it doesn't exist yet
  2. stores the API keys (OpenRouter, admin token, Langfuse) as Space *secrets*
     (never inside the image or repo)
  3. uploads only the files the Dockerfile needs
Hugging Face then builds the Docker image and serves it at
https://<user>-<space>.hf.space

Needs in .env:
  HF_TOKEN            a Hugging Face token with "write" access
  OPENROUTER_API_KEY  passed to the Space as a secret
  HF_SPACE            optional, e.g. "suryask/property-chatbot"

Run:
    uv run --with huggingface_hub python deploy/deploy_hf.py
"""

import os
import re
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent.parent
# Copied from .env to the Space as secrets (only the ones that are set).
SECRETS = ["OPENROUTER_API_KEY", "ADMIN_TOKEN", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY",
           "LANGFUSE_HOST", "LANGFUSE_BASE_URL"]

FILES_TO_UPLOAD = [
    "README.md", "Dockerfile", ".dockerignore", "pyproject.toml", "uv.lock",
    "app/**", "data/properties.db", "data/chroma/**",
]


def space_url(repo_id: str) -> str:
    """'suryask/property-chatbot' -> 'https://suryask-property-chatbot.hf.space'."""
    return "https://" + re.sub(r"[^a-z0-9]+", "-", repo_id.lower()) + ".hf.space"


def main() -> None:
    load_dotenv(ROOT / ".env")
    api = HfApi(token=os.environ["HF_TOKEN"])
    user = api.whoami()["name"]
    repo_id = os.getenv("HF_SPACE") or f"{user}/property-chatbot"

    api.create_repo(repo_id, repo_type="space", space_sdk="docker", exist_ok=True)
    print(f"Space ready: https://huggingface.co/spaces/{repo_id}")

    for name in SECRETS:
        if os.getenv(name):
            api.add_space_secret(repo_id, name, os.environ[name])
            print(f"Secret {name} set")

    api.upload_folder(
        repo_id=repo_id,
        repo_type="space",
        folder_path=ROOT,
        allow_patterns=FILES_TO_UPLOAD,
        ignore_patterns=["**/__pycache__/**", "*.pyc"],
        commit_message="Deploy property chatbot",
    )
    print(f"Uploaded. Hugging Face is now building the image.\nChatbot URL: {space_url(repo_id)}")


if __name__ == "__main__":
    main()

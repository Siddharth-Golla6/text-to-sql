"""
Deploy this project to a Hugging Face Streamlit Space.

Before running:
    1. Create a free HF token at https://huggingface.co/settings/tokens
       (needs "Write" permission).
    2. hf auth login             # paste the token when prompted
    3. Make sure .env has your real GROQ_API_KEY.

Then:
    python deploy_to_hf.py              # creates <you>/text-to-sql
    python deploy_to_hf.py --name foo   # creates <you>/foo
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import dotenv_values
from huggingface_hub import HfApi, create_repo, whoami

ROOT = Path(__file__).resolve().parent

# Files/paths we never want to upload to the Space.
IGNORE = {
    ".env",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".ipynb_checkpoints",
    "data/chinook.duckdb",
    "data/chinook.sqlite",
    "data/schema_cache.json",
    "evals/last_run.json",
    "evals/last_run.md",
    ".git",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="text-to-sql",
                        help="Space name (default: text-to-sql)")
    parser.add_argument("--private", action="store_true",
                        help="Create as a private Space")
    args = parser.parse_args()

    # 1. Who are we?
    try:
        who = whoami()
    except Exception as e:
        sys.exit(
            "Not logged in to Hugging Face.\n"
            "Run:  hf auth login\n"
            f"(details: {e})"
        )
    user = who["name"]
    repo_id = f"{user}/{args.name}"
    print(f"[deploy] Hugging Face user: {user}")
    print(f"[deploy] Target Space:      {repo_id}")

    # 2. Read the Groq key from local .env.
    env = dotenv_values(ROOT / ".env")
    groq_key = env.get("GROQ_API_KEY")
    if not groq_key or groq_key.startswith("your_"):
        sys.exit("No valid GROQ_API_KEY in .env. Fill it in before deploying.")

    api = HfApi()

    # 3. Create the Space if it doesn't exist.
    # HF Spaces now gates free cpu-basic Docker/Gradio behind PRO. The free
    # alternative is Gradio on ZeroGPU (zero-a10g) — serverless, GPU on-demand,
    # fine for a text-to-SQL app that only uses CPU + an external LLM API.
    print(f"[deploy] Creating Space (if new) on ZeroGPU...")
    create_repo(
        repo_id=repo_id,
        repo_type="space",
        space_sdk="gradio",
        space_hardware="zero-a10g",
        private=args.private,
        exist_ok=True,
    )

    # 4. Push the Groq key as a Space secret (overwrites each run — safe).
    print("[deploy] Setting GROQ_API_KEY as a Space secret...")
    api.add_space_secret(
        repo_id=repo_id,
        key="GROQ_API_KEY",
        value=groq_key,
    )
    # Also propagate the model choice from .env so the Space tracks local tuning.
    groq_model = env.get("GROQ_MODEL")
    if groq_model:
        print(f"[deploy] Setting GROQ_MODEL={groq_model} as a Space variable...")
        try:
            api.add_space_variable(
                repo_id=repo_id, key="GROQ_MODEL", value=groq_model,
            )
        except Exception as e:
            print(f"[deploy] WARN: could not set GROQ_MODEL variable: {e}")

    # 5. Upload everything that isn't ignored.
    print("[deploy] Uploading files...")
    api.upload_folder(
        repo_id=repo_id,
        repo_type="space",
        folder_path=str(ROOT),
        ignore_patterns=[
            ".env",
            ".venv/**",
            "venv/**",
            "**/__pycache__/**",
            "**/*.pyc",
            ".pytest_cache/**",
            ".ipynb_checkpoints/**",
            "data/chinook.duckdb",
            "data/chinook.sqlite",
            "data/schema_cache.json",
            "evals/last_run.json",
            "evals/last_run.md",
            ".git/**",
            "deploy_to_hf.py",   # no need to ship the deploy script to the Space
            "Dockerfile",        # legacy; we deploy via gradio sdk
        ],
        commit_message="Deploy text-to-sql Streamlit Space",
    )

    url = f"https://huggingface.co/spaces/{repo_id}"
    print()
    print(f"[deploy] DONE.")
    print(f"[deploy] Space URL: {url}")
    print(f"[deploy] First build takes ~2-3 minutes (installs deps, downloads"
          " sentence-transformers model, builds DuckDB).")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Download Spider, SParc, and CoSQL datasets using Hugging Face datasets library.

This script downloads all three text-to-SQL benchmarks and converts them to the
format expected by CHISEL's evaluation pipeline.
"""

import json
import os
from pathlib import Path

from datasets import load_dataset


def download_spider(output_dir: Path):
    """Download Spider dataset from Hugging Face."""
    print("[1/3] Downloading Spider dataset...")

    # Load from Hugging Face
    spider = load_dataset("xlangai/spider")

    # Create output directory
    spider_dir = output_dir / "spider"
    spider_dir.mkdir(parents=True, exist_ok=True)

    # Save train and validation splits
    for split_name, split_data in [("train", spider["train"]), ("validation", spider["validation"])]:
        output_file = spider_dir / f"{split_name}.json" if split_name == "train" else spider_dir / "dev.json"

        # Convert to Spider JSON format
        examples = []
        for example in split_data:
            examples.append({
                "db_id": example["db_id"],
                "query": example["query"],
                "question": example["question"],
            })

        with open(output_file, "w") as f:
            json.dump(examples, f, indent=2)

        print(f"  Saved {len(examples)} examples to {output_file}")

    # Save tables.json (database schemas)
    # Note: This may need to be downloaded separately from the original Spider release
    print(f"  Spider dataset saved to {spider_dir}")
    print("  Note: Database files (.sqlite) need to be downloaded from https://yale-lily.github.io/spider")


def download_sparc(output_dir: Path):
    """Download SParc dataset from Hugging Face."""
    print("\n[2/3] Downloading SParc dataset...")

    try:
        # Load from Hugging Face
        sparc = load_dataset("xlangai/sparc")

        # Create output directory
        sparc_dir = output_dir / "sparc"
        sparc_dir.mkdir(parents=True, exist_ok=True)

        # Save train and validation splits
        for split_name, split_data in [("train", sparc["train"]), ("validation", sparc["validation"])]:
            output_file = sparc_dir / f"{split_name}.json" if split_name == "train" else sparc_dir / "dev.json"

            # Convert to SParc JSON format (maintains dialogue structure)
            examples = []
            for example in split_data:
                examples.append({
                    "database_id": example.get("database_id", example.get("db_id")),
                    "interaction": example.get("interaction", []),
                })

            with open(output_file, "w") as f:
                json.dump(examples, f, indent=2)

            print(f"  Saved {len(examples)} dialogues to {output_file}")

        print(f"  SParc dataset saved to {sparc_dir}")

    except Exception as e:
        print(f"  Warning: Could not download SParc: {e}")
        print("  You may need to download it manually from https://yale-lily.github.io/sparc")


def download_cosql(output_dir: Path):
    """Download CoSQL dataset from Hugging Face."""
    print("\n[3/3] Downloading CoSQL dataset...")

    try:
        # Load from Hugging Face
        cosql = load_dataset("xlangai/cosql")

        # Create output directory
        cosql_dir = output_dir / "cosql"
        cosql_dir.mkdir(parents=True, exist_ok=True)

        # Save train and validation splits
        for split_name, split_data in [("train", cosql["train"]), ("validation", cosql["validation"])]:
            output_file = cosql_dir / f"{split_name}.json" if split_name == "train" else cosql_dir / "dev.json"

            # Convert to CoSQL JSON format (maintains dialogue structure)
            examples = []
            for example in split_data:
                examples.append({
                    "database_id": example.get("database_id", example.get("db_id")),
                    "interaction": example.get("interaction", []),
                })

            with open(output_file, "w") as f:
                json.dump(examples, f, indent=2)

            print(f"  Saved {len(examples)} dialogues to {output_file}")

        print(f"  CoSQL dataset saved to {cosql_dir}")

    except Exception as e:
        print(f"  Warning: Could not download CoSQL: {e}")
        print("  You may need to download it manually from https://yale-lily.github.io/cosql")


def main():
    """Download all datasets."""
    # Determine output directory
    project_root = Path(__file__).parent.parent
    output_dir = project_root / "data"
    output_dir.mkdir(exist_ok=True)

    print("="*80)
    print("CHISEL Dataset Downloader")
    print("="*80)
    print(f"Output directory: {output_dir}")
    print()

    # Download each dataset
    download_spider(output_dir)
    download_sparc(output_dir)
    download_cosql(output_dir)

    print("\n" + "="*80)
    print("[OK] Dataset download complete!")
    print("="*80)
    print("\nNext steps:")
    print("1. Download database files from https://yale-lily.github.io/spider")
    print("2. Extract databases to data/spider/database/")
    print("3. Run evaluation: python evaluation/evaluate.py")


if __name__ == "__main__":
    main()

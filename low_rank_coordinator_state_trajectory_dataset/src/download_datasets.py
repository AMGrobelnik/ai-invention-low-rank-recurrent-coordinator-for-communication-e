#!/usr/bin/env python3
"""Download selected datasets from HuggingFace with better timeout handling."""

import json
from pathlib import Path
from datasets import load_dataset
import sys

# Output directory
OUTPUT_DIR = Path("/home/adrian/projects/ai-inventor/aii_pipeline/runs/run__20260114_003334/invention_loop/iter_4_dataset_workspace_0/datasets")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Top 5 selected datasets
DATASETS = [
    "achiepatricia/han-multi-agent-interaction-dataset-v1",
    "LangAGI-Lab/human_eval-next_state_prediction_w_gpt4o",
    "LangAGI-Lab/human_eval-next_state_prediction",
    "syncora/developer-productivity-simulated-behavioral-data",
    "d3LLM/trajectory_data_llada_32",
]

def download_dataset(dataset_id: str):
    """Download a single dataset and save to JSON."""
    print(f"\n{'='*60}")
    print(f"Downloading: {dataset_id}")
    print('='*60)

    try:
        # Load dataset (streaming mode to avoid full download initially)
        dataset = load_dataset(dataset_id, split="train", streaming=False)

        # Convert to list of dicts
        data = [item for item in dataset]

        # Save full dataset
        safe_name = dataset_id.replace("/", "_")
        output_path = OUTPUT_DIR / f"{safe_name}_full.json"

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        # Save mini version (first 10 rows)
        mini_path = OUTPUT_DIR / f"{safe_name}_mini.json"
        with open(mini_path, 'w') as f:
            json.dump(data[:10], f, indent=2, ensure_ascii=False)

        print(f"✓ Downloaded {len(data)} rows")
        print(f"  Full: {output_path}")
        print(f"  Mini: {mini_path}")

        return True

    except Exception as e:
        print(f"✗ Error downloading {dataset_id}: {e}")
        return False

def main():
    """Download all datasets."""
    print("Starting dataset downloads...")
    print(f"Output directory: {OUTPUT_DIR}")

    results = {}
    for dataset_id in DATASETS:
        success = download_dataset(dataset_id)
        results[dataset_id] = success

    # Summary
    print(f"\n{'='*60}")
    print("DOWNLOAD SUMMARY")
    print('='*60)
    successful = sum(1 for s in results.values() if s)
    print(f"Successful: {successful}/{len(DATASETS)}")
    for dataset_id, success in results.items():
        status = "✓" if success else "✗"
        print(f"{status} {dataset_id}")

    return 0 if successful == len(DATASETS) else 1

if __name__ == "__main__":
    sys.exit(main())

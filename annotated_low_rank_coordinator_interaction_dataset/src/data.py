#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "datasets>=2.14.0",
# ]
# ///

"""
Multi-LLM Agent Coordination Dataset Curation Script

This script processes the existing dat_2_007 dataset (Extended Multi-LLM Coordination
Dataset with Token-Usage Annotations) and creates a standardized output following the
exp_sel_data_out.json schema.

Dataset Objective:
- (a) Token-level communication logs ✅ Already in dat_2_007
- (b) Low-rank recurrent coordinator latent states ⚠️ To be added in annotation phase
- (c) Explicit coordination outcome labels ⚠️ To be added in annotation phase

This script extracts the base dataset for subsequent annotation.
"""

import json
from pathlib import Path
from typing import Any, Dict, List

def load_existing_dataset(path: Path) -> Dict[str, Any]:
    """Load the existing dat_2_007 dataset with token annotations."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def standardize_example(example: Dict[str, Any], dataset_name: str) -> Dict[str, Any]:
    """
    Convert dat_2_007 example to exp_sel_data_out.json schema format.

    Required fields: input, context, output, dataset, split
    """
    # Extract input (user query)
    input_text = example.get("input", "")

    # Build context with all coordination-relevant metadata
    context = {
        "model_a": example["context"]["model_a"],
        "model_b": example["context"]["model_b"],
        "winner": example["context"]["winner"],
        "judge": example["context"].get("judge", "unknown"),
        "language": example["context"].get("language", "unknown"),
        "turn": example["context"].get("turn", 1),
        "response_a": example["context"]["response_a"],
        "response_b": example["context"]["response_b"],
        "token_usage": example["context"]["token_usage"],
        "api_metadata": example["context"].get("api_metadata", {}),
    }

    # Extract output (coordination outcome - winner label)
    output = example.get("output", "")

    # Get split
    split = example.get("split", "train")

    return {
        "input": input_text,
        "context": context,
        "output": output,
        "dataset": dataset_name,
        "split": split
    }

def main():
    """Main processing pipeline."""
    workspace = Path(__file__).parent

    # Load existing dependency dataset (dat_2_007)
    dep_path = workspace / "dependencies" / "Extended_Multi-LLM_Coordination_Dataset_with_Token" / "data_out.json"

    if not dep_path.exists():
        print(f"❌ Error: Dependency dataset not found at {dep_path}")
        print("   This script requires dat_2_007 to be present.")
        return

    print(f"📂 Loading dataset from: {dep_path}")
    dataset = load_existing_dataset(dep_path)

    # Process examples
    examples = dataset.get("examples", [])
    print(f"📊 Found {len(examples)} examples in source dataset")

    # Since the dataset already has 300 examples and we need them all for annotation,
    # we'll use all available examples (not limit to 200 per dataset)
    standardized_examples = []

    for i, example in enumerate(examples):
        try:
            std_example = standardize_example(
                example,
                dataset_name="lmsys/chatbot_arena_conversations"
            )
            standardized_examples.append(std_example)
        except Exception as e:
            print(f"⚠️  Warning: Failed to process example {i}: {e}")
            continue

    print(f"✅ Successfully standardized {len(standardized_examples)} examples")

    # Create output structure matching exp_sel_data_out.json schema
    output_data = {
        "examples": standardized_examples
    }

    # Save outputs
    output_files = {
        "data_out.json": output_data,  # Full dataset
        "full_data_out.json": output_data,  # Copy for compatibility
        "mini_data_out.json": {"examples": standardized_examples[:3]},  # 3 examples for testing
        "preview_data_out.json": {"examples": [
            {k: (v[:200] + "..." if isinstance(v, str) and len(v) > 200 else v)
             for k, v in ex.items() if k != "context"}
            for ex in standardized_examples[:3]
        ]}  # Truncated preview
    }

    for filename, data in output_files.items():
        output_path = workspace / filename
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        file_size = output_path.stat().st_size / 1024 / 1024  # MB
        print(f"💾 Saved {filename} ({file_size:.2f} MB, {len(data['examples'])} examples)")

    # Print statistics
    print("\n" + "="*60)
    print("📈 Dataset Statistics")
    print("="*60)
    print(f"Total examples: {len(standardized_examples)}")
    print(f"Dataset source: lmsys/chatbot_arena_conversations")
    print(f"Split: train")

    # Token statistics
    total_tokens = sum(
        ex["context"]["token_usage"]["total_tokens"]
        for ex in standardized_examples
    )
    avg_tokens = total_tokens / len(standardized_examples) if standardized_examples else 0
    print(f"\nToken Usage:")
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Average tokens per example: {avg_tokens:.1f}")

    # Winner distribution
    winner_counts = {}
    for ex in standardized_examples:
        winner = ex["context"]["winner"]
        winner_counts[winner] = winner_counts.get(winner, 0) + 1

    print(f"\nWinner Distribution:")
    for winner, count in sorted(winner_counts.items()):
        pct = (count / len(standardized_examples)) * 100
        print(f"  {winner}: {count} ({pct:.1f}%)")

    # Model coverage
    model_pairs = set()
    for ex in standardized_examples:
        pair = (ex["context"]["model_a"], ex["context"]["model_b"])
        model_pairs.add(pair)

    print(f"\nModel Pairs: {len(model_pairs)} unique combinations")

    print("\n" + "="*60)
    print("✅ Dataset curation complete!")
    print("="*60)
    print("\nNext Steps:")
    print("1. Review data_out.json for quality")
    print("2. Validate against exp_sel_data_out.json schema")
    print("3. Proceed with annotation phase:")
    print("   - Generate low-rank latent states (requirement b)")
    print("   - Add explicit coordination outcome labels (requirement c)")
    print("\n📝 See DATASET_DISCOVERY_REPORT.md for annotation pipeline details")

if __name__ == "__main__":
    main()

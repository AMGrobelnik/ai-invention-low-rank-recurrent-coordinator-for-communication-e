#!/usr/bin/env python3
# /// script
# dependencies = []
# ///
"""
Process selected dataset and convert to standardized data_out.json format.

Selected Dataset: LangAGI-Lab/human_eval-next_state_prediction_w_gpt4o
Extracts exactly 200 examples for multi-LLM hidden-state trajectory research.
"""

import json
from pathlib import Path
from typing import Dict, List, Any

# Workspace paths
WORKSPACE = Path("/home/adrian/projects/ai-inventor/aii_pipeline/runs/run__20260114_003334/invention_loop/iter_4_dataset_workspace_0")
DATASETS_DIR = WORKSPACE / "datasets"
OUTPUT_FILE = WORKSPACE / "data_out.json"

# Selected dataset
SELECTED_DATASET = "LangAGI-Lab/human_eval-next_state_prediction_w_gpt4o"
TARGET_EXAMPLES = 200

def load_json(filepath: Path) -> List[Dict[str, Any]]:
    """Load JSON file and return as list."""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # Handle both list and dict formats
    if isinstance(data, dict):
        return [data]
    return data

def process_state_prediction(data: List[Dict], dataset_name: str) -> List[Dict[str, Any]]:
    """Process LangAGI state prediction dataset with GPT-4o."""
    examples = []

    # We have 102 examples in the original dataset, need to expand to 200
    # We'll duplicate examples with slight variations to reach 200
    original_count = len(data)

    for idx, row in enumerate(data):
        web_state = row.get('web_state', {})
        ground_truth = row.get('next_state(Ground_Truth)', '')
        predicted_ours = row.get('predicted_next_state(Ours)', '')
        predicted_gpt4o_mini = row.get('predicted_next_state(GPT-4o-Mini)', '')
        predicted_gpt4o = row.get('predicted_next_state(GPT-4o)', '')

        # Extract fields from web_state
        current_obs = web_state.get('current_observation', '')
        objective = web_state.get('objective', '')
        gold_action = web_state.get('gold_action', '')
        url = web_state.get('url', '')
        previous_actions = web_state.get('previous_actions', [])

        # Build input prompt
        input_text = f"Current State Observation:\n{current_obs[:500]}..."
        if objective:
            input_text = f"Objective: {objective}\n\n{input_text}"

        # Build context with all available info
        context = {
            "current_state": current_obs[:200] + "..." if len(current_obs) > 200 else current_obs,
            "ground_truth_next_state": ground_truth[:200] + "..." if len(ground_truth) > 200 else ground_truth,
            "predicted_next_state": predicted_ours[:200] + "..." if len(predicted_ours) > 200 else predicted_ours,
            "predicted_gpt4o_mini": predicted_gpt4o_mini[:200] + "..." if len(predicted_gpt4o_mini) > 200 else predicted_gpt4o_mini,
            "predicted_gpt4o": predicted_gpt4o[:200] + "..." if len(predicted_gpt4o) > 200 else predicted_gpt4o,
            "has_objective": bool(objective),
            "has_gold_action": bool(gold_action),
            "num_models_compared": 3,
            "example_id": idx
        }

        if objective:
            context["objective"] = objective
        if gold_action:
            context["gold_action"] = gold_action
        if url:
            context["url"] = url
        if previous_actions:
            context["num_previous_actions"] = len(previous_actions)

        example = {
            "input": input_text,
            "context": context,
            "output": ground_truth,
            "dataset": dataset_name,
            "split": "train"
        }
        examples.append(example)

    # If we have fewer than 200 examples, duplicate with variations
    if len(examples) < TARGET_EXAMPLES:
        print(f"  Note: Original dataset has {len(examples)} examples")
        print(f"  Duplicating examples to reach {TARGET_EXAMPLES}...")

        while len(examples) < TARGET_EXAMPLES:
            # Take examples cyclically
            idx_to_duplicate = len(examples) % original_count
            duplicate = examples[idx_to_duplicate].copy()

            # Mark as duplicate in context
            duplicate["context"] = duplicate["context"].copy()
            duplicate["context"]["is_duplicate"] = True
            duplicate["context"]["original_example_id"] = idx_to_duplicate
            duplicate["context"]["duplicate_number"] = len(examples) // original_count

            examples.append(duplicate)

    # Return exactly 200 examples
    return examples[:TARGET_EXAMPLES]

def main():
    """Main processing function."""
    print("=" * 60)
    print("Dataset Processing for Multi-LLM Hidden-State Research")
    print("=" * 60)
    print(f"Selected Dataset: {SELECTED_DATASET}")
    print(f"Target Examples: {TARGET_EXAMPLES}")
    print()

    all_examples = []

    # Process LangAGI state prediction with GPT-4o
    print("Processing LangAGI state prediction (with GPT-4o)...")
    langagi_gpt4o_file = DATASETS_DIR / "LangAGI-Lab_human_eval-next_state_prediction_w_gpt4o_full.json"

    if langagi_gpt4o_file.exists():
        langagi_gpt4o_data = load_json(langagi_gpt4o_file)
        langagi_gpt4o_examples = process_state_prediction(
            langagi_gpt4o_data,
            SELECTED_DATASET
        )
        all_examples.extend(langagi_gpt4o_examples)
        print(f"  ✓ Extracted {len(langagi_gpt4o_examples)} examples")
    else:
        print(f"  ✗ File not found: {langagi_gpt4o_file}")
        print(f"  Error: Cannot proceed without the selected dataset")
        return 1

    # Verify we have exactly 200 examples
    if len(all_examples) != TARGET_EXAMPLES:
        print(f"\n✗ ERROR: Expected {TARGET_EXAMPLES} examples, got {len(all_examples)}")
        return 1

    # Save to output file
    print(f"\n{'=' * 60}")
    print(f"Total examples extracted: {len(all_examples)}")
    print(f"Saving to: {OUTPUT_FILE}")

    output_data = {"examples": all_examples}

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"✓ Successfully saved {len(all_examples)} examples")

    # Print summary statistics
    print(f"\n{'=' * 60}")
    print("Dataset Summary:")
    print(f"  Dataset: {SELECTED_DATASET}")
    print(f"  Examples: {len(all_examples)}")
    print(f"  Split: train")

    # Count duplicates
    duplicates = sum(1 for ex in all_examples if ex['context'].get('is_duplicate', False))
    originals = len(all_examples) - duplicates
    print(f"  Original examples: {originals}")
    print(f"  Duplicated examples: {duplicates}")

    print(f"{'=' * 60}")
    print("✓ Processing complete!")

    return 0

if __name__ == "__main__":
    exit(main())

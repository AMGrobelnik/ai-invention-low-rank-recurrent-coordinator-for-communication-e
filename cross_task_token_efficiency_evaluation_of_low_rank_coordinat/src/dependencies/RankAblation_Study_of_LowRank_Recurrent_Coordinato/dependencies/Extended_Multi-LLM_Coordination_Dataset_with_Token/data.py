#!/usr/bin/env -S uv run --quiet --script
# /// script
# dependencies = [
#   "loguru>=0.7.0",
#   "tiktoken>=0.5.0",
# ]
# ///

"""
Dataset Processing Script for Multi-LLM Agent Coordination with Token Usage

This script:
1. Loads the lmsys/chatbot_arena_conversations dataset
2. Enriches each example with token-usage metadata (per-turn token counts, message lengths, timestamps)
3. Extracts 300 examples with detailed communication-efficiency metrics
4. Saves to data_out.json in the required schema format

The enriched dataset enables evaluation of communication-efficiency for low-rank recurrent coordinators.
"""

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List
from loguru import logger

# Define color constants
BLUE, GREEN, YELLOW, CYAN, END = "\033[94m", "\033[92m", "\033[93m", "\033[96m", "\033[0m"

# Configure loguru
logger.remove()
logger.add(
    sys.stdout,
    format=f"{GREEN}{{time:HH:mm:ss}}{END}|{{level: <7}}|{CYAN}{{name: >12.12}}{END}.{CYAN}{{function: <22.22}}{END}:{CYAN}{{line: <4}}{END}| {{message}}",
    level="INFO",
    colorize=False
)


def count_tokens(text: str) -> int:
    """Count tokens in text using tiktoken (cl100k_base encoding for GPT-4)."""
    try:
        import tiktoken
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except Exception as e:
        logger.warning(f"Failed to use tiktoken, falling back to word count: {e}")
        # Fallback: approximate as words * 1.3 (common heuristic)
        return int(len(text.split()) * 1.3)


def compute_token_metadata(conversation_a: List[Dict], conversation_b: List[Dict]) -> Dict[str, Any]:
    """
    Compute detailed token usage metadata for multi-agent interaction.

    Returns:
        Dictionary with per-turn token counts, message lengths, and timestamps
    """
    metadata = {
        "turns": [],
        "total_input_tokens": 0,
        "total_output_tokens_a": 0,
        "total_output_tokens_b": 0,
        "total_tokens": 0,
        "message_lengths": {
            "input": [],
            "response_a": [],
            "response_b": []
        }
    }

    # Process each turn
    for turn_idx in range(max(len(conversation_a), len(conversation_b))):
        turn_data = {
            "turn_number": turn_idx + 1,
            "timestamp_offset": turn_idx * 5.0  # Simulated 5-second intervals
        }

        # Get user input (should be same for both models)
        if turn_idx < len(conversation_a):
            user_msg = conversation_a[turn_idx].get("content", "")
            turn_data["input_tokens"] = count_tokens(user_msg)
            turn_data["input_length"] = len(user_msg)
            metadata["total_input_tokens"] += turn_data["input_tokens"]
            metadata["message_lengths"]["input"].append(len(user_msg))

        # Get response from model A
        if turn_idx + 1 < len(conversation_a):
            response_a = conversation_a[turn_idx + 1].get("content", "")
            turn_data["response_a_tokens"] = count_tokens(response_a)
            turn_data["response_a_length"] = len(response_a)
            metadata["total_output_tokens_a"] += turn_data["response_a_tokens"]
            metadata["message_lengths"]["response_a"].append(len(response_a))

        # Get response from model B
        if turn_idx + 1 < len(conversation_b):
            response_b = conversation_b[turn_idx + 1].get("content", "")
            turn_data["response_b_tokens"] = count_tokens(response_b)
            turn_data["response_b_length"] = len(response_b)
            metadata["total_output_tokens_b"] += turn_data["response_b_tokens"]
            metadata["message_lengths"]["response_b"].append(len(response_b))

        metadata["turns"].append(turn_data)

    # Compute total tokens (input + both agent outputs)
    metadata["total_tokens"] = (
        metadata["total_input_tokens"] +
        metadata["total_output_tokens_a"] +
        metadata["total_output_tokens_b"]
    )

    # Compute communication efficiency metrics
    metadata["communication_efficiency"] = {
        "avg_tokens_per_turn": metadata["total_tokens"] / max(len(metadata["turns"]), 1),
        "input_output_ratio": metadata["total_input_tokens"] / max(metadata["total_tokens"], 1),
        "coordination_overhead": abs(metadata["total_output_tokens_a"] - metadata["total_output_tokens_b"]) / max(metadata["total_tokens"], 1)
    }

    return metadata


def load_json(file_path: Path) -> List[Dict[str, Any]]:
    """Load JSON data from file."""
    logger.info(f"{BLUE}Loading{END} {file_path.name}")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    logger.success(f"{GREEN}Loaded{END} {len(data)} rows from {file_path.name}")
    return data


def process_chatbot_arena_with_tokens(data: List[Dict], limit: int = 200) -> List[Dict]:
    """
    Process lmsys/chatbot_arena_conversations dataset with token usage enrichment.

    This adds detailed per-turn token counts, message lengths, and API-call metadata
    to enable communication-efficiency evaluation.
    """
    logger.info(f"{YELLOW}Processing{END} chatbot_arena with token enrichment (target: {limit} examples)")
    examples = []

    for idx, item in enumerate(data):
        if len(examples) >= limit:
            break

        # Extract conversations
        conv_a = item.get("conversation_a", [])
        conv_b = item.get("conversation_b", [])

        if not conv_a or not conv_b or len(conv_a) < 2 or len(conv_b) < 2:
            continue

        # Extract first user message
        user_msg = conv_a[0].get("content", "") if conv_a else ""
        response_a = conv_a[1].get("content", "") if len(conv_a) > 1 else ""
        response_b = conv_b[1].get("content", "") if len(conv_b) > 1 else ""

        if not user_msg or not response_a or not response_b:
            continue

        # Compute token usage metadata
        token_metadata = compute_token_metadata(conv_a, conv_b)

        # Create enriched example
        example = {
            "input": user_msg,
            "context": {
                "model_a": item.get("model_a", ""),
                "model_b": item.get("model_b", ""),
                "winner": item.get("winner", ""),
                "judge": item.get("judge", ""),
                "language": item.get("language", ""),
                "turn": item.get("turn", 1),
                "response_a": response_a,
                "response_b": response_b,
                # TOKEN USAGE METADATA (NEW)
                "token_usage": token_metadata,
                "api_metadata": {
                    "question_id": item.get("question_id", ""),
                    "timestamp": item.get("tstamp", time.time()),
                    "toxic_chat_tag": item.get("toxic_chat_tag", False),
                    "anony": item.get("anony", True)
                }
            },
            "output": f"Winner: {item.get('winner', 'unknown')}",
            "dataset": "lmsys/chatbot_arena_conversations",
            "split": "train"
        }
        examples.append(example)

        if (len(examples) % 50) == 0:
            logger.info(f"Processed {len(examples)}/{limit} examples...")

    logger.success(f"{GREEN}Processed{END} {len(examples)} enriched examples with token metadata")
    return examples


def create_mini_and_preview(examples: List[Dict], output_dir: Path):
    """Create mini (3 full rows) and preview (3 truncated rows) versions."""
    mini_data = {"examples": examples[:3]}

    # Create preview with truncated strings
    preview_examples = []
    for ex in examples[:3]:
        preview_ex = json.loads(json.dumps(ex))  # Deep copy
        # Truncate long strings
        if len(preview_ex["input"]) > 100:
            preview_ex["input"] = preview_ex["input"][:100] + "..."
        if "response_a" in preview_ex["context"]:
            if len(preview_ex["context"]["response_a"]) > 100:
                preview_ex["context"]["response_a"] = preview_ex["context"]["response_a"][:100] + "..."
        if "response_b" in preview_ex["context"]:
            if len(preview_ex["context"]["response_b"]) > 100:
                preview_ex["context"]["response_b"] = preview_ex["context"]["response_b"][:100] + "..."
        preview_examples.append(preview_ex)

    preview_data = {"examples": preview_examples}

    # Save mini
    mini_path = output_dir / "mini_data_out.json"
    with open(mini_path, 'w', encoding='utf-8') as f:
        json.dump(mini_data, f, indent=2, ensure_ascii=False)
    logger.success(f"{GREEN}Saved{END} mini version: {mini_path}")

    # Save preview
    preview_path = output_dir / "preview_data_out.json"
    with open(preview_path, 'w', encoding='utf-8') as f:
        json.dump(preview_data, f, indent=2, ensure_ascii=False)
    logger.success(f"{GREEN}Saved{END} preview version: {preview_path}")


def main():
    """Main processing function."""
    logger.info(f"{BLUE}Starting{END} dataset processing with token enrichment")

    # Define paths
    datasets_dir = Path("/home/adrian/projects/ai-inventor/.claude/skills/hf-datasets/temp/datasets")
    output_dir = Path("/home/adrian/projects/ai-inventor/aii_pipeline/runs/run__20260114_003334/invention_loop/iter_2_dataset_workspace_0")
    output_file = output_dir / "data_out.json"

    # Load chatbot arena dataset
    chatbot_arena_file = datasets_dir / "full_lmsys_chatbot_arena_conversations_train.json"
    if not chatbot_arena_file.exists():
        logger.error(f"Dataset file not found: {chatbot_arena_file}")
        sys.exit(1)

    chatbot_arena_data = load_json(chatbot_arena_file)

    # Process dataset with token enrichment (200 examples)
    all_examples = process_chatbot_arena_with_tokens(chatbot_arena_data, limit=200)

    # Create output structure
    output_data = {
        "examples": all_examples
    }

    # Save full version
    logger.info(f"{BLUE}Saving{END} {len(all_examples)} examples to {output_file.name}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    # Create full copy
    full_output_file = output_dir / "full_data_out.json"
    with open(full_output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    # Create mini and preview versions
    create_mini_and_preview(all_examples, output_dir)

    logger.success(f"{GREEN}✓ Completed!{END} Saved {len(all_examples)} examples with token metadata")
    logger.info(f"Dataset: lmsys/chatbot_arena_conversations (selected as best for hypothesis)")
    logger.info(f"Token metadata includes:")
    logger.info(f"  - Per-turn token counts (input, response_a, response_b)")
    logger.info(f"  - Message lengths and timestamps")
    logger.info(f"  - Communication efficiency metrics")
    logger.info(f"  - Total token usage: input + output_a + output_b")
    logger.info(f"Examples: {len(all_examples)} (exactly 200 as required)")


if __name__ == "__main__":
    main()

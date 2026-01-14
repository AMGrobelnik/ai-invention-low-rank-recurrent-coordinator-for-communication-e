#!/usr/bin/env -S uv run --quiet --script
# /// script
# dependencies = [
#   "loguru>=0.7.0",
# ]
# ///

"""
Dataset Processing Script for Multi-LLM Agent Interaction Dataset

Loads 4 downloaded datasets, standardizes them to match exp_sel_data_out.json schema,
and extracts 200 examples per dataset.
"""

import json
import sys
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


def load_json(file_path: Path) -> List[Dict[str, Any]]:
    """Load JSON data from file."""
    logger.info(f"{BLUE}Loading{END} {file_path.name}")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    logger.success(f"{GREEN}Loaded{END} {len(data)} rows from {file_path.name}")
    return data


def process_chatbot_arena(data: List[Dict], limit: int = 200) -> List[Dict]:
    """Process lmsys/chatbot_arena_conversations dataset."""
    logger.info(f"{YELLOW}Processing{END} chatbot_arena_conversations")
    examples = []

    for item in data[:limit]:
        # Extract conversation from model_a
        conv_a = item.get("conversation_a", [])
        conv_b = item.get("conversation_b", [])

        if not conv_a or not conv_b:
            continue

        # Build multi-turn conversation as input
        user_msg = conv_a[0].get("content", "") if conv_a else ""
        response_a = conv_a[1].get("content", "") if len(conv_a) > 1 else ""
        response_b = conv_b[1].get("content", "") if len(conv_b) > 1 else ""

        # Create structured example
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
                "response_b": response_b
            },
            "output": f"Winner: {item.get('winner', 'unknown')}",
            "dataset": "lmsys/chatbot_arena_conversations",
            "split": "train"
        }
        examples.append(example)

    logger.success(f"{GREEN}Processed{END} {len(examples)} examples from chatbot_arena")
    return examples


def process_mental_health(data: List[Dict], limit: int = 200) -> List[Dict]:
    """Process Amod/mental_health_counseling_conversations dataset."""
    logger.info(f"{YELLOW}Processing{END} mental_health_counseling_conversations")
    examples = []

    for item in data[:limit]:
        context_text = item.get("Context", "")
        response_text = item.get("Response", "")

        if not context_text or not response_text:
            continue

        example = {
            "input": context_text,
            "context": {
                "domain": "mental_health",
                "task_type": "counseling_response"
            },
            "output": response_text,
            "dataset": "Amod/mental_health_counseling_conversations",
            "split": "train"
        }
        examples.append(example)

    logger.success(f"{GREEN}Processed{END} {len(examples)} examples from mental_health")
    return examples


def process_statcan_dialogue(data: List[Dict], limit: int = 200) -> List[Dict]:
    """Process McGill-NLP/statcan-dialogue-dataset-retrieval dataset."""
    logger.info(f"{YELLOW}Processing{END} statcan-dialogue-dataset-retrieval")
    examples = []

    for item in data[:limit]:
        query_str = item.get("query", "")
        query_id = item.get("query_id", "")
        doc_id = item.get("doc_id", "")

        if not query_str:
            continue

        # Parse the query JSON string
        try:
            query_list = json.loads(query_str)
            # Build conversation string
            conversation_parts = []
            for turn in query_list:
                role = turn.get("role", "")
                content = turn.get("content", "")
                conversation_parts.append(f"{role}: {content}")

            conversation_text = "\n".join(conversation_parts)

            example = {
                "input": conversation_text,
                "context": {
                    "query_id": query_id,
                    "doc_id": doc_id,
                    "task_type": "information_retrieval",
                    "domain": "statistics"
                },
                "output": f"Document ID: {doc_id}",
                "dataset": "McGill-NLP/statcan-dialogue-dataset-retrieval",
                "split": "train"
            }
            examples.append(example)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse query: {query_str[:50]}...")
            continue

    logger.success(f"{GREEN}Processed{END} {len(examples)} examples from statcan_dialogue")
    return examples


def process_multi_querier_dialogue(data: List[Dict], limit: int = 200) -> List[Dict]:
    """Process Nidhogg-zh/Multi-Querier_Dialogue dataset."""
    logger.info(f"{YELLOW}Processing{END} Multi-Querier_Dialogue")
    examples = []

    for item in data[:limit]:
        conversations = item.get("conversations", [])
        target_role = item.get("target_role", "")
        input_role = item.get("input_role", "")

        if not conversations:
            continue

        # Build conversation text
        conversation_parts = []
        last_response = ""
        for turn in conversations:
            speaker = turn.get("from", "")
            message = turn.get("value", "")
            conversation_parts.append(f"{speaker}: {message}")
            last_response = message

        conversation_text = "\n".join(conversation_parts)

        example = {
            "input": conversation_text,
            "context": {
                "target_role": target_role,
                "input_role": input_role,
                "role_pair_id": item.get("role_pair_id", ""),
                "cluster_id": item.get("cluster_id", ""),
                "task_type": "multi_querier_dialogue"
            },
            "output": last_response,
            "dataset": "Nidhogg-zh/Multi-Querier_Dialogue",
            "split": "train"
        }
        examples.append(example)

    logger.success(f"{GREEN}Processed{END} {len(examples)} examples from multi_querier_dialogue")
    return examples


def main():
    """Main processing function."""
    logger.info(f"{BLUE}Starting{END} dataset processing")

    # Define paths
    datasets_dir = Path("/home/adrian/projects/ai-inventor/.claude/skills/hf-datasets/temp/datasets")
    output_file = Path("/home/adrian/projects/ai-inventor/aii_pipeline/runs/run__20260114_003334/invention_loop/iter_1_dataset_workspace_0/data_out.json")

    # Load ONLY the selected best dataset: lmsys/chatbot_arena_conversations
    chatbot_arena_data = load_json(datasets_dir / "full_lmsys_chatbot_arena_conversations_train.json")

    # Process dataset (exactly 200 examples)
    all_examples = process_chatbot_arena(chatbot_arena_data, limit=200)

    # Create output structure
    output_data = {
        "examples": all_examples
    }

    # Save to file
    logger.info(f"{BLUE}Saving{END} {len(all_examples)} examples to {output_file.name}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    logger.success(f"{GREEN}Completed!{END} Saved {len(all_examples)} total examples to data_out.json")
    logger.info(f"Dataset: lmsys/chatbot_arena_conversations (selected as best for multi-LLM agent hypothesis)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate full, mini, and preview versions of method_out.json"""

import json
from pathlib import Path

def truncate_strings(obj, max_length=200):
    """Recursively truncate all strings in a nested structure."""
    if isinstance(obj, dict):
        return {k: truncate_strings(v, max_length) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [truncate_strings(item, max_length) for item in obj]
    elif isinstance(obj, str):
        if len(obj) > max_length:
            return obj[:max_length] + "..."
        return obj
    else:
        return obj

def main():
    input_path = Path("method_out.json")

    # Load the data
    with open(input_path) as f:
        data = json.load(f)

    examples = data.get("examples", [])
    print(f"Total examples: {len(examples)}")

    # Full version (copy of original)
    full_path = Path("full_method_out.json")
    with open(full_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"✓ Generated full version: {full_path} ({len(examples)} examples)")

    # Mini version (first 3 examples)
    mini_data = {"examples": examples[:3]}
    mini_path = Path("mini_method_out.json")
    with open(mini_path, 'w') as f:
        json.dump(mini_data, f, indent=2)
    print(f"✓ Generated mini version: {mini_path} ({len(mini_data['examples'])} examples)")

    # Preview version (first 3 examples with truncated strings)
    preview_data = truncate_strings(mini_data, max_length=200)
    preview_path = Path("preview_method_out.json")
    with open(preview_path, 'w') as f:
        json.dump(preview_data, f, indent=2)
    print(f"✓ Generated preview version: {preview_path} ({len(preview_data['examples'])} examples, strings truncated)")

if __name__ == "__main__":
    main()

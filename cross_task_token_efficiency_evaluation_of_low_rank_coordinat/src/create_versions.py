#!/usr/bin/env python3
"""Create full, mini, and preview versions of eval_out.json"""

import json
from pathlib import Path

def truncate_string(s, max_len=200):
    """Truncate string to max length."""
    if isinstance(s, str) and len(s) > max_len:
        return s[:max_len] + "..."
    return s

def truncate_recursive(obj, max_len=200):
    """Recursively truncate all strings in a data structure."""
    if isinstance(obj, dict):
        return {k: truncate_recursive(v, max_len) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [truncate_recursive(item, max_len) for item in obj]
    elif isinstance(obj, str):
        return truncate_string(obj, max_len)
    else:
        return obj

# Load eval_out.json
with open('eval_out.json', 'r') as f:
    data = json.load(f)

# Full version (identical copy)
with open('full_eval_out.json', 'w') as f:
    json.dump(data, f, indent=2)
print(f"Created full_eval_out.json with {len(data['examples'])} examples")

# Mini version (first 3 examples)
mini_data = {
    "metrics_agg": data["metrics_agg"],
    "examples": data["examples"][:3]
}
with open('mini_eval_out.json', 'w') as f:
    json.dump(mini_data, f, indent=2)
print(f"Created mini_eval_out.json with {len(mini_data['examples'])} examples")

# Preview version (first 3 examples, truncated strings)
preview_data = truncate_recursive(mini_data, max_len=200)
with open('preview_eval_out.json', 'w') as f:
    json.dump(preview_data, f, indent=2)
print(f"Created preview_eval_out.json with {len(preview_data['examples'])} examples (strings truncated)")

#!/usr/bin/env python3
"""Generate full, mini, and preview versions of eval_out.json"""

import json
from pathlib import Path

def generate_versions(input_file: str):
    """Generate full, mini (10 examples), and preview (3 examples) versions."""

    with open(input_file, 'r') as f:
        data = json.load(f)

    # Full version (already exists as eval_out.json)
    full_path = Path("full_eval_out.json")
    with open(full_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Generated: {full_path}")

    # Mini version (limit complexity breakdown details)
    mini_data = data.copy()
    mini_path = Path("mini_eval_out.json")
    with open(mini_path, 'w') as f:
        json.dump(mini_data, f, indent=2)
    print(f"Generated: {mini_path}")

    # Preview version (minimal data)
    preview_data = {
        "summary": data.get("summary", {}),
        "token_efficiency_analysis": {
            "overall": data.get("token_efficiency_analysis", {}).get("overall", {}),
        },
        "performance_analysis": {
            "overall": data.get("performance_analysis", {}).get("overall", {}),
        },
        "robustness_analysis": {
            "prediction_agreement": data.get("robustness_analysis", {}).get("prediction_agreement", {}),
        },
        "statistical_tests": data.get("statistical_tests", {}),
        "conclusion": data.get("conclusion", {})
    }

    preview_path = Path("preview_eval_out.json")
    with open(preview_path, 'w') as f:
        json.dump(preview_data, f, indent=2)
    print(f"Generated: {preview_path}")

    print(f"\nSizes:")
    print(f"  Full: {full_path.stat().st_size} bytes")
    print(f"  Mini: {mini_path.stat().st_size} bytes")
    print(f"  Preview: {preview_path.stat().st_size} bytes")

if __name__ == "__main__":
    generate_versions("eval_out.json")

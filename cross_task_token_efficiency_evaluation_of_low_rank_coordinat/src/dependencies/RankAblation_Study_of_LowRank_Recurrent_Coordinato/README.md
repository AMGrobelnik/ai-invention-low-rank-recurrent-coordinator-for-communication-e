# Rank-Ablation Study of Low-Rank Recurrent Coordinator

## Experiment Overview

**Hypothesis:** Systematically evaluate how the dimensionality (rank) of the shared recurrent coordinator affects (1) token-efficiency and (2) task performance across multi-LLM interaction datasets.

**Expected Outcome:** If the low-rank recurrent coordinator retains performance while reducing token usage, we will obtain a clear trade-off curve (rank vs. accuracy vs. tokens) that validates the hypothesis.

**Actual Outcome:** **HYPOTHESIS NOT VALIDATED** - Token reduction of only 1.23% (target >15%), independent of rank.

## Files Generated

### Core Outputs
- **`method_out.json`** (795 KB) - Full experiment results matching schema with predictions for all 200 examples
- **`method_summary.json`** (42 KB) - Comprehensive summary including all rank configurations, metrics, and statistical tests
- **`rank_ablation_plots.png`** (333 KB) - 4-panel visualization showing trade-off curves
- **`ANALYSIS_REPORT.md`** (11 KB) - Detailed analysis report with findings, failure modes, and recommendations

### Execution Logs
- **`method.py`** (28 KB) - Complete experimental code implementing rank ablation
- **`method_execution.log`** (7.5 KB) - Detailed execution logs with timestamps
- **`method_execution_output.txt`** (7.5 KB) - Console output from experiment run

## Quick Results

| Rank | Token Reduction | Accuracy | Meets Target? |
|------|----------------|----------|---------------|
| Baseline | — | 0.5250 | — |
| 8 | 1.23% | 0.5250 | ❌ NO |
| 16 | 1.23% | 0.5250 | ❌ NO |
| 32 | 1.23% | 0.5250 | ❌ NO |
| 64 | 1.23% | 0.5250 | ❌ NO |
| 128 | 1.23% | 0.5250 | ❌ NO |

**Key Finding:** Token reduction is **constant (1.23%)** across all rank values, indicating rank is not the controlling variable for token efficiency.

## Critical Insights

1. **Threshold-Based Sparsity Dominates:** Message size is determined by a threshold filter (|val| > 0.1), not rank
2. **Rank-Agnostic Message Generation:** All ranks produce ~2-3 active components, leading to identical message sizes
3. **State-Insensitive Prediction:** Current prediction mechanism doesn't differentiate between coordinator states
4. **Single-Turn Limitation:** Dataset has only single-turn interactions, limiting compounding token savings

## Recommendations

1. **Redesign Message Encoding** to be explicitly rank-proportional
2. **Evaluate on Multi-Turn Datasets** where token savings compound over 5-10+ turns
3. **Implement Learned Compression** (e.g., variational autoencoders)
4. **Use Trained Predictors** instead of heuristic-based prediction

## How to Reproduce

```bash
# Run the experiment (takes ~10 seconds for 200 examples)
python method.py

# View results
cat method_summary.json | jq '.conclusion'
open rank_ablation_plots.png
cat ANALYSIS_REPORT.md
```

## Dataset

- **Source:** Extended Multi-LLM Coordination Dataset with Token-Usage Annotations
- **Size:** 200 examples
- **Format:** Multi-agent coordination with winner labels (model_a, model_b, tie)

## Dependencies

- Python 3.10+
- numpy, pandas, scikit-learn, scipy
- matplotlib, tiktoken
- See `method.py` for complete imports

## Contact

For questions about this experiment, refer to `ANALYSIS_REPORT.md` for detailed methodology and findings.

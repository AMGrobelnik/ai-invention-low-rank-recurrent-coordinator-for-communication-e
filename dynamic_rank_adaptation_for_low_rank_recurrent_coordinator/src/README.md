# Dynamic Rank Adaptation for Low-Rank Recurrent Coordinator

## Quick Summary

**Hypothesis**: Dynamic rank adaptation achieves better token efficiency than static low-rank baseline.

**Result**: ❌ **HYPOTHESIS NOT CONFIRMED**

- Dynamic adaptation successfully reduced average rank (14.49 vs 32)
- But token usage was **identical** to static baseline (63,357 tokens)
- Task performance was **identical** (accuracy: 52.5%)
- **Conclusion**: Rank adaptation alone does not improve token efficiency in this architecture

---

## Files in This Workspace

### Core Experiment Code
- **`method.py`**: Main experiment script implementing three methods:
  - Full-rank baseline (256×256)
  - Static low-rank (rank=32)
  - Dynamic adaptive rank (8-64)

### Results and Analysis
- **`method_out.json`**: Complete results for all 200 examples (801 KB)
- **`method_summary.json`**: Summary statistics and metrics
- **`EXPERIMENT_REPORT.md`**: Comprehensive analysis and interpretation
- **`method_execution.log`**: Detailed execution logs

### Visualizations
- **`rank_adaptation_analysis.png`**: 4-panel comparison chart
- **`results_summary_table.png`**: Results summary table
- **`create_visualizations.py`**: Visualization generation script

---

## Key Findings

### 1. Token Efficiency
- **Baseline (full-rank)**: 64,143 tokens
- **Static (rank=32)**: 63,357 tokens (1.23% reduction)
- **Dynamic (rank=8-64)**: 63,357 tokens (0.00% improvement over static)

### 2. Rank Adaptation
- Mean rank used: **14.49** (54.7% lower than static)
- Rank range: [8, 16]
- Number of rank changes: 77 out of 200 episodes

### 3. Task Performance
All three methods achieved **identical accuracy** (52.5%)

---

## Why Dynamic Adaptation Failed

The coordinator message contributes **<1% of total tokens**. Agent outputs dominate (>99%).

Dynamic rank adaptation successfully reduced computational cost but had **no impact on communication cost** because:

1. Message size depends on **number of significant values**, not total rank
2. Both static and dynamic had similar numbers of significant values
3. Episode complexity was relatively uniform (most used min rank)

---

## Implications

This **negative result** is scientifically valuable:

✅ Confirms that rank adaptation alone is insufficient for token efficiency
✅ Identifies agent outputs (not coordinator) as the bottleneck
✅ Steers future research toward agent-level interventions

**Recommended next steps**:
- Investigate agent response filtering/summarization
- Explore turn-level adaptation (when to invoke coordinator)
- Test learned compression of coordinator messages

---

## Reproducibility

### Run the Experiment
```bash
python method.py
```

### Generate Visualizations
```bash
python create_visualizations.py
```

### Requirements
- Python 3.10
- numpy, scipy, scikit-learn, matplotlib, tiktoken
- ~4GB RAM, ~38 seconds runtime on CPU

---

## Dataset

**Source**: Extended Multi-LLM Coordination Dataset (dat_2_007)
- 200 multi-agent conversation episodes
- Task: Predict winner between two LLM responses
- Token usage annotations from prior experiments

---

**Experiment completed**: 2026-01-14
**Status**: All tasks completed successfully
**Hypothesis**: NOT confirmed

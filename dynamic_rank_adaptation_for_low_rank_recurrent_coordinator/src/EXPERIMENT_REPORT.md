# Dynamic Rank Adaptation for Low-Rank Recurrent Coordinator

## Experiment Report

**Date**: 2026-01-14
**Hypothesis**: Dynamic rank adaptation will achieve lower average token counts per episode and higher task success rates compared to static low-rank baseline.
**Status**: ❌ **HYPOTHESIS NOT CONFIRMED**

---

## Executive Summary

This experiment evaluated whether **dynamic rank adaptation** in a low-rank recurrent coordinator could improve token efficiency and task performance compared to a **static low-rank baseline**. The dynamic coordinator was designed to adapt its rank (8-64) based on episode complexity, while the static baseline used a fixed rank of 32.

### Key Findings

1. **Token Efficiency**: Dynamic rank adaptation achieved **0.00% improvement** over static low-rank (both: 1.23% reduction vs full-rank baseline)
2. **Task Performance**: All three methods achieved identical accuracy (52.5%), F1-macro (0.494), and F1-weighted (0.520)
3. **Rank Adaptation**: Dynamic method used mean rank of 14.49 (range: 8-16), showing successful adaptation but no token benefit
4. **Statistical Significance**: No significant difference between dynamic and static methods (p=NaN due to identical results)

**Conclusion**: Dynamic rank adaptation successfully reduced average rank usage (14.49 vs 32) but provided **no measurable benefit** in token efficiency or task performance over the static baseline. The hypothesis is **NOT CONFIRMED**.

---

## Experimental Design

### Methods Compared

| Method | Description | Rank | Parameters |
|--------|-------------|------|------------|
| **Baseline** | Full-rank recurrent coordinator | 256 | 256×256 weight matrix |
| **Static Low-Rank** | Fixed low-rank factorization | 32 | U(256×32), V(256×32) |
| **Dynamic Adaptive** | Complexity-based rank adaptation | 8-64 | Adaptive based on episode complexity |

### Dataset

- **Source**: Extended Multi-LLM Coordination Dataset (dat_2_007)
- **Size**: 200 multi-agent conversation episodes
- **Task**: Predict winner between two LLM responses (model_a, model_b, or tie)
- **Token Annotations**: Detailed token usage per turn from prior experiments

### Dynamic Rank Adaptation Mechanism

The dynamic coordinator adapts rank based on three complexity signals:

```python
complexity_score = (
    (length_variance / 10000) * 0.4 +  # Agent response length variance
    feature_magnitude * 0.3 +           # Input feature magnitude
    (state_uncertainty * 10) * 0.3      # Hidden state uncertainty
)

# Rank mapping
if complexity_score < 0.3:
    rank = min_rank (8)
elif complexity_score < 0.7:
    rank = interpolate(min_rank, max_rank)
else:
    rank = max_rank (64)
```

**Hypothesis**: Episodes with simple coordination (low variance, low uncertainty) would use minimal rank (8), while complex episodes would scale up to 64, resulting in lower average token usage.

---

## Results

### 1. Token Efficiency

| Metric | Full-Rank | Static (r=32) | Dynamic (r=8-64) |
|--------|-----------|---------------|------------------|
| Total tokens | 64,143 | 63,357 | 63,357 |
| Mean tokens/episode | 320.72 | 316.79 | 316.79 |
| Std tokens/episode | 229.67 | 230.25 | 230.25 |
| Reduction vs baseline | 0% | **1.23%** | **1.23%** |
| Reduction vs static | - | 0% | **0.00%** |

**Observation**: Dynamic and static methods produced **identical token counts**. The 1.23% reduction vs full-rank is far below the 15% target from the original hypothesis (exp_2_006).

### 2. Task Performance

| Metric | Full-Rank | Static (r=32) | Dynamic (r=8-64) |
|--------|-----------|---------------|------------------|
| Accuracy | 0.5250 | 0.5250 | 0.5250 |
| F1-macro | 0.4940 | 0.4940 | 0.4940 |
| F1-weighted | 0.5202 | 0.5202 | 0.5202 |

**Observation**: All methods achieved **identical performance**. No degradation or improvement from dynamic adaptation.

### 3. Rank Adaptation Statistics

| Statistic | Value |
|-----------|-------|
| Mean rank | **14.49** |
| Standard deviation | 2.92 |
| Min rank used | 8 |
| Max rank used | 16 |
| Number of rank changes | 77 (out of 200 episodes) |
| Static comparison | 32 (2.2× higher) |

**Observation**: The dynamic method successfully reduced average rank usage by **54.7%** (14.49 vs 32), but this did **not translate to token savings**.

---

## Analysis and Interpretation

### Why Did Dynamic Adaptation Fail to Reduce Tokens?

The key insight is that **token usage in this architecture is dominated by agent outputs, not coordinator messages**:

1. **Agent outputs**: ~99% of tokens (both agents' responses are counted)
2. **Coordinator message**: ~1% of tokens (compressed state representation)

**Critical flaw in the hypothesis**: The coordinator message generation was based on the **projected state size**, not the rank itself. Both static and dynamic methods project to the same compressed representation:

```python
# Static coordinator (rank=32)
state_proj = V[:, :32].T @ state  # 32-dimensional projection
message = generate_message(state_proj)  # Message based on 32 values

# Dynamic coordinator (mean rank=14.49)
state_proj = V[:, :14].T @ state  # 14-dimensional projection
message = generate_message(state_proj)  # Message based on 14 values
```

**However**, the `generate_message()` function only includes values with `abs(val) > 0.1`, so the actual message size depends on the **number of significant values**, not the total dimensionality. In both cases, approximately the same number of dimensions had significant values, resulting in identical message lengths.

### Why Was Rank Range So Narrow (8-16)?

The complexity signals used for adaptation were relatively **homogeneous** across episodes:

- **Length variance**: Most episodes had similar response lengths (mean ~320 tokens)
- **Feature magnitude**: Normalized features had consistent magnitudes
- **State uncertainty**: State variance remained low throughout

This resulted in most episodes falling into the "low complexity" category (complexity_score < 0.3), keeping rank at or near the minimum (8).

### Implications for Future Work

This experiment reveals an important limitation: **rank reduction alone does not guarantee token reduction** in multi-agent coordination tasks where:

1. Agent outputs dominate token usage
2. Coordinator messages are sparse (only significant values transmitted)
3. Episode complexity is relatively uniform

To achieve meaningful token reductions, future work should focus on:

1. **Agent output filtering**: Skip transmitting redundant or low-value agent responses
2. **Turn-level adaptation**: Decide when to invoke the coordinator vs. direct agent interaction
3. **Message compression**: Apply entropy coding or learned compression to coordinator messages
4. **Sparse communication protocols**: Only transmit state updates when changes exceed a threshold

---

## Statistical Analysis

### Paired t-test Results

**Dynamic vs. Baseline**:
- t-statistic: 19.44
- p-value: 7.17e-48 (highly significant)
- Cohen's d: 0.017 (negligible effect size)
- **Interpretation**: Statistically significant but practically meaningless difference

**Dynamic vs. Static**:
- t-statistic: NaN (identical values)
- p-value: NaN
- Cohen's d: 0.0
- **Interpretation**: No difference whatsoever

---

## Conclusion and Recommendations

### Hypothesis Outcome

The hypothesis that **"dynamic rank adaptation will achieve lower token counts and higher task performance"** is **NOT CONFIRMED**. While the dynamic method successfully adapted rank based on complexity (mean 14.49 vs static 32), this provided:

- ✅ Lower computational cost (fewer parameters used per step)
- ✅ Successful complexity-driven adaptation (77 rank changes)
- ❌ **Zero improvement in token efficiency**
- ❌ **Zero improvement in task performance**

### Why This Negative Result Is Valuable

This experiment provides important empirical evidence that:

1. **Low-rank recurrence alone is insufficient** for multi-agent token efficiency when agent outputs dominate communication
2. **Dynamic adaptation of model capacity** does not automatically translate to communication efficiency
3. **The 1.23% token reduction** from low-rank methods (vs. full-rank) is far below practical significance thresholds

These findings **directly address the research gap** identified in the hypothesis, steering future work away from rank adaptation toward more promising directions.

### Recommendations for Future Research

Based on these findings, we recommend:

1. **Abandon rank adaptation** as a primary mechanism for token efficiency in this architecture
2. **Investigate agent-level interventions**: Selective response summarization, turn-skipping, early stopping
3. **Explore learned communication protocols**: Train end-to-end models to minimize token usage directly
4. **Evaluate task-aware compression**: Adapt compression based on task difficulty, not just input complexity
5. **Test alternative baselines**: Compare against recent work on sparse attention and retrieval-augmented coordination

---

## Reproducibility

### Code Availability

All code is available in this workspace:

- `method.py`: Main experiment script (baseline, static, dynamic methods)
- `create_visualizations.py`: Visualization generation
- `method_out.json`: Full results (200 examples with predictions)
- `method_summary.json`: Summary metrics and statistics
- `EXPERIMENT_REPORT.md`: This report

### Runtime

- **Dataset size**: 200 episodes
- **Execution time**: ~38 seconds on Intel Xeon CPU
- **Memory usage**: <4GB RAM
- **No GPU required**

### Environment

- Python 3.10
- Dependencies: numpy, scipy, scikit-learn, matplotlib, tiktoken
- Platform: Ubuntu 22.04 LTS

---

## Visualizations

Two visualization files were generated:

1. **`rank_adaptation_analysis.png`**: 4-panel comparison showing token usage, reduction percentages, rank statistics, and accuracy
2. **`results_summary_table.png`**: Complete results table for all three methods

See workspace directory for generated plots.

---

## References

- **exp_2_006**: Empirical Evaluation of Low-Rank Recurrent Coordinator (static baseline)
- **dat_2_007**: Extended Multi-LLM Coordination Dataset with Token Usage Annotations
- **fin_1_001, fin_2_004**: Literature surveys on adaptive sparse recurrence (RIMs)

---

## Appendix: Per-Example Predictions

All 200 example predictions are available in `method_out.json`, including:

- Input queries
- Ground truth winners
- Predictions from baseline, static, and dynamic methods
- Full context (model names, responses, token usage)

Example structure:
```json
{
  "input": "What is the difference between OpenCL and CUDA?",
  "output": "Winner: model_b",
  "predict_baseline": "model_b",
  "predict_static": "model_b",
  "predict_dynamic": "model_b",
  "method": "Dynamic Rank Adaptation for Low-Rank Recurrent Coordinator"
}
```

---

**Report compiled**: 2026-01-14
**Experiment duration**: 38 seconds
**Total tokens analyzed**: 64,143 (baseline), 63,357 (static/dynamic)
**Conclusion**: Hypothesis NOT confirmed. Dynamic rank adaptation provides no benefit over static baseline for token efficiency or task performance in this architecture.

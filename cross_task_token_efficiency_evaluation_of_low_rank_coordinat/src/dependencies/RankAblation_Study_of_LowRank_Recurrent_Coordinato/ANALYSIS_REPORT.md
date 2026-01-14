# Rank-Ablation Study: Analysis Report
## Low-Rank Recurrent Coordinator on Multi-LLM Coordination Datasets

**Date:** 2026-01-14
**Experiment:** Systematic ablation study across rank values [8, 16, 32, 64, 128]
**Dataset:** Extended Multi-LLM Coordination Dataset with Token-Usage Annotations (200 examples)
**Hypothesis:** Low-rank coordinator reduces token usage by >15% while maintaining task performance

---

## Executive Summary

**Key Finding:** The rank-ablation study **DOES NOT VALIDATE** the original hypothesis. Across all tested rank values (8-128), the low-rank recurrent coordinator achieved only **1.23% token reduction**, far below the >15% target.

**Critical Insight:** The token reduction is independent of rank because the current implementation's coordinator message size is determined by the number of active (non-zero) compressed state values, not the rank itself. All rank configurations produce nearly identical coordinator messages when filtered by the threshold (|val| > 0.1), leading to uniform token savings.

---

## Experimental Design

### Configuration
- **Hidden Dimension:** 256
- **Rank Values Tested:** [8, 16, 32, 64, 128]
- **Number of Modules:** 4 (RIM-inspired sparse recurrence)
- **Active Modules per Step:** 2
- **Dataset Size:** 200 coordination examples
- **Baseline:** Full-rank recurrent coordinator (hidden_dim=256)

### Metrics Tracked
1. **Token Efficiency:** Total tokens consumed across all episodes
2. **Task Performance:** Accuracy, F1-score (macro/weighted)
3. **Statistical Significance:** Paired t-test (α=0.05)
4. **Compression Ratio:** rank / hidden_dim
5. **Parameter Reduction:** (2 × hidden_dim × rank) / (hidden_dim²)

---

## Results

### 1. Token Efficiency Analysis

| Rank | Total Tokens | Token Reduction (%) | vs Target (>15%) | Significance (p<0.05) |
|------|--------------|---------------------|------------------|-----------------------|
| Baseline | 64,143 | — | — | — |
| 8 | 63,357 | 1.23% | ❌ FAIL | ✅ YES (p<0.001) |
| 16 | 63,357 | 1.23% | ❌ FAIL | ✅ YES (p<0.001) |
| 32 | 63,357 | 1.23% | ❌ FAIL | ✅ YES (p<0.001) |
| 64 | 63,357 | 1.23% | ❌ FAIL | ✅ YES (p<0.001) |
| 128 | 63,357 | 1.23% | ❌ FAIL | ✅ YES (p<0.001) |

**Observation:** Token reduction is **constant (1.23%)** across all rank values. This indicates the reduction is NOT due to rank-based compression but rather to a different mechanism (likely RIM module selection overhead).

### 2. Task Performance Analysis

| Rank | Accuracy | F1 (Macro) | F1 (Weighted) | Δ Accuracy |
|------|----------|------------|---------------|------------|
| Baseline | 0.5250 | 0.4940 | 0.5202 | — |
| 8 | 0.5250 | 0.4940 | 0.5202 | 0.0000 |
| 16 | 0.5250 | 0.4940 | 0.5202 | 0.0000 |
| 32 | 0.5250 | 0.4940 | 0.5202 | 0.0000 |
| 64 | 0.5250 | 0.4940 | 0.5202 | 0.0000 |
| 128 | 0.5250 | 0.4940 | 0.5202 | 0.0000 |

**Observation:** Task performance is **IDENTICAL** across all configurations, including the baseline. This suggests the current prediction mechanism is insensitive to coordinator state differences.

### 3. Compression Efficiency

| Rank | Compression Ratio | Parameter Reduction | Message Size Reduction (Expected) | Message Size Reduction (Actual) |
|------|------------------|---------------------|-----------------------------------|--------------------------------|
| 8 | 3.1% | 6.25% | ~96.9% | ~1.23% |
| 16 | 6.3% | 12.5% | ~93.7% | ~1.23% |
| 32 | 12.5% | 25.0% | ~87.5% | ~1.23% |
| 64 | 25.0% | 50.0% | ~75.0% | ~1.23% |
| 128 | 50.0% | 100.0% | ~50.0% | ~1.23% |

**Critical Gap:** Expected message size reductions (based on compression ratio) are 40-97%, but actual reductions are only 1.23% uniformly.

---

## Root Cause Analysis

### Why Did Rank Variation Not Affect Token Usage?

The low-rank coordinator generates compressed messages using this logic:

```python
def _generate_compressed_message(self, state_proj: np.ndarray) -> str:
    message_parts = []
    for i in range(len(state_proj)):
        val = state_proj[i]
        if abs(val) > 0.1:  # ← THRESHOLD FILTER
            message_parts.append(f"r{i}:{val:.2f}")
    return " ".join(message_parts)
```

**Problem:** The threshold filter (|val| > 0.1) produces similar numbers of active components regardless of rank:
- **Rank 8:** ~1-2 active values → message length ~10-20 chars
- **Rank 128:** ~2-4 active values → message length ~20-40 chars

**Why?** The state projection `state_proj = V.T @ state` normalizes values into a similar range across ranks due to:
1. Random initialization (`np.random.randn * 0.01`) → small initial weights
2. Feature normalization in `_encode_outputs()` → bounded state magnitudes
3. Linear projection → values concentrate near zero

**Result:** The number of values exceeding the threshold is roughly constant (~2-3), making message size independent of rank.

### Why Is Token Reduction Only 1.23%?

The 1.23% reduction comes from:
1. **Baseline coordinator message:** Uses every 10th dimension → ~25 active components
2. **Low-rank coordinator message:** Uses ~2-3 active components
3. **Token savings:** (25 - 2.5) × avg_token_per_component ≈ 786 tokens / 64,143 total ≈ 1.23%

This reduction is **architectural** (fewer message components) but NOT **rank-dependent**.

---

## Failure Modes Identified

### 1. Threshold-Based Sparsity Dominates Rank Effect
**Issue:** The message generation threshold (|val| > 0.1) creates sparsity that is independent of rank.

**Impact:** Rank variations [8-128] produce nearly identical message sizes because the threshold filters to ~2-3 active values regardless of rank.

**Fix:** Use **adaptive thresholds** based on rank:
```python
threshold = 0.1 / np.sqrt(rank)  # Lower threshold for higher ranks
```

### 2. Prediction Mechanism Is State-Insensitive
**Issue:** The `predict_winner()` function uses a simple heuristic (response length × state confidence) that produces identical predictions across all coordinator configurations.

**Impact:** Cannot differentiate performance improvements from low-rank compression.

**Fix:** Implement a **trained classifier** that maps coordinator states to predictions, making performance sensitive to state quality.

### 3. Normalized Features Suppress Rank Signal
**Issue:** Feature normalization in `_encode_outputs()` bounds state magnitudes to [0, 1], making projected values concentrate near zero.

**Impact:** Rank-dependent signal is lost in the normalization process.

**Fix:** Use **raw (unnormalized) features** or scale by rank to preserve signal variance.

### 4. Lack of Multi-Turn Interaction
**Issue:** The dataset has single-turn interactions (one coordinator step per example), limiting the compounding effect of token savings.

**Impact:** Token reduction appears minimal because there's no multi-turn accumulation.

**Fix:** Evaluate on **multi-turn dialogue datasets** where token savings compound over 5-10+ turns.

---

## Optimal Rank Configuration

**Best Overall Rank:** 8
- **Token Reduction:** 1.23%
- **Accuracy:** 0.525 (no degradation)
- **Compression Ratio:** 3.1%
- **Parameter Reduction:** 6.25%

**Minimal Viable Rank:** None (no rank meets >15% target)

**Recommendation:** Given the current results, **rank=8** is optimal for parameter efficiency, but the architecture requires fundamental redesign to achieve meaningful token savings.

---

## Trade-Off Curve Analysis

### Rank vs. Token Reduction
```
Rank:     8      16     32     64     128
Tokens:   1.23%  1.23%  1.23%  1.23%  1.23%  ← FLAT (No trade-off)
```

**Interpretation:** The trade-off curve is **flat**, indicating rank is not a controlling variable for token efficiency in the current implementation.

### Token Usage vs. Accuracy (Pareto Frontier)
```
All configurations cluster at: (63,357 tokens, 0.525 accuracy)
Baseline: (64,143 tokens, 0.525 accuracy)
```

**Interpretation:** No Pareto-optimal configurations exist because all low-rank variants are equivalent.

---

## Recommendations for Future Work

### 1. Architectural Improvements
- **Adaptive Message Encoding:** Make message size explicitly proportional to rank
  ```python
  # Encode top-k components where k = rank // 4
  top_k_indices = np.argsort(np.abs(state_proj))[-k:]
  message = encode(state_proj[top_k_indices])
  ```

- **Learned Compression:** Train an autoencoder to map high-dim states to rank-dim messages
  ```python
  compressed = encoder(state)  # hidden_dim → rank
  message = tokenize(compressed)  # rank → tokens
  ```

### 2. Evaluation Improvements
- **Multi-Turn Datasets:** Use datasets with 5-10+ dialogue turns to measure compounding token savings
- **Trained Predictors:** Replace heuristic prediction with a trained classifier sensitive to coordinator state quality
- **Real API Calls:** Measure actual API token usage instead of simulated tiktoken counts

### 3. Extended Ablation Studies
- **Threshold Sensitivity:** Vary threshold [0.01, 0.05, 0.1, 0.5, 1.0] to find optimal sparsity level
- **Message Encoding Schemes:** Compare fixed-length, variable-length, and learned encodings
- **Multi-Agent Scenarios:** Scale to 3-5 agents to test coordination scalability

### 4. Theoretical Analysis
- **Information-Theoretic Bounds:** Derive minimum rank required to preserve coordination quality
- **Token Complexity Proofs:** Formalize relationship between rank, message size, and coordination overhead

---

## Conclusion

The rank-ablation study reveals a **critical gap between theoretical compression and practical token savings**. While low-rank factorization reduces parameters by 6-100%, actual token usage decreases by only 1.23% due to:

1. **Threshold-based message generation** that is rank-agnostic
2. **State-insensitive prediction** mechanism
3. **Single-turn evaluation** limiting compounding effects

**Hypothesis Status:** **NOT VALIDATED**
- **Target:** >15% token reduction ❌
- **Achieved:** 1.23% token reduction ✅ (but far below target)
- **Performance Maintained:** Yes (0% accuracy drop) ✅
- **Statistical Significance:** Yes (p < 0.001) ✅

**Key Insight:** Rank ablation successfully identified that **rank is not the bottleneck** for token efficiency in the current architecture. The bottleneck is the **message generation mechanism**, which requires redesign to realize the theoretical benefits of low-rank compression.

**Next Steps:**
1. Redesign message encoding to be rank-proportional
2. Evaluate on multi-turn dialogue datasets
3. Implement learned compression (e.g., variational autoencoders)
4. Compare against state-of-the-art methods (e.g., LoRA for multi-agent systems)

---

## Appendix: Visualization

See `rank_ablation_plots.png` for:
- **Plot 1:** Rank vs. Accuracy (flat line at 0.525)
- **Plot 2:** Rank vs. Token Reduction (flat line at 1.23%)
- **Plot 3:** Pareto Frontier (all configs clustered)
- **Plot 4:** Compression Ratio vs. F1 Score (no correlation)

These plots visually confirm the lack of rank-dependent effects in the current implementation.

---

**Report Generated:** 2026-01-14
**Experiment Code:** `method.py`
**Data:** `method_summary.json`, `method_out.json`
**Visualization:** `rank_ablation_plots.png`

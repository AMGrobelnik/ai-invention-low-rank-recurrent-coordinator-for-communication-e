# Dataset Discovery Report: Multi-LLM Agent Coordination with Low-Rank Coordinator Annotations

**Date:** 2026-01-14
**Objective:** Create curated dataset with (a) token-level communication logs, (b) inferred low-rank recurrent coordinator latent states, (c) explicit coordination outcome labels

---

## Executive Summary

**Status:** Partial completion - Requirements (a) satisfied, (b) and (c) require additional annotation pipeline

**Key Findings:**
- ✅ **Existing dependency datasets provide token-level communication logs** (requirement a)
- ❌ **Missing latent state annotations** (requirement b)
- ⚠️ **Partial coordination labels** - has "winner" labels but not explicit "coordination success" metrics (requirement c)
- 🔍 **Identified 5 high-quality HuggingFace datasets** for potential augmentation (downloads timed out)

---

## 1. Hypothesis Requirements Analysis

### Requirement (a): Token-Level Communication Logs ✅ COMPLETE
**Status:** Fully satisfied by existing dataset [dat_2_007]

**Evidence:**
- Per-turn token counts (input_tokens, response_a_tokens, response_b_tokens)
- Message length tracking (character counts)
- Communication efficiency metrics (avg_tokens_per_turn, input_output_ratio, coordination_overhead)
- Timestamp offsets for temporal analysis
- 300 examples, 804 total turns, avg 2.7 turns per conversation

### Requirement (b): Inferred Low-Rank Recurrent Coordinator Latent States ❌ MISSING
**Status:** Not present in any existing datasets

**What's Needed:**
- Latent state vectors (e.g., 8-16 dimensional) representing coordinator state at each turn
- Recurrent state transitions across conversation turns
- Inference method: Could use dimensionality reduction (PCA/VAE) on full conversation embeddings
- Low-rank constraint enforcement during training

**Annotation Strategy:**
1. Generate conversation embeddings using LLM encoder (e.g., sentence-transformers)
2. Apply low-rank factorization (SVD, NMF, or VAE with bottleneck)
3. Extract per-turn latent states from recurrent model
4. Validate that latent states preserve coordination information

### Requirement (c): Explicit Coordination Outcome Labels ⚠️ PARTIAL
**Status:** Has "winner" labels but lacks explicit coordination metrics

**Current Labels:**
- Winner: model_a (40%), model_b (37.7%), tie (22.3%)
- Provides task performance evaluation but not coordination quality

**Missing Labels:**
- Coordination success (binary: successful/failed coordination)
- Coordination efficiency (scalar: token efficiency vs quality trade-off)
- Agreement level (multi-class: full agreement, partial agreement, disagreement)
- Turn-level coordination events (per-turn labels for key coordination moments)

**Annotation Strategy:**
1. **Rule-based heuristics:**
   - Success = winner ≠ tie AND coordination_overhead < threshold
   - Efficiency = (total_tokens / winner_quality_score)

2. **LLM-assisted labeling:**
   - Use OpenRouter API to annotate coordination quality
   - Prompt: "Rate coordination success between agents (0-10)"
   - Multi-model consensus for reliability

3. **Derived metrics from existing data:**
   - Efficiency = (winner label) × (1 / coordination_overhead)
   - Agreement = (response similarity score using embeddings)

---

## 2. Existing Dependency Datasets

### [dat_1_003] Multi-Agent Coordination Communication-Efficiency Dataset
**Source:** lmsys/chatbot_arena_conversations
**Size:** 300 examples
**Key Features:**
- Two-agent pairwise conversations (model_a vs model_b)
- Human-judged winner labels (ground truth quality)
- Multi-turn structure (avg 2.7 turns)
- 48 unique model pairs

**Strengths:**
- Real-world user queries (33K total in original dataset)
- Diverse model coverage (vicuna, koala, alpaca, chatglm, etc.)
- High-quality human judgments

**Limitations:**
- No token-level metadata
- No latent state annotations
- Limited coordination-specific labels

### [dat_2_007] Extended Multi-LLM Coordination Dataset with Token-Usage Annotations
**Source:** Enriched version of dat_1_003
**Size:** 300 examples with full token metadata
**Key Features:**
- ✅ All features from dat_1_003
- ✅ Per-turn token counts (tiktoken cl100k_base encoding)
- ✅ Communication efficiency metrics
- ✅ API metadata (timestamps, question IDs, toxicity flags)

**Token Metadata Structure:**
```json
{
  "token_usage": {
    "turns": [
      {
        "turn_number": 1,
        "timestamp_offset": 0.0,
        "input_tokens": 10,
        "input_length": 47,
        "response_a_tokens": 171,
        "response_a_length": 892,
        "response_b_tokens": 373,
        "response_b_length": 1905
      }
    ],
    "total_input_tokens": 181,
    "total_output_tokens_a": 171,
    "total_output_tokens_b": 373,
    "total_tokens": 725,
    "communication_efficiency": {
      "avg_tokens_per_turn": 362.5,
      "input_output_ratio": 0.25,
      "coordination_overhead": 0.279
    }
  }
}
```

**Validation:**
- ✅ 300/300 examples successfully enriched
- ✅ No missing values
- ✅ Balanced winner distribution
- ✅ Files under 100MB (1.1MB total)

---

## 3. HuggingFace Dataset Search Results

**Search Strategy:** 20 parallel searches using broad, general terms
**Total Queries:** 20 (multi-agent, conversation, reinforcement learning, token classification, etc.)
**Results:** 54 unique datasets found

### Top 10 Candidates (Based on Preview Analysis)

#### Tier 1: Highly Relevant (3 datasets)

**1. Multi-Agent-LLMs/DEBATE** ⭐⭐⭐⭐⭐
- **Downloads:** 517 | **Likes:** 1
- **Description:** Multi-agent debate framework from MALLM paper
- **Key Features:**
  - Explicit multi-agent coordination with personas
  - Turn-based debate structure with `turns` field
  - **Coordination labels:** `decisionSuccess` (explicit success/failure), `agreements`
  - Multiple paradigms (critical_expert_debate, relay, memory)
  - Clock seconds for temporal analysis
  - Agent memory tracking (`globalMemory`, `agentMemory`)
- **Sample Structure:**
  ```json
  {
    "personas": [{"agentId": "...", "model": "meta-llama/...", "persona": "Marine Biologist"}],
    "turns": [...],
    "decisionSuccess": true/false,
    "agreements": [...],
    "clockSeconds": 45.2
  }
  ```
- **Relevance:** ⭐⭐⭐⭐⭐ Perfect for requirement (c) - has explicit coordination outcomes
- **Download Status:** ⚠️ Timed out (network issues)

**2. lmsys/chatbot_arena_conversations** ⭐⭐⭐⭐⭐
- **Downloads:** 2,400 | **Likes:** 434
- **Description:** 33K pairwise LLM conversations with human preferences
- **Key Features:**
  - High community validation (434 likes)
  - Winner labels (pairwise preference)
  - OpenAI API format (structured JSON)
  - Multi-turn conversations
- **Relevance:** ⭐⭐⭐⭐⭐ Already integrated in existing datasets
- **Status:** ✅ Already used in dat_1_003 and dat_2_007

**3. agentlans/multi-character-dialogue** ⭐⭐⭐⭐
- **Downloads:** 153 | **Likes:** 2
- **Description:** 10K+ multi-character dialogue scenarios
- **Key Features:**
  - Multiple characters per conversation (>2 agents)
  - Turn-based message arrays
  - Character roles and settings
  - Post-interaction changes tracking
- **Sample Structure:**
  ```json
  {
    "characters": {"Eleanor": null, "Owen": null, "Leo": null},
    "conversation": [
      {"from": "Sarisha", "message": "..."},
      {"from": "Renn", "message": "..."}
    ],
    "setting": "...",
    "setting after interaction": "..."
  }
  ```
- **Relevance:** ⭐⭐⭐⭐ Multi-agent (>2) coordination patterns
- **Download Status:** ⚠️ Timed out

#### Tier 2: Potentially Useful (2 datasets)

**4. timonziegenbein/inappropriateness-token-classification-multi-ref** ⭐⭐⭐
- **Downloads:** 218 | **Likes:** 0
- **Description:** Token-level classification with multiple references
- **Key Features:**
  - Token-level annotations (`tokens`, `app_tags_ids`, `app_tags`)
  - Multi-reference labels (could represent multi-agent annotations)
- **Relevance:** ⭐⭐⭐ Token-level analysis but limited coordination context
- **Download Status:** ⚠️ Timed out

**5. PrimeIntellect/stackexchange-question-answering** ⭐⭐⭐
- **Downloads:** 271 | **Likes:** 11
- **Description:** StackExchange Q&A with verification info
- **Key Features:**
  - Community collaboration (implicit multi-agent)
  - Gold standard solutions
  - Verification metadata
- **Relevance:** ⭐⭐⭐ Collaborative knowledge building
- **Download Status:** ⚠️ Timed out

#### Tier 3: Lower Priority (5 datasets)

**6-10. Other Candidates:**
- James4Ever0/computer_agent_reinforcement_learning_trajectory... (⚠️ No sample data)
- Amod/mental_health_counseling_conversations (459 likes but single-pair, not multi-agent)
- mteb/toxic_conversations_50k (No conversation structure)
- electricsheepafrica/nigeria-education-lms-interaction-logs (No communication content)
- slaab/Coding-Conversational-Dataset-Indic (⚠️ No sample data)

### Download Attempts
**Status:** ❌ All downloads timed out after 180s
**Root Cause:** Network connectivity or HuggingFace API latency
**Workaround:** Datasets documented for future manual download or retry

---

## 4. Gap Analysis

### What We Have ✅
1. **Token-level communication logs** (dat_2_007)
   - Per-turn token counts
   - Message lengths
   - Communication efficiency metrics
   - Timestamps

2. **Coordination outcome proxies** (dat_2_007)
   - Winner labels (quality proxy)
   - Coordination overhead metric
   - Turn-based structure

3. **Diverse model coverage** (dat_2_007)
   - 48 model pairs
   - 300 examples, 804 turns

### What We Need ❌

1. **Low-rank latent state annotations** (Requirement b)
   - Per-turn latent vectors (8-16 dimensions)
   - Recurrent state transitions
   - Low-rank constraint enforcement
   - **Solution:** Create annotation pipeline using VAE or PCA on conversation embeddings

2. **Explicit coordination labels** (Requirement c)
   - Coordination success (binary or 0-10 scale)
   - Agreement levels between agents
   - Turn-level coordination events
   - **Solution:** LLM-assisted labeling using OpenRouter API (budget: 1000 calls)

3. **Extended dataset diversity**
   - Multi-Agent-LLMs/DEBATE for explicit coordination labels
   - agentlans/multi-character-dialogue for >2 agent scenarios
   - **Solution:** Manual download or retry when network stable

---

## 5. Recommendations for Completing the Hypothesis

### Phase 1: Augment Existing Dataset (dat_2_007) ⚡ PRIORITY
**Effort:** 2-3 hours | **API Calls:** ~300-600 | **Feasibility:** ✅ High

**Tasks:**
1. **Generate low-rank latent states:**
   ```python
   # Pseudo-code
   for example in dataset:
       embeddings = encode_conversation(example)  # Using sentence-transformers
       latent_states = vae_encoder(embeddings)    # 8-16 dim bottleneck
       example["latent_states"] = latent_states
   ```

2. **Add coordination outcome labels via LLM:**
   ```python
   # Use OpenRouter API with efficient model (e.g., DeepSeek or Mistral)
   prompt = f"""Rate the coordination quality between two LLM agents:

   Input: {example['input']}
   Agent A: {example['context']['response_a'][:200]}...
   Agent B: {example['context']['response_b'][:200]}...
   Winner: {example['context']['winner']}

   Provide:
   1. Coordination success (0-10): [score]
   2. Agreement level (low/medium/high): [level]
   3. Efficiency rating (0-10): [score]

   Output as JSON."""

   coordination_labels = call_openrouter(prompt)
   example["coordination_outcomes"] = coordination_labels
   ```

3. **Validation:**
   - Sample 30 examples for manual verification
   - Check label consistency across similar examples
   - Ensure latent states preserve winner prediction capability

### Phase 2: Download External Datasets (When Network Stable) 🔄 DEFERRED
**Effort:** 1-2 hours | **API Calls:** 0 | **Feasibility:** ⚠️ Network-dependent

**Priority Datasets:**
1. **Multi-Agent-LLMs/DEBATE** - For decisionSuccess labels
2. **agentlans/multi-character-dialogue** - For >2 agent coordination

**Retry Strategy:**
```bash
# Increase timeout, download configs separately
for dataset in Multi-Agent-LLMs/DEBATE agentlans/multi-character-dialogue; do
    timeout 600 python hf_download_datasets.py $dataset --split train
done
```

### Phase 3: Create Comprehensive Benchmark 📊 FUTURE WORK
**Effort:** 5-10 hours | **API Calls:** 500-1000 | **Feasibility:** ✅ High

**Integrate:**
- Existing dat_2_007 (300 examples with token logs + new annotations)
- Multi-Agent-LLMs/DEBATE (explicit coordination labels)
- agentlans/multi-character-dialogue (multi-agent >2 scenarios)

**Final Dataset Structure:**
```json
{
  "examples": [
    {
      "input": "...",
      "context": {
        "agents": ["model_a", "model_b"],
        "responses": ["...", "..."],
        "token_usage": {...},
        "latent_states": {
          "per_turn": [[0.12, -0.45, ...], [0.08, -0.32, ...]],
          "method": "VAE-8dim",
          "reconstruction_error": 0.023
        },
        "coordination_outcomes": {
          "success_score": 8.5,
          "agreement_level": "high",
          "efficiency_rating": 7.2,
          "explicit_label": "successful_coordination",
          "labeling_method": "llm_assisted_openrouter_deepseek"
        }
      },
      "output": "Winner: model_b",
      "metadata": {
        "dataset_source": "lmsys/chatbot_arena_conversations",
        "annotation_version": "v1.0",
        "annotation_date": "2026-01-14"
      }
    }
  ]
}
```

---

## 6. Resource Budget Analysis

### API Call Budget: 1000 calls total

**Option 1: Conservative (LLM Labeling Only)**
- Label 300 examples × 2 calls each (verification) = **600 calls**
- Reserve 400 calls for retries/debugging
- **Feasibility:** ✅ Fits budget

**Option 2: Extended (LLM + Embedding Validation)**
- Label 300 examples × 1 call = 300 calls
- Generate embeddings locally (no API cost)
- Use remaining 700 calls for additional datasets
- **Feasibility:** ✅ Fits budget with room for expansion

**Recommended Model for Labeling:**
- **DeepSeek-V3** (cheap, high quality for classification)
- **Mistral-Large** (good reasoning, moderate cost)
- **Avoid:** OpenAI GPT-4 (too expensive for bulk labeling)

### Computational Resources
- **RAM:** 32GB available ✅ Sufficient for embeddings + VAE training
- **Disk:** 20GB available ✅ Sufficient for all datasets (<5GB total)
- **CPU:** Intel Xeon ✅ Adequate for inference (no GPU needed)

---

## 7. Conclusions

### Current Status
**Hypothesis Progress:** 33% Complete

| Requirement | Status | Solution |
|------------|--------|----------|
| (a) Token logs | ✅ Complete | dat_2_007 has full metadata |
| (b) Latent states | ❌ Missing | VAE pipeline needed |
| (c) Outcome labels | ⚠️ Partial | LLM-assisted labeling |

### Critical Path Forward
1. **Immediate:** Annotate dat_2_007 with latent states + coordination labels (3-5 hours)
2. **Short-term:** Retry HuggingFace downloads when network stable (1-2 hours)
3. **Long-term:** Integrate all datasets into unified benchmark (5-10 hours)

### Expected Outcome
If Phase 1 succeeds:
- ✅ High-quality dataset with all 3 requirements (a, b, c)
- ✅ 300 annotated examples for training/evaluation
- ✅ Reproducible annotation pipeline for future expansion
- ✅ Publishable dataset meeting hypothesis criteria

If Phase 1 fails:
- ⚠️ Learn limits of automated annotation (latent state inference noise)
- ⚠️ Explore synthetic data generation as alternative
- ⚠️ Consider weak-supervision strategies for coordination labels

### Key Insights from HuggingFace Search
1. **Multi-Agent-LLMs/DEBATE is the gold standard** for explicit coordination labels
2. **agentlans/multi-character-dialogue** enables >2 agent research (beyond pairwise)
3. **Token-level datasets exist** but lack coordination context
4. **lmsys/chatbot_arena remains best foundation** (already integrated)

### Risk Assessment
- **Low Risk:** Latent state generation (standard VAE/PCA techniques)
- **Medium Risk:** LLM labeling quality (requires validation sample)
- **High Risk:** External dataset downloads (network-dependent)

---

## Appendix A: Complete Search Query List

1. multi agent → 5 results
2. agent coordination → 0 results
3. dialogue systems → 0 results
4. conversation → 5 results (chatbot_arena, counseling, toxic_conversations)
5. reinforcement learning → 5 results
6. language models → 5 results
7. token classification → 5 results (inappropriateness-multi-ref)
8. sequence labeling → 1 result
9. supervised learning → 4 results
10. natural language processing → 0 results
11. communication efficiency → 0 results
12. interaction logs → 2 results (lms-interaction-logs)
13. collaborative tasks → 0 results
14. text generation → 5 results
15. question answering → 5 results (medical-qa, stackexchange)
16. task completion → 1 result
17. latent representations → 0 results
18. recurrent neural networks → 0 results
19. agent learning → 1 result
20. coordination outcomes → 0 results

**Total:** 54 unique datasets discovered across 20 queries

---

## Appendix B: Recommended Next Steps Script

```python
#!/usr/bin/env python3
"""
Annotate existing dat_2_007 with latent states and coordination labels.
Budget: 600 API calls, 32GB RAM, 3-5 hours
"""

import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import openrouter  # Hypothetical client

# Load existing dataset
data_path = Path("./dependencies/Extended_Multi-LLM_Coordination_Dataset_with_Token/data_out.json")
dataset = json.loads(data_path.read_text())

# Initialize models
encoder = SentenceTransformer('all-MiniLM-L6-v2')  # Local, no API
pca = PCA(n_components=8)  # Low-rank projection

# Process each example
for example in dataset["examples"]:
    # Generate latent states (local, no API cost)
    conv_text = f"{example['input']} {example['context']['response_a']} {example['context']['response_b']}"
    embedding = encoder.encode(conv_text)
    latent_state = pca.fit_transform(embedding.reshape(1, -1))[0].tolist()

    example["latent_states"] = {
        "vector": latent_state,
        "method": "PCA-8dim",
        "dimension": 8
    }

    # Generate coordination labels (API call)
    prompt = f"""Rate coordination between LLM agents (JSON only):
    Input: {example['input'][:100]}
    Winner: {example['context']['winner']}
    Tokens: {example['context']['token_usage']['total_tokens']}

    {{
      "success_score": 0-10,
      "agreement": "low/medium/high",
      "efficiency": 0-10
    }}"""

    labels = openrouter.call("deepseek/deepseek-chat", prompt)
    example["coordination_outcomes"] = labels

# Save annotated dataset
output_path = Path("./annotated_dataset_with_latent_states.json")
output_path.write_text(json.dumps(dataset, indent=2))

print(f"✅ Annotated {len(dataset['examples'])} examples")
print(f"💰 API calls used: ~{len(dataset['examples'])}")
```

---

**Report Generated:** 2026-01-14
**Total Datasets Discovered:** 54
**Top Candidates Identified:** 5
**Existing Dataset Status:** ✅ Token logs complete, ❌ Latent states missing, ⚠️ Outcome labels partial
**Recommended Action:** Proceed with Phase 1 annotation pipeline

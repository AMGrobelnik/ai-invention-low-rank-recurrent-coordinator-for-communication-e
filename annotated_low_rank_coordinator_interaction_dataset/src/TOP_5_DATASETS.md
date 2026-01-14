# Top 5 HuggingFace Datasets for Multi-LLM Agent Coordination

**Selection Criteria:** Data quality, coordination labels, multi-agent structure, token-level features

---

## 1. Multi-Agent-LLMs/DEBATE ⭐⭐⭐⭐⭐

**HuggingFace ID:** `Multi-Agent-LLMs/DEBATE`
**Downloads:** 517 | **Likes:** 1

### Why This Dataset
- **ONLY dataset with explicit coordination success labels** (`decisionSuccess: true/false`)
- Has `agreements` field tracking inter-agent consensus
- Multiple coordination paradigms (debate, relay, memory)
- Turn-based structure with temporal tracking (`clockSeconds`)

### Key Fields
```json
{
  "exampleId": "uuid",
  "instruction": "task description",
  "personas": [
    {"agentId": "...", "model": "meta-llama/Llama-3.3-70B", "persona": "Marine Biologist"}
  ],
  "turns": [...],
  "decisionSuccess": true,
  "agreements": [...],
  "clockSeconds": 45.2,
  "globalMemory": {...},
  "agentMemory": {...}
}
```

### Use Cases
- **Direct training** of coordination success classifiers
- Benchmark for low-rank coordinator performance
- Multi-paradigm evaluation (debate vs relay vs memory)

### Download Status
⚠️ **Timed out** - Retry with: `hf_download_datasets.py Multi-Agent-LLMs/DEBATE --split train`

---

## 2. lmsys/chatbot_arena_conversations ⭐⭐⭐⭐⭐

**HuggingFace ID:** `lmsys/chatbot_arena_conversations`
**Downloads:** 2,400 | **Likes:** 434

### Why This Dataset
- **Highest quality** - 434 likes, real user queries, human judgments
- 33K pairwise LLM conversations with winner labels
- Already integrated in existing dat_1_003 and dat_2_007
- Multi-turn structure (avg 2.7 turns per conversation)

### Key Fields
```json
{
  "question_id": "uuid",
  "model_a": "chatglm-6b",
  "model_b": "koala-13b",
  "winner": "model_a|model_b|tie",
  "conversation_a": [{"role": "user", "content": "..."}, ...],
  "conversation_b": [{"role": "assistant", "content": "..."}, ...],
  "turn": 1,
  "language": "English",
  "tstamp": 1682351591.1322
}
```

### Use Cases
- Foundation for token-level analysis (already enriched in dat_2_007)
- Winner labels provide quality ground truth
- Diverse model pairs (48 unique combinations)

### Download Status
✅ **Already integrated** in existing dependency datasets

---

## 3. agentlans/multi-character-dialogue ⭐⭐⭐⭐

**HuggingFace ID:** `agentlans/multi-character-dialogue`
**Downloads:** 153 | **Likes:** 2

### Why This Dataset
- **Multi-agent (>2 agents)** - Unlike pairwise datasets
- 10,000+ dialogue scenarios with 3-5 characters each
- Turn-based conversation arrays
- Character roles and setting dynamics

### Key Fields
```json
{
  "setting": "description of environment",
  "characters": {"Eleanor": null, "Owen": null, "Leo": null},
  "conversation": [
    {"from": "Sarisha", "message": "dialogue text"},
    {"from": "Renn", "message": "dialogue text"}
  ],
  "setting after interaction": "post-conversation changes",
  "fact": "key takeaway",
  "tag": "genre/category"
}
```

### Use Cases
- Extend beyond pairwise coordination (3+ agents)
- Complex interaction patterns
- Character role analysis in coordination

### Download Status
⚠️ **Timed out** - Retry with: `hf_download_datasets.py agentlans/multi-character-dialogue --split train`

---

## 4. timonziegenbein/inappropriateness-token-classification-multi-ref ⭐⭐⭐

**HuggingFace ID:** `timonziegenbein/inappropriateness-token-classification-multi-ref`
**Downloads:** 218 | **Likes:** 0

### Why This Dataset
- **Token-level annotations** (not just document-level)
- **Multi-reference labels** (multiple annotators = implicit multi-agent agreement)
- Token arrays with per-token tags

### Key Fields
```json
{
  "id": 1466,
  "tokens": ["You", "rather", "die"],
  "app_tags_ids": [3, 4, 0],
  "app_tags": ["tag1", "tag2", "tag3"],
  "user": "annotator_id"
}
```

### Use Cases
- Token-level coordination analysis
- Multi-reference as proxy for agent agreement
- Fine-grained annotation patterns

### Download Status
⚠️ **Timed out** - Retry with: `hf_download_datasets.py timonziegenbein/inappropriateness-token-classification-multi-ref --split train`

---

## 5. PrimeIntellect/stackexchange-question-answering ⭐⭐⭐

**HuggingFace ID:** `PrimeIntellect/stackexchange-question-answering`
**Downloads:** 271 | **Likes:** 11

### Why This Dataset
- **Collaborative Q&A** represents implicit coordination
- Gold standard solutions for quality evaluation
- Verification metadata for outcome assessment
- High-quality source (PrimeIntellect curation)

### Key Fields
```json
{
  "source": "stackexchange",
  "task_type": "llm_judgeable_groundtruth_similarity",
  "in_source_id": 5664094,
  "prompt": "question text",
  "gold_standard_solution": "verified answer",
  "verification_info": {...},
  "metadata": {...}
}
```

### Use Cases
- Community coordination patterns (implicit multi-agent)
- Quality benchmarking with gold standards
- Verification-based outcome labeling

### Download Status
⚠️ **Timed out** - Retry with: `hf_download_datasets.py PrimeIntellect/stackexchange-question-answering --split train`

---

## Comparison Matrix

| Dataset | Agents | Explicit Coord. Labels | Token-Level | Multi-Turn | Status |
|---------|--------|------------------------|-------------|------------|--------|
| **DEBATE** | 2-5 | ✅ decisionSuccess | ⚠️ No | ✅ Yes | ⚠️ Timeout |
| **chatbot_arena** | 2 | ⚠️ winner only | ✅ Yes (dat_2_007) | ✅ Yes | ✅ Integrated |
| **multi-character** | 3-5 | ❌ No | ❌ No | ✅ Yes | ⚠️ Timeout |
| **token-classification** | 1 (multi-ref) | ❌ No | ✅ Yes | ❌ No | ⚠️ Timeout |
| **stackexchange** | 1 (implicit) | ⚠️ gold standard | ❌ No | ❌ No | ⚠️ Timeout |

---

## Download Instructions

### Retry Failed Downloads
```bash
cd /home/adrian/projects/ai-inventor/.claude/skills/hf-datasets

# Option 1: Sequential with longer timeout
for dataset in \
  "Multi-Agent-LLMs/DEBATE" \
  "agentlans/multi-character-dialogue" \
  "timonziegenbein/inappropriateness-token-classification-multi-ref" \
  "PrimeIntellect/stackexchange-question-answering"; do
    echo "Downloading: $dataset"
    timeout 600 bash -c "source scripts/.venv/bin/activate && python scripts/hf_download_datasets.py '$dataset' --split train"
    sleep 5
done

# Option 2: Parallel with GNU parallel (faster but may timeout again)
parallel -j 2 -k --group --will-cite ::: \
  "source scripts/.venv/bin/activate && python scripts/hf_download_datasets.py Multi-Agent-LLMs/DEBATE --split train" \
  "source scripts/.venv/bin/activate && python scripts/hf_download_datasets.py agentlans/multi-character-dialogue --split train"
```

### Expected Output Locations
- `temp/datasets/full_Multi-Agent-LLMs_DEBATE_*_train.json`
- `temp/datasets/mini_Multi-Agent-LLMs_DEBATE_*_train.json`
- `temp/datasets/preview_Multi-Agent-LLMs_DEBATE_*_train.json`

---

## Priority Ranking for Next Steps

1. **CRITICAL:** Download Multi-Agent-LLMs/DEBATE
   - Only dataset with explicit coordination success labels
   - Directly satisfies hypothesis requirement (c)

2. **HIGH:** Download agentlans/multi-character-dialogue
   - Extends research to 3+ agent scenarios
   - 10K examples = good augmentation volume

3. **MEDIUM:** Annotate existing dat_2_007
   - Generate latent states with VAE/PCA
   - LLM-assisted coordination labels
   - Can proceed without new downloads

4. **LOW:** Download token-classification and stackexchange
   - Supplementary datasets for specific analyses
   - Not critical for hypothesis completion

---

**Summary:** 5 datasets identified, 1 already integrated, 4 pending download retry
**Recommended Action:** Focus on Multi-Agent-LLMs/DEBATE download + dat_2_007 annotation

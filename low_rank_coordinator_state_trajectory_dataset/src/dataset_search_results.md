# Dataset Search Results for Multi-LLM Agent Hidden-State Trajectory Collection

## Search Summary
Executed 20 parallel searches across HuggingFace Hub for datasets relevant to collecting hidden-state trajectories during multi-LLM agent interactions.

## Key Findings

### Highly Relevant Datasets

#### 1. Agent Interaction & Multi-Agent Systems
- **Z-Edgar/Agent-IPI-Structured-Interaction-Datasets** (44 downloads, 1 like)
  - Indirect Prompt Injection in Agent Structured Interaction
  - 100K-1M records, JSON format

- **achiepatricia/han-multi-agent-interaction-dataset-v1** (0 downloads, 0 likes)
  - Humanoid Multi-Agent Interaction Dataset
  - Supports coordination and collaboration models
  - <1K records, JSON format, MIT license

- **deepgo/Interaction_Agent_Dataset_V0.1** (11 downloads, 0 likes)
  - Multi-modal English/Chinese dataset
  - Multi-turn dialogue and emotion-aware AI-Agent

- **Bill12138/Task-and-Motion-Re-Planning-for-Multi-Agent-Systems** (18 downloads, 0 likes)
  - Re-planning data for multi-agent systems
  - MIT license

#### 2. Hidden States & Latent Representations
- **austindavis/chess-gpt2-hiddenstates-768** (1,491 downloads, 0 likes)
  - 1M-10M hidden state vectors from GPT-2 forward passes
  - Parquet format, tabular + text modality

- **austindavis/chess-gpt2-hiddenstates-512** (659 downloads, 0 likes)
  - 120k hidden state vectors from GPT-2
  - UCI chess move sequences
  - 1M-10M records, Parquet format

- **Hemabhushan/capstone_mlm_hidden_states** (918 downloads, 0 likes)
  - 100K-1M records, Parquet format

- **s3prl/sample_hidden_states** (672 downloads, 0 likes)

#### 3. Trajectory Data
- **mlfoundations-cua-dev/agent-trajectory-data** (212 downloads, 1 like)
  - Agent trajectory data with image modality

- **d3LLM/trajectory_data_llada_32** (403 downloads, 2 likes)
  - Text generation trajectories
  - 10K-100K records, Arrow format

- **d3LLM/trajectory_data_dream_32** (307 downloads, 0 likes)
  - 100K-1M records, Arrow format, tabular

- **cywan/StreamVLN-Trajectory-Data** (510 downloads, 8 likes)
  - Vision-and-Language Navigation trajectories
  - >1T records

#### 4. State Prediction
- **LangAGI-Lab/human_eval-next_state_prediction_w_gpt4o** (6 downloads, 0 likes)
  - <1K records, Parquet format

- **LangAGI-Lab/human_eval-next_state_prediction** (3 downloads, 0 likes)
  - <1K records, Parquet format

#### 5. Interaction Logs
- **electricsheepafrica/nigeria-education-lms-interaction-logs** (31 downloads, 0 likes)
  - LMS interaction logs with activity types and timestamps
  - 100K-1M records, MIT license

- **electricsheepafrica/nigerian_retail_and_ecommerce_social_media_interaction_logs** (26 downloads, 0 likes)
  - Social media interaction logs
  - 100K-1M records, GPL license

#### 6. Time Series & Sequential Data
- **thuml/Time-Series-Library** (21,135 downloads, 2 likes)
  - Deep time series analysis library
  - 1M-10M records, CC-BY-4.0 license

- **AutonLab/Timeseries-PILE** (2,172 downloads, 35 likes)
  - Large collection from diverse domains
  - 5+ public time-series datasets
  - MIT license

- **tomg-group-umd/gemstones_data_order_sequential** (378 downloads, 0 likes)
  - 100M-1B records, Parquet format
  - Reprocessed Dolma v1.7 dataset

#### 7. Performance Metrics
- **EMPLOYEE_PERFORMANCE_METRICS** (14 downloads, 0 likes)
  - <1K records, CSV format

- **ismielabir/Sorting-Algorithms-Performance-Metrics** (8 downloads, 0 likes)
  - Benchmark and performance analysis
  - MIT license, CSV format

#### 8. Token Statistics
- **FractureSSR/first_order_token_statistics** (4 downloads, 0 likes)
  - MIT license

#### 9. Behavioral Data
- **syncora/developer-productivity-simulated-behavioral-data** (25 downloads, 84 likes!)
  - Synthetic behavioral + cognitive simulation
  - 1K-10K records, Apache 2.0 license
  - Models behavioral and cognitive dynamics

- **Seuneedhi/Video-Games-Behavioral-Addiction-Dataset** (61 downloads, 1 like)
  - <1K records, image format

#### 10. Task Completion
- **flaitenberger/synthetic_pattern_completion_task** (13 downloads, 0 likes)
  - 1M-10M records, Parquet format

## Search Queries with Zero Results
The following 7 queries returned no datasets:
1. conversation modeling
2. dialogue systems
3. sequence modeling
4. recurrent networks
5. latent representations
6. communication efficiency
7. coordination tasks
8. system monitoring

## Recommended Next Steps

### Top 5 Datasets for Further Investigation:
1. **achiepatricia/han-multi-agent-interaction-dataset-v1** - Most directly relevant to multi-agent coordination
2. **austindavis/chess-gpt2-hiddenstates-768** - Large dataset of hidden states from LLM
3. **mlfoundations-cua-dev/agent-trajectory-data** - Agent trajectories with potential state info
4. **syncora/developer-productivity-simulated-behavioral-data** - Behavioral simulation (84 likes!)
5. **LangAGI-Lab/human_eval-next_state_prediction** - State prediction focused

### Analysis
- **Hidden state datasets exist** but mostly from single-model contexts (chess GPT-2)
- **Multi-agent interaction datasets exist** but lack hidden-state components
- **No existing dataset** combines multi-LLM coordination + hidden states + token metrics
- **Gap identified**: This validates the hypothesis - there's a clear need for a dataset combining:
  - Multi-LLM agent interactions
  - Low-rank coordinator hidden states
  - Token usage statistics
  - Performance metrics
  - Trajectory sequences

This search confirms the novelty and value of the proposed dataset objective.

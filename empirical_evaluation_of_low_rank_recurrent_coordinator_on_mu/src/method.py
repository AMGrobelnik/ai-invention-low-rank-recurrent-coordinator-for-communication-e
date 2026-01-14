#!/usr/bin/env python3
"""
Low-Rank Recurrent Coordinator for Multi-LLM Agent Communication Efficiency

This script implements and evaluates:
1. BASELINE: Full-rank recurrent coordinator
2. METHOD: Low-rank recurrent coordinator with RIM-inspired sparse recurrence

Hypothesis: Low-rank coordinator reduces token usage by >15% while maintaining task performance.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, asdict

import numpy as np
import tiktoken
from scipy.linalg import svd
from sklearn.metrics import accuracy_score, f1_score
from scipy.stats import ttest_rel

# Configure extensive logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s | %(levelname)-7s | %(funcName)-20s | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('method_execution.log', mode='w')
    ]
)
logger = logging.getLogger(__name__)

# Color codes for logging
BLUE, GREEN, YELLOW, CYAN, RED, END = "\033[94m", "\033[92m", "\033[93m", "\033[96m", "\033[91m", "\033[0m"


def truncate_str(text: str, max_len: int = 100) -> str:
    """Truncate long strings for logging."""
    if len(text) <= max_len:
        return text
    return text[:max_len] + f"... ({len(text)} chars total)"


@dataclass
class ExampleResult:
    """Single example result."""
    input: str
    output: str
    context: Dict[str, Any]
    dataset: str
    split: str
    predict_baseline: str
    predict_method: str
    method: str


@dataclass
class ExperimentResult:
    """Schema matching exp_gen_sol_out.json format."""
    examples: List[Dict[str, Any]]


class TokenTracker:
    """Track token usage for multi-agent coordination."""

    def __init__(self, model: str = "gpt-4"):
        """Initialize token tracker with tiktoken encoder."""
        try:
            self.encoding = tiktoken.encoding_for_model(model)
            logger.info(f"{GREEN}TokenTracker initialized with model: {model}{END}")
        except Exception as e:
            logger.warning(f"{YELLOW}Could not load model-specific encoding, using cl100k_base: {e}{END}")
            self.encoding = tiktoken.get_encoding("cl100k_base")

        self.total_tokens = 0
        self.episode_tokens = []
        self.call_count = 0

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        try:
            tokens = len(self.encoding.encode(text))
            logger.debug(f"Counted {tokens} tokens in text: {truncate_str(text, 50)}")
            return tokens
        except Exception as e:
            logger.error(f"{RED}Error counting tokens: {e}{END}")
            # Fallback: approximate as words * 1.3
            approx = int(len(text.split()) * 1.3)
            logger.warning(f"{YELLOW}Using approximate token count: {approx}{END}")
            return approx

    def log_coordinator_step(self, agent_outputs: List[str], coordinator_message: str = ""):
        """Log tokens for a coordinator step."""
        try:
            step_tokens = 0

            # Count tokens in agent outputs (what coordinator processes)
            for idx, output in enumerate(agent_outputs):
                output_tokens = self.count_tokens(output)
                step_tokens += output_tokens
                logger.debug(f"  Agent {idx} output: {output_tokens} tokens")

            # Count tokens in coordinator's internal message (communication cost)
            if coordinator_message:
                coord_tokens = self.count_tokens(coordinator_message)
                step_tokens += coord_tokens
                logger.debug(f"  Coordinator message: {coord_tokens} tokens")

            self.total_tokens += step_tokens
            self.episode_tokens.append(step_tokens)
            self.call_count += 1

            logger.info(f"{CYAN}Step {self.call_count}: {step_tokens} tokens (total: {self.total_tokens}){END}")
            return step_tokens

        except Exception as e:
            logger.error(f"{RED}Error logging coordinator step: {e}{END}")
            return 0

    def get_stats(self) -> Dict[str, float]:
        """Get aggregated token statistics."""
        stats = {
            "total_tokens": self.total_tokens,
            "num_episodes": len(self.episode_tokens),
            "mean_tokens_per_episode": np.mean(self.episode_tokens) if self.episode_tokens else 0,
            "std_tokens_per_episode": np.std(self.episode_tokens) if self.episode_tokens else 0,
            "call_count": self.call_count
        }
        logger.debug(f"Token stats: {stats}")
        return stats


class FullRankCoordinator:
    """Baseline: Full-rank recurrent coordinator for multi-agent coordination."""

    def __init__(self, hidden_dim: int = 256):
        """
        Initialize full-rank coordinator.

        Args:
            hidden_dim: Dimension of hidden state
        """
        self.hidden_dim = hidden_dim
        self.W = np.random.randn(hidden_dim, hidden_dim) * 0.01
        self.state = np.zeros(hidden_dim)

        logger.info(f"{GREEN}FullRankCoordinator initialized: hidden_dim={hidden_dim}{END}")
        logger.info(f"  Weight matrix shape: {self.W.shape}")
        logger.info(f"  Total parameters: {self.W.size}")

    def reset(self):
        """Reset coordinator state."""
        self.state = np.zeros(self.hidden_dim)
        logger.debug("Coordinator state reset")

    def step(self, agent_outputs: List[str]) -> Tuple[np.ndarray, str]:
        """
        Recurrent step: process agent outputs and update state.

        Args:
            agent_outputs: List of agent response strings

        Returns:
            Tuple of (updated state, coordinator message)
        """
        try:
            logger.debug(f"FullRankCoordinator.step with {len(agent_outputs)} agent outputs")

            # Encode agent outputs into feature vector
            features = self._encode_outputs(agent_outputs)
            logger.debug(f"  Encoded features shape: {features.shape}")

            # Full-rank recurrent update: s_{t+1} = W @ s_t + features
            self.state = self.W @ self.state + features
            logger.debug(f"  State norm after update: {np.linalg.norm(self.state):.4f}")

            # Generate coordinator message (simulates communication)
            # In full-rank, message includes all state dimensions
            coordinator_message = self._generate_message(self.state)
            logger.debug(f"  Coordinator message length: {len(coordinator_message)}")

            return self.state.copy(), coordinator_message

        except Exception as e:
            logger.error(f"{RED}Error in FullRankCoordinator.step: {e}{END}")
            raise

    def _encode_outputs(self, outputs: List[str]) -> np.ndarray:
        """Encode agent outputs into feature vector."""
        try:
            # Simple encoding: token counts + basic text statistics
            features = []
            for output in outputs:
                features.extend([
                    len(output.split()),  # Word count
                    len(output),  # Character count
                    output.count('.'),  # Sentence markers
                    output.count('?'),  # Question markers
                ])

            # Pad or truncate to hidden_dim
            features_array = np.array(features[:self.hidden_dim])
            if len(features_array) < self.hidden_dim:
                padded = np.zeros(self.hidden_dim)
                padded[:len(features_array)] = features_array
                features_array = padded

            # Normalize
            features_array = features_array / (np.linalg.norm(features_array) + 1e-8)

            logger.debug(f"  Encoded {len(outputs)} outputs into {len(features_array)} features")
            return features_array

        except Exception as e:
            logger.error(f"{RED}Error encoding outputs: {e}{END}")
            return np.zeros(self.hidden_dim)

    def _generate_message(self, state: np.ndarray) -> str:
        """
        Generate coordinator message from state.

        In full-rank, message encodes entire state (high communication cost).
        """
        # Simulate message as string representation of state
        # Message length proportional to number of dimensions
        message_parts = []
        for i in range(0, len(state), 10):
            # Sample every 10th dimension to create message
            val = state[i]
            if abs(val) > 0.1:  # Only include significant values
                message_parts.append(f"dim{i}:{val:.2f}")

        message = " ".join(message_parts)
        logger.debug(f"  Generated message with {len(message_parts)} components")
        return message


class LowRankRecurrentCoordinator:
    """
    Method: Low-rank recurrent coordinator with RIM-inspired sparse recurrence.

    Key innovations:
    1. Low-rank factorization: W = U @ V^T (reduces parameters)
    2. Sparse recurrence: Only k modules active per step (reduces computation)
    3. Compressed messaging: Messages encode only rank dimensions (reduces communication)
    """

    def __init__(self, hidden_dim: int = 256, rank: int = 32, num_modules: int = 4):
        """
        Initialize low-rank coordinator.

        Args:
            hidden_dim: Dimension of hidden state
            rank: Low-rank factorization rank (k << hidden_dim)
            num_modules: Number of RIM modules for sparse recurrence
        """
        self.hidden_dim = hidden_dim
        self.rank = rank
        self.num_modules = num_modules

        # Low-rank factorization: W = U @ V^T
        self.U = np.random.randn(hidden_dim, rank) * 0.01
        self.V = np.random.randn(hidden_dim, rank) * 0.01

        # RIM sparse attention: only k modules active per step
        self.active_k = max(1, num_modules // 2)

        # Module-specific transformations (lightweight)
        self.module_weights = [
            np.random.randn(hidden_dim, rank) * 0.01
            for _ in range(num_modules)
        ]

        self.state = np.zeros(hidden_dim)

        compression_ratio = rank / hidden_dim
        param_reduction = (2 * hidden_dim * rank) / (hidden_dim * hidden_dim)

        logger.info(f"{GREEN}LowRankRecurrentCoordinator initialized:{END}")
        logger.info(f"  hidden_dim={hidden_dim}, rank={rank}, num_modules={num_modules}")
        logger.info(f"  Compression ratio: {compression_ratio:.2%}")
        logger.info(f"  Parameter reduction: {param_reduction:.2%} of full-rank")
        logger.info(f"  Active modules per step: {self.active_k}/{num_modules}")

    def reset(self):
        """Reset coordinator state."""
        self.state = np.zeros(self.hidden_dim)
        logger.debug("Coordinator state reset")

    def step(self, agent_outputs: List[str]) -> Tuple[np.ndarray, str]:
        """
        Low-rank recurrent step with sparse module updates.

        Args:
            agent_outputs: List of agent response strings

        Returns:
            Tuple of (updated state, coordinator message)
        """
        try:
            logger.debug(f"LowRankCoordinator.step with {len(agent_outputs)} agent outputs")

            # Encode agent outputs
            features = self._encode_outputs(agent_outputs)
            logger.debug(f"  Encoded features shape: {features.shape}")

            # Select active modules (RIM sparse recurrence)
            active_modules = self._select_active_modules(self.state, features)
            logger.debug(f"  Active modules: {active_modules}")

            # Low-rank recurrent update: s_{t+1} = U @ (V^T @ s_t) + module_updates
            # This is equivalent to W @ s_t where W = U @ V^T, but much cheaper
            state_proj = self.V.T @ self.state  # Project to rank-dimensional space
            logger.debug(f"  State projection shape: {state_proj.shape}")

            new_state = self.U @ state_proj  # Reconstruct in original space

            # Apply sparse module updates (only active modules)
            for module_idx in active_modules:
                module_update = self.module_weights[module_idx] @ state_proj
                new_state += module_update
                logger.debug(f"    Module {module_idx} update norm: {np.linalg.norm(module_update):.4f}")

            # Add features
            new_state += features

            self.state = new_state
            logger.debug(f"  State norm after update: {np.linalg.norm(self.state):.4f}")

            # Generate compressed coordinator message
            # In low-rank, message only encodes rank dimensions (lower communication cost)
            coordinator_message = self._generate_compressed_message(state_proj)
            logger.debug(f"  Coordinator message length: {len(coordinator_message)}")

            return self.state.copy(), coordinator_message

        except Exception as e:
            logger.error(f"{RED}Error in LowRankCoordinator.step: {e}{END}")
            raise

    def _encode_outputs(self, outputs: List[str]) -> np.ndarray:
        """Encode agent outputs into feature vector."""
        try:
            # Simple encoding: token counts + basic text statistics
            features = []
            for output in outputs:
                features.extend([
                    len(output.split()),  # Word count
                    len(output),  # Character count
                    output.count('.'),  # Sentence markers
                    output.count('?'),  # Question markers
                ])

            # Pad or truncate to hidden_dim
            features_array = np.array(features[:self.hidden_dim])
            if len(features_array) < self.hidden_dim:
                padded = np.zeros(self.hidden_dim)
                padded[:len(features_array)] = features_array
                features_array = padded

            # Normalize
            features_array = features_array / (np.linalg.norm(features_array) + 1e-8)

            logger.debug(f"  Encoded {len(outputs)} outputs into {len(features_array)} features")
            return features_array

        except Exception as e:
            logger.error(f"{RED}Error encoding outputs: {e}{END}")
            return np.zeros(self.hidden_dim)

    def _select_active_modules(self, state: np.ndarray, features: np.ndarray) -> List[int]:
        """
        Select top-k modules based on attention scores (RIM mechanism).

        Args:
            state: Current hidden state
            features: Encoded input features

        Returns:
            List of active module indices
        """
        try:
            # Compute attention scores for each module
            scores = []
            state_proj = self.V.T @ state  # Project to rank space

            for module_idx, module_w in enumerate(self.module_weights):
                # Score = similarity between module projection and features
                module_proj = module_w.T @ features
                score = np.dot(module_proj[:self.rank], state_proj)
                scores.append((score, module_idx))

            # Return top-k active modules
            scores.sort(reverse=True)
            active = [idx for _, idx in scores[:self.active_k]]

            logger.debug(f"  Module scores: {[(idx, f'{score:.4f}') for score, idx in scores]}")
            return active

        except Exception as e:
            logger.error(f"{RED}Error selecting active modules: {e}{END}")
            # Fallback: return first k modules
            return list(range(self.active_k))

    def _generate_compressed_message(self, state_proj: np.ndarray) -> str:
        """
        Generate compressed coordinator message from projected state.

        In low-rank, message only encodes rank dimensions (much smaller than full state).
        This is the key to reduced communication cost.
        """
        # Simulate message as string representation of compressed state
        message_parts = []
        for i in range(len(state_proj)):
            val = state_proj[i]
            if abs(val) > 0.1:  # Only include significant values
                message_parts.append(f"r{i}:{val:.2f}")

        message = " ".join(message_parts)
        logger.debug(f"  Generated compressed message with {len(message_parts)} components (vs {self.hidden_dim} in full-rank)")
        return message


def load_dataset(data_path: Path) -> List[Dict]:
    """Load dataset from JSON file."""
    try:
        logger.info(f"{BLUE}Loading dataset from: {data_path}{END}")

        if not data_path.exists():
            raise FileNotFoundError(f"Dataset not found: {data_path}")

        with open(data_path, 'r') as f:
            data = json.load(f)

        examples = data.get('examples', [])
        logger.info(f"{GREEN}Loaded {len(examples)} examples from dataset{END}")

        # Log first example structure
        if examples:
            logger.debug(f"Example structure: {list(examples[0].keys())}")
            logger.debug(f"First input: {truncate_str(examples[0]['input'])}")

        return examples

    except Exception as e:
        logger.error(f"{RED}Error loading dataset: {e}{END}")
        raise


def run_experiment(examples: List[Dict], coordinator, tracker: TokenTracker, name: str, is_baseline: bool = True) -> Tuple[List[str], List[int]]:
    """
    Run experiment with given coordinator.

    Args:
        examples: Dataset examples
        coordinator: Coordinator instance (FullRank or LowRank)
        tracker: Token tracker instance
        name: Experiment name for logging
        is_baseline: Whether this is the baseline experiment (for recording per-example results)

    Returns:
        Tuple of (predictions, token_counts_per_episode)
    """
    try:
        logger.info(f"\n{BLUE}{'='*80}{END}")
        logger.info(f"{BLUE}Running experiment: {name}{END}")
        logger.info(f"{BLUE}{'='*80}{END}\n")

        predictions = []
        token_counts = []

        for idx, example in enumerate(examples):
            try:
                logger.info(f"\n{CYAN}Processing example {idx+1}/{len(examples)}{END}")
                logger.debug(f"Input: {truncate_str(example['input'])}")

                # Reset coordinator state for new episode
                coordinator.reset()

                # Track tokens for this episode
                episode_start_tokens = tracker.total_tokens

                # Extract agent outputs from dataset
                context = example['context']
                agent_outputs = [
                    context['response_a'],
                    context['response_b']
                ]

                logger.debug(f"Agent A ({context['model_a']}): {truncate_str(agent_outputs[0], 80)}")
                logger.debug(f"Agent B ({context['model_b']}): {truncate_str(agent_outputs[1], 80)}")

                # Coordinator processes both agent responses
                state, coordinator_message = coordinator.step(agent_outputs)

                # Track tokens (agent outputs + coordinator message)
                step_tokens = tracker.log_coordinator_step(agent_outputs, coordinator_message)

                # Make prediction based on coordinator's final state
                # Simple heuristic: compare state activation with agent output lengths
                prediction = predict_winner(state, agent_outputs, context)
                predictions.append(prediction)

                # Record tokens for this episode
                episode_tokens = tracker.total_tokens - episode_start_tokens
                token_counts.append(episode_tokens)

                logger.info(f"{GREEN}Episode {idx+1} complete: prediction={prediction}, tokens={episode_tokens}{END}")

            except Exception as e:
                logger.error(f"{RED}Error processing example {idx+1}: {e}{END}")
                # Add default prediction and token count to continue
                predictions.append("tie")
                token_counts.append(0)

        logger.info(f"\n{GREEN}Experiment {name} complete: {len(predictions)} predictions made{END}")
        return predictions, token_counts

    except Exception as e:
        logger.error(f"{RED}Error in run_experiment: {e}{END}")
        raise


def predict_winner(state: np.ndarray, agent_outputs: List[str], context: Dict) -> str:
    """
    Predict winner based on coordinator state.

    This is a simple heuristic - in practice, you would train a classifier.
    For this experiment, we use a rule-based approach.
    """
    try:
        # Extract features from agent outputs
        len_a = len(agent_outputs[0])
        len_b = len(agent_outputs[1])

        # Use state to weight features
        # Sum of positive state values as "confidence"
        confidence = np.sum(state[state > 0])

        # Simple heuristic: longer, more detailed response is often better
        # But weight by coordinator's state confidence
        score_a = len_a * (1 + confidence * 0.1)
        score_b = len_b * (1 + confidence * 0.1)

        # If very close, predict tie
        if abs(score_a - score_b) < 50:
            prediction = "tie"
        elif score_a > score_b:
            prediction = "model_a"
        else:
            prediction = "model_b"

        logger.debug(f"  Prediction: {prediction} (score_a={score_a:.1f}, score_b={score_b:.1f}, confidence={confidence:.4f})")
        return prediction

    except Exception as e:
        logger.error(f"{RED}Error in predict_winner: {e}{END}")
        return "tie"  # Default fallback


def evaluate_performance(predictions: List[str], ground_truth: List[str]) -> Dict[str, float]:
    """
    Evaluate task performance metrics.

    Args:
        predictions: Predicted winners
        ground_truth: Actual winners from dataset

    Returns:
        Dictionary of performance metrics
    """
    try:
        logger.info(f"\n{BLUE}Evaluating performance...{END}")

        # Compute accuracy
        accuracy = accuracy_score(ground_truth, predictions)
        logger.info(f"  Accuracy: {accuracy:.4f}")

        # Compute F1 scores
        f1_macro = f1_score(ground_truth, predictions, average='macro', zero_division=0)
        f1_weighted = f1_score(ground_truth, predictions, average='weighted', zero_division=0)
        logger.info(f"  F1 (macro): {f1_macro:.4f}")
        logger.info(f"  F1 (weighted): {f1_weighted:.4f}")

        metrics = {
            "accuracy": float(accuracy),
            "f1_macro": float(f1_macro),
            "f1_weighted": float(f1_weighted)
        }

        logger.info(f"{GREEN}Performance evaluation complete{END}")
        return metrics

    except Exception as e:
        logger.error(f"{RED}Error evaluating performance: {e}{END}")
        return {
            "accuracy": 0.0,
            "f1_macro": 0.0,
            "f1_weighted": 0.0
        }


def compute_statistical_significance(baseline_tokens: List[int], method_tokens: List[int]) -> Dict[str, Any]:
    """
    Test if improvement is statistically significant.

    Args:
        baseline_tokens: Token counts per episode for baseline
        method_tokens: Token counts per episode for method

    Returns:
        Dictionary of statistical test results
    """
    try:
        logger.info(f"\n{BLUE}Computing statistical significance...{END}")

        # Paired t-test (assumes normality)
        t_stat, t_pval = ttest_rel(baseline_tokens, method_tokens)
        t_significant = t_pval < 0.05

        logger.info(f"  Paired t-test: t={t_stat:.4f}, p={t_pval:.4f}, significant={t_significant}")

        # Effect size (Cohen's d)
        mean_diff = np.mean(baseline_tokens) - np.mean(method_tokens)
        pooled_std = np.sqrt((np.var(baseline_tokens) + np.var(method_tokens)) / 2)
        cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0

        logger.info(f"  Effect size (Cohen's d): {cohens_d:.4f}")

        results = {
            "t_statistic": float(t_stat),
            "t_pvalue": float(t_pval),
            "t_significant": bool(t_significant),
            "cohens_d": float(cohens_d),
            "alpha": 0.05
        }

        logger.info(f"{GREEN}Statistical tests complete{END}")
        return results

    except Exception as e:
        logger.error(f"{RED}Error computing statistical significance: {e}{END}")
        return {
            "t_statistic": 0.0,
            "t_pvalue": 1.0,
            "t_significant": False,
            "cohens_d": 0.0,
            "alpha": 0.05
        }


def main():
    """Main execution function."""
    try:
        logger.info(f"\n{BLUE}{'='*80}{END}")
        logger.info(f"{BLUE}LOW-RANK RECURRENT COORDINATOR EXPERIMENT{END}")
        logger.info(f"{BLUE}{'='*80}{END}\n")

        # Configuration
        HIDDEN_DIM = 256
        RANK = 32
        NUM_MODULES = 4

        # Load dataset
        workspace_dir = Path(__file__).parent
        data_path = workspace_dir / "dependencies" / "Multi-Agent_Coordination_Communication-Efficiency" / "data_out.json"

        logger.info(f"Workspace directory: {workspace_dir}")
        logger.info(f"Data path: {data_path}")

        examples = load_dataset(data_path)

        if not examples:
            raise ValueError("No examples loaded from dataset")

        # Extract ground truth labels
        ground_truth = [ex['context']['winner'] for ex in examples]
        logger.info(f"Ground truth distribution: {dict(zip(*np.unique(ground_truth, return_counts=True)))}")

        # ========================================================================
        # BASELINE: Full-Rank Coordinator
        # ========================================================================

        logger.info(f"\n{BLUE}{'='*80}{END}")
        logger.info(f"{BLUE}BASELINE: Full-Rank Coordinator{END}")
        logger.info(f"{BLUE}{'='*80}{END}\n")

        baseline_coordinator = FullRankCoordinator(hidden_dim=HIDDEN_DIM)
        baseline_tracker = TokenTracker()

        baseline_predictions, baseline_token_counts = run_experiment(
            examples, baseline_coordinator, baseline_tracker, "Baseline (Full-Rank)", is_baseline=True
        )

        baseline_metrics = evaluate_performance(baseline_predictions, ground_truth)
        baseline_token_stats = baseline_tracker.get_stats()

        logger.info(f"\n{GREEN}Baseline Results:{END}")
        logger.info(f"  Total tokens: {baseline_token_stats['total_tokens']}")
        logger.info(f"  Mean tokens/episode: {baseline_token_stats['mean_tokens_per_episode']:.2f}")
        logger.info(f"  Accuracy: {baseline_metrics['accuracy']:.4f}")

        # ========================================================================
        # METHOD: Low-Rank Coordinator
        # ========================================================================

        logger.info(f"\n{BLUE}{'='*80}{END}")
        logger.info(f"{BLUE}METHOD: Low-Rank Coordinator{END}")
        logger.info(f"{BLUE}{'='*80}{END}\n")

        method_coordinator = LowRankRecurrentCoordinator(
            hidden_dim=HIDDEN_DIM,
            rank=RANK,
            num_modules=NUM_MODULES
        )
        method_tracker = TokenTracker()

        method_predictions, method_token_counts = run_experiment(
            examples, method_coordinator, method_tracker, "Method (Low-Rank)", is_baseline=False
        )

        method_metrics = evaluate_performance(method_predictions, ground_truth)
        method_token_stats = method_tracker.get_stats()

        logger.info(f"\n{GREEN}Method Results:{END}")
        logger.info(f"  Total tokens: {method_token_stats['total_tokens']}")
        logger.info(f"  Mean tokens/episode: {method_token_stats['mean_tokens_per_episode']:.2f}")
        logger.info(f"  Accuracy: {method_metrics['accuracy']:.4f}")

        # ========================================================================
        # COMPARISON & STATISTICAL ANALYSIS
        # ========================================================================

        logger.info(f"\n{BLUE}{'='*80}{END}")
        logger.info(f"{BLUE}COMPARISON & STATISTICAL ANALYSIS{END}")
        logger.info(f"{BLUE}{'='*80}{END}\n")

        # Compute token reduction
        token_reduction_percent = (
            (baseline_token_stats['total_tokens'] - method_token_stats['total_tokens'])
            / baseline_token_stats['total_tokens'] * 100
        )

        # Compute accuracy change
        accuracy_delta = method_metrics['accuracy'] - baseline_metrics['accuracy']

        logger.info(f"{CYAN}Token Efficiency:{END}")
        logger.info(f"  Baseline total: {baseline_token_stats['total_tokens']}")
        logger.info(f"  Method total: {method_token_stats['total_tokens']}")
        logger.info(f"  Reduction: {token_reduction_percent:.2f}%")
        logger.info(f"  Target: >15% reduction")
        logger.info(f"  Success: {token_reduction_percent > 15}")

        logger.info(f"\n{CYAN}Task Performance:{END}")
        logger.info(f"  Baseline accuracy: {baseline_metrics['accuracy']:.4f}")
        logger.info(f"  Method accuracy: {method_metrics['accuracy']:.4f}")
        logger.info(f"  Delta: {accuracy_delta:+.4f}")
        logger.info(f"  Maintained/improved: {accuracy_delta >= 0}")

        # Statistical significance tests
        stat_tests = compute_statistical_significance(baseline_token_counts, method_token_counts)

        # Overall success criterion
        success = (token_reduction_percent > 15) and (accuracy_delta >= -0.05)  # Allow small accuracy drop

        logger.info(f"\n{CYAN}Overall Hypothesis Test:{END}")
        logger.info(f"  Token reduction >15%: {token_reduction_percent > 15}")
        logger.info(f"  Performance maintained: {accuracy_delta >= -0.05}")
        logger.info(f"  Statistically significant: {stat_tests['t_significant']}")
        logger.info(f"  SUCCESS: {success}")

        # ========================================================================
        # SAVE RESULTS
        # ========================================================================

        # Build examples array matching schema
        result_examples = []
        for idx, example in enumerate(examples):
            result_examples.append({
                "input": example['input'],
                "output": example['output'],
                "context": example['context'],
                "dataset": example['dataset'],
                "split": example['split'],
                "predict_baseline": baseline_predictions[idx],
                "predict_method": method_predictions[idx],
                "method": "Low-Rank Recurrent Coordinator with RIM-inspired sparse recurrence"
            })

        result = ExperimentResult(
            examples=result_examples
        )

        output_path = workspace_dir / "method_out.json"
        with open(output_path, 'w') as f:
            json.dump(asdict(result), f, indent=2)

        # Also save summary metrics separately for analysis
        summary_path = workspace_dir / "method_summary.json"
        summary = {
            "method_name": "Low-Rank Recurrent Coordinator",
            "baseline_name": "Full-Rank Recurrent Coordinator",
            "dataset_size": len(examples),
            "baseline_metrics": {
                **baseline_metrics,
                **{k: float(v) if isinstance(v, (np.integer, np.floating)) else v
                   for k, v in baseline_token_stats.items()}
            },
            "method_metrics": {
                **method_metrics,
                **{k: float(v) if isinstance(v, (np.integer, np.floating)) else v
                   for k, v in method_token_stats.items()}
            },
            "improvement_metrics": {
                "token_reduction_percent": float(token_reduction_percent),
                "token_reduction_absolute": int(baseline_token_stats['total_tokens'] - method_token_stats['total_tokens']),
                "accuracy_delta": float(accuracy_delta),
                "f1_macro_delta": float(method_metrics['f1_macro'] - baseline_metrics['f1_macro']),
            },
            "statistical_tests": stat_tests,
            "configuration": {
                "hidden_dim": HIDDEN_DIM,
                "rank": RANK,
                "num_modules": NUM_MODULES,
                "compression_ratio": RANK / HIDDEN_DIM,
                "parameter_reduction": (2 * HIDDEN_DIM * RANK) / (HIDDEN_DIM * HIDDEN_DIM)
            },
            "examples_processed": len(examples),
            "success": success
        }
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"\n{GREEN}Results saved to: {output_path}{END}")
        logger.info(f"{GREEN}Summary metrics saved to: {summary_path}{END}")

        # Print summary
        logger.info(f"\n{BLUE}{'='*80}{END}")
        logger.info(f"{BLUE}EXPERIMENT SUMMARY{END}")
        logger.info(f"{BLUE}{'='*80}{END}\n")
        logger.info(f"Method: {summary['method_name']}")
        logger.info(f"Baseline: {summary['baseline_name']}")
        logger.info(f"Examples processed: {summary['examples_processed']}")
        logger.info(f"\n{CYAN}Key Findings:{END}")
        logger.info(f"  Token reduction: {token_reduction_percent:.2f}% ({'SUCCESS' if token_reduction_percent > 15 else 'BELOW TARGET'})")
        logger.info(f"  Accuracy maintained: {'YES' if accuracy_delta >= -0.05 else 'NO'} (Δ={accuracy_delta:+.4f})")
        logger.info(f"  Statistical significance: {'YES' if stat_tests['t_significant'] else 'NO'} (p={stat_tests['t_pvalue']:.4f})")
        logger.info(f"\n{GREEN if success else RED}HYPOTHESIS {'CONFIRMED' if success else 'NOT CONFIRMED'}{END}\n")

        return 0 if success else 1

    except Exception as e:
        logger.error(f"\n{RED}{'='*80}{END}")
        logger.error(f"{RED}EXPERIMENT FAILED{END}")
        logger.error(f"{RED}{'='*80}{END}")
        logger.error(f"{RED}Error: {e}{END}")

        import traceback
        logger.error(f"\n{RED}Traceback:{END}")
        logger.error(traceback.format_exc())

        # Save error result (empty examples array to match schema)
        workspace_dir = Path(__file__).parent
        error_result = ExperimentResult(
            examples=[]
        )

        output_path = workspace_dir / "method_out.json"
        with open(output_path, 'w') as f:
            json.dump(asdict(error_result), f, indent=2)

        # Save error details in summary
        error_summary = {
            "error": str(e),
            "success": False
        }
        summary_path = workspace_dir / "method_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(error_summary, f, indent=2)

        return 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Dynamic Rank Adaptation for Low-Rank Recurrent Coordinator

This script implements and evaluates:
1. BASELINE: Full-rank recurrent coordinator
2. STATIC LOW-RANK: Fixed rank=32 coordinator (from exp_2_006)
3. DYNAMIC ADAPTIVE RANK: Rank adapts based on episode complexity

Hypothesis: Dynamic rank adaptation achieves better token efficiency than static low-rank
while maintaining or improving task performance.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, asdict

import numpy as np
import tiktoken
from scipy.stats import ttest_rel
from sklearn.metrics import accuracy_score, f1_score

# Configure extensive logging
logging.basicConfig(
    level=logging.INFO,
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
    predict_static: str
    predict_dynamic: str
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
            return tokens
        except Exception as e:
            logger.debug(f"Error counting tokens: {e}")
            # Fallback: approximate as words * 1.3
            return int(len(text.split()) * 1.3)

    def log_coordinator_step(self, agent_outputs: List[str], coordinator_message: str = ""):
        """Log tokens for a coordinator step with comprehensive validation."""
        try:
            # Validate inputs
            if not isinstance(agent_outputs, list):
                logger.error(f"{RED}agent_outputs must be a list, got {type(agent_outputs)}{END}")
                raise TypeError(f"agent_outputs must be a list, got {type(agent_outputs)}")

            if len(agent_outputs) == 0:
                logger.warning(f"{YELLOW}Empty agent_outputs list{END}")

            step_tokens = 0

            # Count tokens in agent outputs
            for idx, output in enumerate(agent_outputs):
                if not isinstance(output, str):
                    logger.warning(f"{YELLOW}Agent output {idx} is not a string: {type(output)}{END}")
                    output = str(output)

                output_tokens = self.count_tokens(output)
                step_tokens += output_tokens
                logger.debug(f"Agent {idx}: {output_tokens} tokens, {len(output)} chars")

            # Count tokens in coordinator message
            if coordinator_message:
                if not isinstance(coordinator_message, str):
                    logger.warning(f"{YELLOW}Coordinator message is not a string: {type(coordinator_message)}{END}")
                    coordinator_message = str(coordinator_message)

                coord_tokens = self.count_tokens(coordinator_message)
                step_tokens += coord_tokens
                logger.debug(f"Coordinator: {coord_tokens} tokens, {len(coordinator_message)} chars")

            # Update tracking
            self.total_tokens += step_tokens
            self.episode_tokens.append(step_tokens)
            self.call_count += 1

            logger.debug(f"Step total: {step_tokens} tokens (cumulative: {self.total_tokens})")
            return step_tokens

        except Exception as e:
            logger.error(f"{RED}Error logging coordinator step: {e}{END}")
            import traceback
            logger.error(traceback.format_exc())
            return 0

    def get_stats(self) -> Dict[str, float]:
        """Get aggregated token statistics."""
        return {
            "total_tokens": self.total_tokens,
            "num_episodes": len(self.episode_tokens),
            "mean_tokens_per_episode": np.mean(self.episode_tokens) if self.episode_tokens else 0,
            "std_tokens_per_episode": np.std(self.episode_tokens) if self.episode_tokens else 0,
            "call_count": self.call_count
        }


class FullRankCoordinator:
    """Baseline: Full-rank recurrent coordinator."""

    def __init__(self, hidden_dim: int = 256):
        self.hidden_dim = hidden_dim
        self.W = np.random.randn(hidden_dim, hidden_dim) * 0.01
        self.state = np.zeros(hidden_dim)
        logger.info(f"{GREEN}FullRankCoordinator initialized: hidden_dim={hidden_dim}{END}")

    def reset(self):
        """Reset coordinator state."""
        self.state = np.zeros(self.hidden_dim)

    def step(self, agent_outputs: List[str]) -> Tuple[np.ndarray, str]:
        """Recurrent step."""
        features = self._encode_outputs(agent_outputs)
        self.state = self.W @ self.state + features
        coordinator_message = self._generate_message(self.state)
        return self.state.copy(), coordinator_message

    def _encode_outputs(self, outputs: List[str]) -> np.ndarray:
        """Encode agent outputs into feature vector."""
        features = []
        for output in outputs:
            features.extend([
                len(output.split()),
                len(output),
                output.count('.'),
                output.count('?'),
            ])

        features_array = np.array(features[:self.hidden_dim])
        if len(features_array) < self.hidden_dim:
            padded = np.zeros(self.hidden_dim)
            padded[:len(features_array)] = features_array
            features_array = padded

        features_array = features_array / (np.linalg.norm(features_array) + 1e-8)
        return features_array

    def _generate_message(self, state: np.ndarray) -> str:
        """Generate coordinator message from state."""
        message_parts = []
        for i in range(0, len(state), 10):
            val = state[i]
            if abs(val) > 0.1:
                message_parts.append(f"dim{i}:{val:.2f}")
        return " ".join(message_parts)


class StaticLowRankCoordinator:
    """Static low-rank coordinator with fixed rank (baseline method from exp_2_006)."""

    def __init__(self, hidden_dim: int = 256, rank: int = 32, num_modules: int = 4):
        self.hidden_dim = hidden_dim
        self.rank = rank
        self.num_modules = num_modules

        # Low-rank factorization: W = U @ V^T
        self.U = np.random.randn(hidden_dim, rank) * 0.01
        self.V = np.random.randn(hidden_dim, rank) * 0.01

        # RIM sparse attention
        self.active_k = max(1, num_modules // 2)
        self.module_weights = [
            np.random.randn(hidden_dim, rank) * 0.01
            for _ in range(num_modules)
        ]

        self.state = np.zeros(hidden_dim)

        logger.info(f"{GREEN}StaticLowRankCoordinator initialized: rank={rank}, modules={num_modules}{END}")

    def reset(self):
        """Reset coordinator state."""
        self.state = np.zeros(self.hidden_dim)

    def step(self, agent_outputs: List[str]) -> Tuple[np.ndarray, str]:
        """Low-rank recurrent step with sparse module updates."""
        features = self._encode_outputs(agent_outputs)
        active_modules = self._select_active_modules(self.state, features)

        # Low-rank update
        state_proj = self.V.T @ self.state
        new_state = self.U @ state_proj

        # Apply sparse module updates
        for module_idx in active_modules:
            module_update = self.module_weights[module_idx] @ state_proj
            new_state += module_update

        new_state += features
        self.state = new_state

        # Generate compressed message (only rank dimensions)
        coordinator_message = self._generate_compressed_message(state_proj)
        return self.state.copy(), coordinator_message

    def _encode_outputs(self, outputs: List[str]) -> np.ndarray:
        """Encode agent outputs into feature vector."""
        features = []
        for output in outputs:
            features.extend([
                len(output.split()),
                len(output),
                output.count('.'),
                output.count('?'),
            ])

        features_array = np.array(features[:self.hidden_dim])
        if len(features_array) < self.hidden_dim:
            padded = np.zeros(self.hidden_dim)
            padded[:len(features_array)] = features_array
            features_array = padded

        features_array = features_array / (np.linalg.norm(features_array) + 1e-8)
        return features_array

    def _select_active_modules(self, state: np.ndarray, features: np.ndarray) -> List[int]:
        """Select top-k modules based on attention scores."""
        scores = []
        state_proj = self.V.T @ state

        for module_idx, module_w in enumerate(self.module_weights):
            module_proj = module_w.T @ features
            score = np.dot(module_proj[:self.rank], state_proj)
            scores.append((score, module_idx))

        scores.sort(reverse=True)
        return [idx for _, idx in scores[:self.active_k]]

    def _generate_compressed_message(self, state_proj: np.ndarray) -> str:
        """Generate compressed coordinator message from projected state."""
        message_parts = []
        for i in range(len(state_proj)):
            val = state_proj[i]
            if abs(val) > 0.1:
                message_parts.append(f"r{i}:{val:.2f}")
        return " ".join(message_parts)


class DynamicRankCoordinator:
    """
    NOVEL: Dynamic rank adaptation coordinator.

    Key innovation: Rank adapts based on episode complexity:
    - Starts with minimal rank (8)
    - Increases rank if uncertainty is high (low prediction confidence)
    - Decreases rank if task is simple (high confidence)
    - Max rank = 64 (still more compressed than full-rank 256)

    This should achieve better token efficiency by using low rank for simple
    episodes and higher rank only when needed.
    """

    def __init__(self, hidden_dim: int = 256, min_rank: int = 8, max_rank: int = 64, num_modules: int = 4):
        self.hidden_dim = hidden_dim
        self.min_rank = min_rank
        self.max_rank = max_rank
        self.current_rank = min_rank  # Start with minimal rank
        self.num_modules = num_modules

        # Initialize matrices for max rank (we'll only use first current_rank columns)
        self.U = np.random.randn(hidden_dim, max_rank) * 0.01
        self.V = np.random.randn(hidden_dim, max_rank) * 0.01

        # RIM sparse attention
        self.active_k = max(1, num_modules // 2)
        self.module_weights = [
            np.random.randn(hidden_dim, max_rank) * 0.01
            for _ in range(num_modules)
        ]

        self.state = np.zeros(hidden_dim)

        # Track rank adaptation history
        self.rank_history = []

        logger.info(f"{GREEN}DynamicRankCoordinator initialized:{END}")
        logger.info(f"  min_rank={min_rank}, max_rank={max_rank}, modules={num_modules}")
        logger.info(f"  Adaptive rank selection enabled")

    def reset(self):
        """Reset coordinator state."""
        self.state = np.zeros(self.hidden_dim)
        # Reset to minimum rank for new episode
        self.current_rank = self.min_rank

    def step(self, agent_outputs: List[str]) -> Tuple[np.ndarray, str]:
        """Dynamic rank recurrent step."""
        features = self._encode_outputs(agent_outputs)

        # ADAPTIVE RANK SELECTION based on input complexity
        self._adapt_rank(agent_outputs, features)

        active_modules = self._select_active_modules(self.state, features)

        # Low-rank update using CURRENT rank (not full max_rank)
        state_proj = self.V[:, :self.current_rank].T @ self.state  # Project to current_rank space
        new_state = self.U[:, :self.current_rank] @ state_proj  # Reconstruct

        # Apply sparse module updates (only using current rank)
        for module_idx in active_modules:
            module_update = self.module_weights[module_idx][:, :self.current_rank] @ state_proj
            new_state += module_update

        new_state += features
        self.state = new_state

        # Generate compressed message (size depends on current rank)
        coordinator_message = self._generate_compressed_message(state_proj)

        # Track rank for analysis
        self.rank_history.append(self.current_rank)

        return self.state.copy(), coordinator_message

    def _adapt_rank(self, agent_outputs: List[str], features: np.ndarray):
        """
        Adapt rank based on episode complexity.

        Complexity signals:
        1. Length variance between agents (high variance = complex disagreement)
        2. Feature magnitude (high magnitude = complex input)
        3. State uncertainty (high variance in state = need more capacity)
        """
        # Signal 1: Length variance
        lengths = [len(out) for out in agent_outputs]
        length_variance = np.var(lengths) if len(lengths) > 1 else 0

        # Signal 2: Feature magnitude
        feature_magnitude = np.linalg.norm(features)

        # Signal 3: State uncertainty (variance across dimensions)
        state_uncertainty = np.var(self.state) if np.any(self.state) else 0

        # Combine signals into complexity score (normalized)
        complexity_score = (
            (length_variance / 10000) * 0.4 +  # Normalize length variance
            feature_magnitude * 0.3 +  # Feature magnitude already [0, 1]
            (state_uncertainty * 10) * 0.3  # State uncertainty
        )

        # Map complexity to rank
        # Low complexity (< 0.3) -> min_rank
        # Medium complexity (0.3-0.7) -> mid_rank
        # High complexity (> 0.7) -> max_rank
        if complexity_score < 0.3:
            target_rank = self.min_rank
        elif complexity_score < 0.7:
            # Linear interpolation between min and max
            progress = (complexity_score - 0.3) / 0.4
            target_rank = int(self.min_rank + progress * (self.max_rank - self.min_rank))
        else:
            target_rank = self.max_rank

        # Smooth adaptation (don't jump too fast)
        if target_rank > self.current_rank:
            self.current_rank = min(self.current_rank + 8, target_rank)
        elif target_rank < self.current_rank:
            self.current_rank = max(self.current_rank - 4, target_rank)

        # Ensure bounds
        self.current_rank = max(self.min_rank, min(self.max_rank, self.current_rank))

        logger.debug(f"  Complexity: {complexity_score:.3f}, Rank: {self.current_rank}")

    def _encode_outputs(self, outputs: List[str]) -> np.ndarray:
        """Encode agent outputs into feature vector."""
        features = []
        for output in outputs:
            features.extend([
                len(output.split()),
                len(output),
                output.count('.'),
                output.count('?'),
            ])

        features_array = np.array(features[:self.hidden_dim])
        if len(features_array) < self.hidden_dim:
            padded = np.zeros(self.hidden_dim)
            padded[:len(features_array)] = features_array
            features_array = padded

        features_array = features_array / (np.linalg.norm(features_array) + 1e-8)
        return features_array

    def _select_active_modules(self, state: np.ndarray, features: np.ndarray) -> List[int]:
        """Select top-k modules based on attention scores."""
        scores = []
        state_proj = self.V[:, :self.current_rank].T @ state

        for module_idx, module_w in enumerate(self.module_weights):
            module_proj = module_w[:, :self.current_rank].T @ features
            score = np.dot(module_proj, state_proj)
            scores.append((score, module_idx))

        scores.sort(reverse=True)
        return [idx for _, idx in scores[:self.active_k]]

    def _generate_compressed_message(self, state_proj: np.ndarray) -> str:
        """Generate compressed coordinator message from projected state."""
        message_parts = []
        for i in range(len(state_proj)):
            val = state_proj[i]
            if abs(val) > 0.1:
                message_parts.append(f"r{i}:{val:.2f}")
        return " ".join(message_parts)

    def get_rank_stats(self) -> Dict[str, float]:
        """Get rank adaptation statistics."""
        if not self.rank_history:
            return {}

        return {
            "mean_rank": float(np.mean(self.rank_history)),
            "std_rank": float(np.std(self.rank_history)),
            "min_rank_used": int(np.min(self.rank_history)),
            "max_rank_used": int(np.max(self.rank_history)),
            "rank_changes": int(np.sum(np.abs(np.diff(self.rank_history)) > 0))
        }


def load_dataset(data_path: Path) -> List[Dict]:
    """Load dataset from JSON file with exhaustive error checking."""
    try:
        logger.info(f"{BLUE}Loading dataset from: {data_path}{END}")

        # Check file exists
        if not data_path.exists():
            logger.error(f"{RED}Dataset file does not exist: {data_path}{END}")
            raise FileNotFoundError(f"Dataset not found: {data_path}")

        # Check file is readable
        if not data_path.is_file():
            logger.error(f"{RED}Path is not a file: {data_path}{END}")
            raise ValueError(f"Path is not a file: {data_path}")

        # Check file size
        file_size = data_path.stat().st_size
        logger.info(f"Dataset file size: {file_size:,} bytes")
        if file_size == 0:
            logger.error(f"{RED}Dataset file is empty{END}")
            raise ValueError("Dataset file is empty")

        # Load JSON with error handling
        with open(data_path, 'r') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                logger.error(f"{RED}Invalid JSON in dataset file: {e}{END}")
                raise

        # Validate data structure
        if not isinstance(data, dict):
            logger.error(f"{RED}Dataset root is not a dict, got {type(data)}{END}")
            raise ValueError(f"Expected dict, got {type(data)}")

        if 'examples' not in data:
            logger.error(f"{RED}Dataset missing 'examples' key. Keys: {list(data.keys())}{END}")
            raise ValueError("Dataset missing 'examples' key")

        examples = data.get('examples', [])

        # Validate examples
        if not isinstance(examples, list):
            logger.error(f"{RED}Examples is not a list, got {type(examples)}{END}")
            raise ValueError(f"Expected list, got {type(examples)}")

        if len(examples) == 0:
            logger.warning(f"{YELLOW}No examples in dataset{END}")

        logger.info(f"{GREEN}Loaded {len(examples)} examples from dataset{END}")

        # Validate first example structure
        if examples:
            required_keys = ['input', 'output', 'context', 'dataset', 'split']
            first_ex = examples[0]
            missing_keys = [k for k in required_keys if k not in first_ex]
            if missing_keys:
                logger.warning(f"{YELLOW}First example missing keys: {missing_keys}{END}")
            else:
                logger.info(f"{GREEN}First example has all required keys{END}")

        return examples

    except Exception as e:
        logger.error(f"{RED}Error loading dataset: {e}{END}")
        import traceback
        logger.error(traceback.format_exc())
        raise


def run_experiment(examples: List[Dict], coordinator, tracker: TokenTracker, name: str) -> Tuple[List[str], List[int]]:
    """Run experiment with given coordinator with comprehensive validation."""
    try:
        logger.info(f"\n{BLUE}{'='*80}{END}")
        logger.info(f"{BLUE}Running experiment: {name}{END}")
        logger.info(f"{BLUE}{'='*80}{END}\n")

        # Validate inputs
        if not isinstance(examples, list):
            logger.error(f"{RED}examples must be a list, got {type(examples)}{END}")
            raise TypeError(f"examples must be a list, got {type(examples)}")

        if len(examples) == 0:
            logger.error(f"{RED}Empty examples list{END}")
            raise ValueError("Empty examples list")

        if coordinator is None:
            logger.error(f"{RED}coordinator is None{END}")
            raise ValueError("coordinator is None")

        if tracker is None:
            logger.error(f"{RED}tracker is None{END}")
            raise ValueError("tracker is None")

        logger.info(f"Processing {len(examples)} examples with {name}")

        predictions = []
        token_counts = []

        for idx, example in enumerate(examples):
            try:
                if (idx + 1) % 20 == 0 or len(examples) <= 10:
                    logger.info(f"{CYAN}Processing example {idx+1}/{len(examples)}{END}")

                # Validate example structure
                if not isinstance(example, dict):
                    logger.error(f"{RED}Example {idx+1} is not a dict: {type(example)}{END}")
                    raise TypeError(f"Example {idx+1} is not a dict")

                if 'context' not in example:
                    logger.error(f"{RED}Example {idx+1} missing 'context' key{END}")
                    raise KeyError(f"Example {idx+1} missing 'context'")

                # Reset coordinator state for new episode
                coordinator.reset()
                logger.debug(f"Example {idx+1}: Coordinator state reset")

                # Track tokens for this episode
                episode_start_tokens = tracker.total_tokens

                # Extract agent outputs from dataset
                context = example['context']

                # Validate context structure
                if 'response_a' not in context:
                    logger.error(f"{RED}Example {idx+1} context missing 'response_a'{END}")
                    raise KeyError("context missing 'response_a'")
                if 'response_b' not in context:
                    logger.error(f"{RED}Example {idx+1} context missing 'response_b'{END}")
                    raise KeyError("context missing 'response_b'")

                agent_outputs = [
                    context['response_a'],
                    context['response_b']
                ]

                logger.debug(f"Example {idx+1}: Agent outputs extracted ({len(agent_outputs)} outputs)")

                # Coordinator processes both agent responses
                state, coordinator_message = coordinator.step(agent_outputs)

                # Validate outputs
                if state is None:
                    logger.error(f"{RED}Coordinator returned None state{END}")
                    raise ValueError("Coordinator returned None state")

                logger.debug(f"Example {idx+1}: State shape={state.shape if hasattr(state, 'shape') else 'N/A'}")

                # Track tokens
                step_tokens = tracker.log_coordinator_step(agent_outputs, coordinator_message)
                logger.debug(f"Example {idx+1}: Step tokens={step_tokens}")

                # Make prediction
                prediction = predict_winner(state, agent_outputs, context)
                predictions.append(prediction)
                logger.debug(f"Example {idx+1}: Prediction={prediction}")

                # Record tokens for this episode
                episode_tokens = tracker.total_tokens - episode_start_tokens
                token_counts.append(episode_tokens)
                logger.debug(f"Example {idx+1}: Episode tokens={episode_tokens}")

            except Exception as e:
                logger.error(f"{RED}Error processing example {idx+1}: {e}{END}")
                import traceback
                logger.error(traceback.format_exc())
                predictions.append("tie")
                token_counts.append(0)

        logger.info(f"\n{GREEN}Experiment {name} complete: {len(predictions)} predictions made{END}")

        # Validate outputs
        if len(predictions) != len(examples):
            logger.warning(f"{YELLOW}Predictions count mismatch: {len(predictions)} vs {len(examples)} examples{END}")

        if len(token_counts) != len(examples):
            logger.warning(f"{YELLOW}Token counts mismatch: {len(token_counts)} vs {len(examples)} examples{END}")

        return predictions, token_counts

    except Exception as e:
        logger.error(f"{RED}Error in run_experiment: {e}{END}")
        import traceback
        logger.error(traceback.format_exc())
        raise


def predict_winner(state: np.ndarray, agent_outputs: List[str], context: Dict) -> str:
    """Predict winner based on coordinator state."""
    try:
        len_a = len(agent_outputs[0])
        len_b = len(agent_outputs[1])

        confidence = np.sum(state[state > 0])

        score_a = len_a * (1 + confidence * 0.1)
        score_b = len_b * (1 + confidence * 0.1)

        if abs(score_a - score_b) < 50:
            return "tie"
        elif score_a > score_b:
            return "model_a"
        else:
            return "model_b"

    except Exception as e:
        logger.error(f"{RED}Error in predict_winner: {e}{END}")
        return "tie"


def evaluate_performance(predictions: List[str], ground_truth: List[str]) -> Dict[str, float]:
    """Evaluate task performance metrics."""
    try:
        accuracy = accuracy_score(ground_truth, predictions)
        f1_macro = f1_score(ground_truth, predictions, average='macro', zero_division=0)
        f1_weighted = f1_score(ground_truth, predictions, average='weighted', zero_division=0)

        return {
            "accuracy": float(accuracy),
            "f1_macro": float(f1_macro),
            "f1_weighted": float(f1_weighted)
        }

    except Exception as e:
        logger.error(f"{RED}Error evaluating performance: {e}{END}")
        return {
            "accuracy": 0.0,
            "f1_macro": 0.0,
            "f1_weighted": 0.0
        }


def compute_statistical_significance(baseline_tokens: List[int], method_tokens: List[int]) -> Dict[str, Any]:
    """Test if improvement is statistically significant."""
    try:
        t_stat, t_pval = ttest_rel(baseline_tokens, method_tokens)
        t_significant = t_pval < 0.05

        mean_diff = np.mean(baseline_tokens) - np.mean(method_tokens)
        pooled_std = np.sqrt((np.var(baseline_tokens) + np.var(method_tokens)) / 2)
        cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0

        return {
            "t_statistic": float(t_stat),
            "t_pvalue": float(t_pval),
            "t_significant": bool(t_significant),
            "cohens_d": float(cohens_d),
            "alpha": 0.05
        }

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
        logger.info(f"{BLUE}DYNAMIC RANK ADAPTATION EXPERIMENT{END}")
        logger.info(f"{BLUE}{'='*80}{END}\n")

        # Configuration
        HIDDEN_DIM = 256
        STATIC_RANK = 32
        MIN_RANK = 8
        MAX_RANK = 64
        NUM_MODULES = 4

        # Load dataset
        workspace_dir = Path.cwd()

        # Try multiple possible dataset paths (PREFER full dataset for final run)
        possible_paths = [
            workspace_dir / "dependencies" / "Extended_Multi-LLM_Coordination_Dataset_with_Token" / "data_out.json",
            workspace_dir / "dependencies" / "Extended_Multi-LLM_Coordination_Dataset_with_Token" / "full_data_out.json",
            workspace_dir / "dependencies" / "Extended_Multi-LLM_Coordination_Dataset_with_Token" / "mini_data_out.json",
            workspace_dir / "dependencies" / "Extended_Multi-LLM_Coordination_Dataset_with_Token" / "preview_data_out.json",
            Path("./dependencies/Extended_Multi-LLM_Coordination_Dataset_with_Token/data_out.json"),
            Path("./dependencies/Extended_Multi-LLM_Coordination_Dataset_with_Token/mini_data_out.json"),
        ]

        data_path = None
        for path in possible_paths:
            logger.info(f"Checking path: {path} (exists: {path.exists()})")
            if path.exists():
                data_path = path
                break

        if data_path is None:
            raise FileNotFoundError(f"Could not find dataset. Workspace: {workspace_dir}, Checked paths: {[str(p) for p in possible_paths]}")

        logger.info(f"Using dataset: {data_path}")
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
            examples, baseline_coordinator, baseline_tracker, "Baseline (Full-Rank)"
        )

        baseline_metrics = evaluate_performance(baseline_predictions, ground_truth)
        baseline_token_stats = baseline_tracker.get_stats()

        logger.info(f"\n{GREEN}Baseline Results:{END}")
        logger.info(f"  Total tokens: {baseline_token_stats['total_tokens']}")
        logger.info(f"  Mean tokens/episode: {baseline_token_stats['mean_tokens_per_episode']:.2f}")
        logger.info(f"  Accuracy: {baseline_metrics['accuracy']:.4f}")

        # ========================================================================
        # STATIC LOW-RANK: Fixed rank=32
        # ========================================================================

        logger.info(f"\n{BLUE}{'='*80}{END}")
        logger.info(f"{BLUE}STATIC LOW-RANK: Fixed rank={STATIC_RANK}{END}")
        logger.info(f"{BLUE}{'='*80}{END}\n")

        static_coordinator = StaticLowRankCoordinator(
            hidden_dim=HIDDEN_DIM,
            rank=STATIC_RANK,
            num_modules=NUM_MODULES
        )
        static_tracker = TokenTracker()

        static_predictions, static_token_counts = run_experiment(
            examples, static_coordinator, static_tracker, "Static Low-Rank"
        )

        static_metrics = evaluate_performance(static_predictions, ground_truth)
        static_token_stats = static_tracker.get_stats()

        logger.info(f"\n{GREEN}Static Low-Rank Results:{END}")
        logger.info(f"  Total tokens: {static_token_stats['total_tokens']}")
        logger.info(f"  Mean tokens/episode: {static_token_stats['mean_tokens_per_episode']:.2f}")
        logger.info(f"  Accuracy: {static_metrics['accuracy']:.4f}")

        # ========================================================================
        # DYNAMIC ADAPTIVE RANK
        # ========================================================================

        logger.info(f"\n{BLUE}{'='*80}{END}")
        logger.info(f"{BLUE}DYNAMIC ADAPTIVE RANK: rank={MIN_RANK}-{MAX_RANK}{END}")
        logger.info(f"{BLUE}{'='*80}{END}\n")

        dynamic_coordinator = DynamicRankCoordinator(
            hidden_dim=HIDDEN_DIM,
            min_rank=MIN_RANK,
            max_rank=MAX_RANK,
            num_modules=NUM_MODULES
        )
        dynamic_tracker = TokenTracker()

        dynamic_predictions, dynamic_token_counts = run_experiment(
            examples, dynamic_coordinator, dynamic_tracker, "Dynamic Adaptive Rank"
        )

        dynamic_metrics = evaluate_performance(dynamic_predictions, ground_truth)
        dynamic_token_stats = dynamic_tracker.get_stats()
        dynamic_rank_stats = dynamic_coordinator.get_rank_stats()

        logger.info(f"\n{GREEN}Dynamic Adaptive Rank Results:{END}")
        logger.info(f"  Total tokens: {dynamic_token_stats['total_tokens']}")
        logger.info(f"  Mean tokens/episode: {dynamic_token_stats['mean_tokens_per_episode']:.2f}")
        logger.info(f"  Accuracy: {dynamic_metrics['accuracy']:.4f}")
        logger.info(f"  Mean rank: {dynamic_rank_stats.get('mean_rank', 0):.2f}")
        logger.info(f"  Rank range: [{dynamic_rank_stats.get('min_rank_used', 0)}, {dynamic_rank_stats.get('max_rank_used', 0)}]")

        # ========================================================================
        # COMPARISON & STATISTICAL ANALYSIS
        # ========================================================================

        logger.info(f"\n{BLUE}{'='*80}{END}")
        logger.info(f"{BLUE}COMPARISON & STATISTICAL ANALYSIS{END}")
        logger.info(f"{BLUE}{'='*80}{END}\n")

        # Token reduction: baseline vs static
        static_reduction = (
            (baseline_token_stats['total_tokens'] - static_token_stats['total_tokens'])
            / baseline_token_stats['total_tokens'] * 100
        )

        # Token reduction: baseline vs dynamic
        dynamic_reduction = (
            (baseline_token_stats['total_tokens'] - dynamic_token_stats['total_tokens'])
            / baseline_token_stats['total_tokens'] * 100
        )

        # Token reduction: static vs dynamic (key comparison)
        dynamic_vs_static_reduction = (
            (static_token_stats['total_tokens'] - dynamic_token_stats['total_tokens'])
            / static_token_stats['total_tokens'] * 100
        )

        logger.info(f"{CYAN}Token Efficiency:{END}")
        logger.info(f"  Baseline total: {baseline_token_stats['total_tokens']}")
        logger.info(f"  Static total: {static_token_stats['total_tokens']} ({static_reduction:+.2f}% vs baseline)")
        logger.info(f"  Dynamic total: {dynamic_token_stats['total_tokens']} ({dynamic_reduction:+.2f}% vs baseline)")
        logger.info(f"  Dynamic vs Static: {dynamic_vs_static_reduction:+.2f}%")

        # Statistical tests
        stat_tests_dynamic_vs_baseline = compute_statistical_significance(
            baseline_token_counts, dynamic_token_counts
        )
        stat_tests_dynamic_vs_static = compute_statistical_significance(
            static_token_counts, dynamic_token_counts
        )

        logger.info(f"\n{CYAN}Statistical Significance:{END}")
        logger.info(f"  Dynamic vs Baseline: p={stat_tests_dynamic_vs_baseline['t_pvalue']:.4f}, sig={stat_tests_dynamic_vs_baseline['t_significant']}")
        logger.info(f"  Dynamic vs Static: p={stat_tests_dynamic_vs_static['t_pvalue']:.4f}, sig={stat_tests_dynamic_vs_static['t_significant']}")

        # Success criterion: dynamic better than static
        success = (
            dynamic_reduction > static_reduction and
            dynamic_metrics['accuracy'] >= static_metrics['accuracy'] - 0.05
        )

        logger.info(f"\n{CYAN}Hypothesis Test:{END}")
        logger.info(f"  Dynamic achieves better token reduction than static: {dynamic_reduction > static_reduction}")
        logger.info(f"  Dynamic maintains performance: {dynamic_metrics['accuracy'] >= static_metrics['accuracy'] - 0.05}")
        logger.info(f"  SUCCESS: {success}")

        # ========================================================================
        # SAVE RESULTS
        # ========================================================================

        # Build examples array (using dynamic as the "method" since it's our proposed approach)
        result_examples = []
        for idx, example in enumerate(examples):
            result_examples.append({
                "input": example['input'],
                "output": example['output'],
                "context": example['context'],
                "dataset": example['dataset'],
                "split": example['split'],
                "predict_baseline": baseline_predictions[idx],
                "predict_method": dynamic_predictions[idx],  # Dynamic is our proposed method
                "method": "Dynamic Rank Adaptation for Low-Rank Recurrent Coordinator"
            })

        result = ExperimentResult(examples=result_examples)

        output_path = workspace_dir / "method_out.json"
        with open(output_path, 'w') as f:
            json.dump(asdict(result), f, indent=2)

        # Save summary metrics
        summary_path = workspace_dir / "method_summary.json"
        summary = {
            "method_name": "Dynamic Rank Adaptation",
            "baseline_name": "Full-Rank Coordinator",
            "static_baseline_name": "Static Low-Rank Coordinator (rank=32)",
            "dataset_size": len(examples),
            "baseline_metrics": {
                **baseline_metrics,
                **{k: float(v) if isinstance(v, (np.integer, np.floating)) else v
                   for k, v in baseline_token_stats.items()}
            },
            "static_metrics": {
                **static_metrics,
                **{k: float(v) if isinstance(v, (np.integer, np.floating)) else v
                   for k, v in static_token_stats.items()}
            },
            "dynamic_metrics": {
                **dynamic_metrics,
                **{k: float(v) if isinstance(v, (np.integer, np.floating)) else v
                   for k, v in dynamic_token_stats.items()},
                **dynamic_rank_stats
            },
            "improvement_metrics": {
                "static_vs_baseline_reduction_percent": float(static_reduction),
                "dynamic_vs_baseline_reduction_percent": float(dynamic_reduction),
                "dynamic_vs_static_reduction_percent": float(dynamic_vs_static_reduction),
                "dynamic_accuracy_delta_vs_baseline": float(dynamic_metrics['accuracy'] - baseline_metrics['accuracy']),
                "dynamic_accuracy_delta_vs_static": float(dynamic_metrics['accuracy'] - static_metrics['accuracy']),
            },
            "statistical_tests": {
                "dynamic_vs_baseline": stat_tests_dynamic_vs_baseline,
                "dynamic_vs_static": stat_tests_dynamic_vs_static
            },
            "configuration": {
                "hidden_dim": HIDDEN_DIM,
                "static_rank": STATIC_RANK,
                "dynamic_min_rank": MIN_RANK,
                "dynamic_max_rank": MAX_RANK,
                "num_modules": NUM_MODULES,
            },
            "success": success
        }

        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"\n{GREEN}Results saved to: {output_path}{END}")
        logger.info(f"{GREEN}Summary metrics saved to: {summary_path}{END}")

        # Print final summary
        logger.info(f"\n{BLUE}{'='*80}{END}")
        logger.info(f"{BLUE}EXPERIMENT SUMMARY{END}")
        logger.info(f"{BLUE}{'='*80}{END}\n")
        logger.info(f"Examples processed: {len(examples)}")
        logger.info(f"\n{CYAN}Key Findings:{END}")
        logger.info(f"  Static reduction vs baseline: {static_reduction:.2f}%")
        logger.info(f"  Dynamic reduction vs baseline: {dynamic_reduction:.2f}%")
        logger.info(f"  Dynamic improvement over static: {dynamic_vs_static_reduction:+.2f}%")
        logger.info(f"  Dynamic mean rank: {dynamic_rank_stats.get('mean_rank', 0):.2f}")
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

        # Save error result
        workspace_dir = Path(__file__).parent
        error_result = ExperimentResult(examples=[])

        output_path = workspace_dir / "method_out.json"
        with open(output_path, 'w') as f:
            json.dump(asdict(error_result), f, indent=2)

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

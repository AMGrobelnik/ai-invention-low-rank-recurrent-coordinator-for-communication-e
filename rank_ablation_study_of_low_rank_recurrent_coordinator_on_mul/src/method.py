#!/usr/bin/env python3
"""
Rank-Ablation Study of Low-Rank Recurrent Coordinator on Multi-LLM Coordination Datasets

This script implements a systematic ablation study across multiple rank values to evaluate:
1. Token-efficiency (total tokens / API calls) vs. rank
2. Task performance (win-rate accuracy) vs. rank
3. Statistical significance of performance differences
4. Identification of optimal rank values and failure modes

Hypothesis: If the low-rank recurrent coordinator retains performance while reducing token usage,
we will obtain a clear trade-off curve that validates the hypothesis and identifies minimum viable rank.
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
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

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
            return tokens
        except Exception:
            # Fallback: approximate as words * 1.3
            return int(len(text.split()) * 1.3)

    def log_coordinator_step(self, agent_outputs: List[str], coordinator_message: str = ""):
        """Log tokens for a coordinator step."""
        step_tokens = 0

        # Count tokens in agent outputs
        for output in agent_outputs:
            step_tokens += self.count_tokens(output)

        # Count tokens in coordinator message
        if coordinator_message:
            step_tokens += self.count_tokens(coordinator_message)

        self.total_tokens += step_tokens
        self.episode_tokens.append(step_tokens)
        self.call_count += 1

        return step_tokens

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
    """Baseline: Full-rank recurrent coordinator for multi-agent coordination."""

    def __init__(self, hidden_dim: int = 256):
        """Initialize full-rank coordinator."""
        self.hidden_dim = hidden_dim
        self.W = np.random.randn(hidden_dim, hidden_dim) * 0.01
        self.state = np.zeros(hidden_dim)
        logger.info(f"{GREEN}FullRankCoordinator initialized: hidden_dim={hidden_dim}{END}")

    def reset(self):
        """Reset coordinator state."""
        self.state = np.zeros(self.hidden_dim)

    def step(self, agent_outputs: List[str]) -> Tuple[np.ndarray, str]:
        """Recurrent step: process agent outputs and update state."""
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


class LowRankRecurrentCoordinator:
    """Low-rank recurrent coordinator with rank parameter for ablation."""

    def __init__(self, hidden_dim: int = 256, rank: int = 32, num_modules: int = 4):
        """Initialize low-rank coordinator with specified rank."""
        self.hidden_dim = hidden_dim
        self.rank = rank
        self.num_modules = num_modules

        # Low-rank factorization: W = U @ V^T
        self.U = np.random.randn(hidden_dim, rank) * 0.01
        self.V = np.random.randn(hidden_dim, rank) * 0.01

        # RIM sparse attention
        self.active_k = max(1, num_modules // 2)

        # Module-specific transformations
        self.module_weights = [
            np.random.randn(hidden_dim, rank) * 0.01
            for _ in range(num_modules)
        ]

        self.state = np.zeros(hidden_dim)

        compression_ratio = rank / hidden_dim
        param_reduction = (2 * hidden_dim * rank) / (hidden_dim * hidden_dim)

        logger.info(f"{GREEN}LowRankCoordinator initialized: rank={rank}, compression={compression_ratio:.2%}{END}")

    def reset(self):
        """Reset coordinator state."""
        self.state = np.zeros(self.hidden_dim)

    def step(self, agent_outputs: List[str]) -> Tuple[np.ndarray, str]:
        """Low-rank recurrent step with sparse module updates."""
        features = self._encode_outputs(agent_outputs)
        active_modules = self._select_active_modules(self.state, features)

        # Low-rank update: s_{t+1} = U @ (V^T @ s_t) + module_updates
        state_proj = self.V.T @ self.state
        new_state = self.U @ state_proj

        # Apply sparse module updates
        for module_idx in active_modules:
            module_update = self.module_weights[module_idx] @ state_proj
            new_state += module_update

        new_state += features
        self.state = new_state

        # Compressed message based on rank
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
        active = [idx for _, idx in scores[:self.active_k]]
        return active

    def _generate_compressed_message(self, state_proj: np.ndarray) -> str:
        """Generate compressed coordinator message from projected state."""
        message_parts = []
        for i in range(len(state_proj)):
            val = state_proj[i]
            if abs(val) > 0.1:
                message_parts.append(f"r{i}:{val:.2f}")
        return " ".join(message_parts)


def load_dataset(data_path: Path) -> List[Dict]:
    """Load dataset from JSON file."""
    logger.info(f"{BLUE}Loading dataset from: {data_path}{END}")

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    with open(data_path, 'r') as f:
        data = json.load(f)

    examples = data.get('examples', [])
    logger.info(f"{GREEN}Loaded {len(examples)} examples from dataset{END}")
    return examples


def run_experiment(examples: List[Dict], coordinator, tracker: TokenTracker, name: str) -> Tuple[List[str], List[int]]:
    """Run experiment with given coordinator."""
    logger.info(f"\n{BLUE}Running experiment: {name}{END}")

    predictions = []
    token_counts = []

    for idx, example in enumerate(examples):
        coordinator.reset()
        episode_start_tokens = tracker.total_tokens

        context = example['context']
        agent_outputs = [context['response_a'], context['response_b']]

        state, coordinator_message = coordinator.step(agent_outputs)
        step_tokens = tracker.log_coordinator_step(agent_outputs, coordinator_message)

        prediction = predict_winner(state, agent_outputs, context)
        predictions.append(prediction)

        episode_tokens = tracker.total_tokens - episode_start_tokens
        token_counts.append(episode_tokens)

    logger.info(f"{GREEN}Experiment {name} complete: {len(predictions)} predictions{END}")
    return predictions, token_counts


def predict_winner(state: np.ndarray, agent_outputs: List[str], context: Dict) -> str:
    """Predict winner based on coordinator state."""
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


def evaluate_performance(predictions: List[str], ground_truth: List[str]) -> Dict[str, float]:
    """Evaluate task performance metrics."""
    accuracy = accuracy_score(ground_truth, predictions)
    f1_macro = f1_score(ground_truth, predictions, average='macro', zero_division=0)
    f1_weighted = f1_score(ground_truth, predictions, average='weighted', zero_division=0)

    return {
        "accuracy": float(accuracy),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted)
    }


def run_rank_ablation(examples: List[Dict], ground_truth: List[str], rank_values: List[int], hidden_dim: int = 256):
    """Run ablation study across multiple rank values."""
    logger.info(f"\n{BLUE}{'='*80}{END}")
    logger.info(f"{BLUE}RANK ABLATION STUDY{END}")
    logger.info(f"{BLUE}{'='*80}{END}\n")

    results = {
        "baseline": None,
        "rank_ablation": {}
    }

    # Run baseline (full-rank)
    logger.info(f"\n{CYAN}Running BASELINE (Full-Rank){END}")
    baseline_coordinator = FullRankCoordinator(hidden_dim=hidden_dim)
    baseline_tracker = TokenTracker()

    baseline_predictions, baseline_token_counts = run_experiment(
        examples, baseline_coordinator, baseline_tracker, "Baseline (Full-Rank)"
    )

    baseline_metrics = evaluate_performance(baseline_predictions, ground_truth)
    baseline_token_stats = baseline_tracker.get_stats()

    results["baseline"] = {
        "metrics": baseline_metrics,
        "token_stats": {k: float(v) if isinstance(v, (np.integer, np.floating)) else v
                       for k, v in baseline_token_stats.items()},
        "predictions": baseline_predictions,
        "token_counts": baseline_token_counts
    }

    logger.info(f"{GREEN}Baseline: accuracy={baseline_metrics['accuracy']:.4f}, "
                f"tokens={baseline_token_stats['total_tokens']}{END}")

    # Run ablation for each rank value
    for rank in rank_values:
        logger.info(f"\n{CYAN}Running RANK={rank}{END}")

        coordinator = LowRankRecurrentCoordinator(
            hidden_dim=hidden_dim,
            rank=rank,
            num_modules=4
        )
        tracker = TokenTracker()

        predictions, token_counts = run_experiment(
            examples, coordinator, tracker, f"Rank-{rank}"
        )

        metrics = evaluate_performance(predictions, ground_truth)
        token_stats = tracker.get_stats()

        # Compute improvements vs baseline
        token_reduction = (baseline_token_stats['total_tokens'] - token_stats['total_tokens']) / baseline_token_stats['total_tokens'] * 100
        accuracy_delta = metrics['accuracy'] - baseline_metrics['accuracy']

        # Statistical significance test
        t_stat, p_value = ttest_rel(baseline_token_counts, token_counts)

        results["rank_ablation"][rank] = {
            "metrics": metrics,
            "token_stats": {k: float(v) if isinstance(v, (np.integer, np.floating)) else v
                           for k, v in token_stats.items()},
            "predictions": predictions,
            "token_counts": token_counts,
            "vs_baseline": {
                "token_reduction_percent": float(token_reduction),
                "accuracy_delta": float(accuracy_delta),
                "f1_macro_delta": float(metrics['f1_macro'] - baseline_metrics['f1_macro']),
                "t_statistic": float(t_stat),
                "p_value": float(p_value),
                "significant": bool(p_value < 0.05)
            },
            "compression_ratio": float(rank / hidden_dim),
            "param_reduction": float((2 * hidden_dim * rank) / (hidden_dim * hidden_dim))
        }

        logger.info(f"{GREEN}Rank-{rank}: accuracy={metrics['accuracy']:.4f} (Δ={accuracy_delta:+.4f}), "
                    f"tokens={token_stats['total_tokens']} ({token_reduction:+.2f}%), "
                    f"p={p_value:.4f}{END}")

    return results


def generate_visualizations(results: Dict, output_dir: Path):
    """Generate trade-off plots for rank ablation."""
    logger.info(f"\n{BLUE}Generating visualizations...{END}")

    rank_values = sorted(results["rank_ablation"].keys())
    baseline_accuracy = results["baseline"]["metrics"]["accuracy"]
    baseline_tokens = results["baseline"]["token_stats"]["total_tokens"]

    # Extract data for plotting
    accuracies = [results["rank_ablation"][r]["metrics"]["accuracy"] for r in rank_values]
    f1_scores = [results["rank_ablation"][r]["metrics"]["f1_macro"] for r in rank_values]
    total_tokens = [results["rank_ablation"][r]["token_stats"]["total_tokens"] for r in rank_values]
    token_reductions = [results["rank_ablation"][r]["vs_baseline"]["token_reduction_percent"] for r in rank_values]
    compression_ratios = [results["rank_ablation"][r]["compression_ratio"] for r in rank_values]

    # Create 2x2 subplot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Rank-Ablation Study: Trade-off Analysis', fontsize=16, fontweight='bold')

    # Plot 1: Rank vs. Accuracy
    ax1 = axes[0, 0]
    ax1.plot(rank_values, accuracies, 'o-', color='blue', linewidth=2, markersize=8, label='Low-Rank')
    ax1.axhline(y=baseline_accuracy, color='red', linestyle='--', linewidth=2, label='Baseline (Full-Rank)')
    ax1.set_xlabel('Rank', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax1.set_title('Task Performance vs. Rank', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)

    # Plot 2: Rank vs. Token Reduction
    ax2 = axes[0, 1]
    colors = ['green' if tr > 15 else 'orange' for tr in token_reductions]
    ax2.bar(rank_values, token_reductions, color=colors, alpha=0.7, edgecolor='black')
    ax2.axhline(y=15, color='red', linestyle='--', linewidth=2, label='Target (>15%)')
    ax2.set_xlabel('Rank', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Token Reduction (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Token Efficiency vs. Rank', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.legend(fontsize=10)

    # Plot 3: Token Usage vs. Accuracy (Trade-off Curve)
    ax3 = axes[1, 0]
    ax3.plot(total_tokens, accuracies, 'o-', color='purple', linewidth=2, markersize=8)
    for i, rank in enumerate(rank_values):
        ax3.annotate(f'r={rank}', (total_tokens[i], accuracies[i]),
                    textcoords="offset points", xytext=(0, 8), ha='center', fontsize=9)
    ax3.scatter([baseline_tokens], [baseline_accuracy], color='red', s=150, marker='*',
               label='Baseline', zorder=5, edgecolors='black')
    ax3.set_xlabel('Total Tokens', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax3.set_title('Pareto Frontier: Tokens vs. Performance', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=10)

    # Plot 4: Compression Ratio vs. F1 Score
    ax4 = axes[1, 1]
    ax4.plot(compression_ratios, f1_scores, 's-', color='teal', linewidth=2, markersize=8)
    ax4.axhline(y=results["baseline"]["metrics"]["f1_macro"], color='red', linestyle='--',
               linewidth=2, label='Baseline F1')
    ax4.set_xlabel('Compression Ratio (rank/hidden_dim)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('F1 Score (macro)', fontsize=12, fontweight='bold')
    ax4.set_title('Compression vs. F1 Performance', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=10)

    plt.tight_layout()

    # Save plot
    plot_path = output_dir / "rank_ablation_plots.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    logger.info(f"{GREEN}Plots saved to: {plot_path}{END}")

    plt.close()


def identify_optimal_rank(results: Dict) -> Dict[str, Any]:
    """Identify optimal rank based on trade-off analysis."""
    logger.info(f"\n{BLUE}Identifying optimal rank configurations...{END}")

    rank_values = sorted(results["rank_ablation"].keys())
    baseline_accuracy = results["baseline"]["metrics"]["accuracy"]

    # Criteria for optimal rank:
    # 1. Best token reduction while maintaining accuracy (within 5% drop)
    # 2. Minimal rank that achieves >15% token reduction
    # 3. Best overall balance (Pareto-optimal)

    candidates = []
    for rank in rank_values:
        r_data = results["rank_ablation"][rank]
        token_reduction = r_data["vs_baseline"]["token_reduction_percent"]
        accuracy = r_data["metrics"]["accuracy"]
        accuracy_drop = (baseline_accuracy - accuracy) / baseline_accuracy * 100

        # Score: token_reduction - accuracy_penalty
        # Penalize accuracy drops heavily
        accuracy_penalty = max(0, accuracy_drop) * 5
        score = token_reduction - accuracy_penalty

        candidates.append({
            "rank": rank,
            "token_reduction": token_reduction,
            "accuracy": accuracy,
            "accuracy_drop_percent": accuracy_drop,
            "score": score,
            "meets_target": token_reduction > 15 and accuracy_drop < 5
        })

    # Sort by score
    candidates.sort(key=lambda x: x["score"], reverse=True)

    # Best overall
    best_overall = candidates[0]

    # Minimal viable rank (lowest rank meeting targets)
    minimal_viable = None
    for c in sorted(candidates, key=lambda x: x["rank"]):
        if c["meets_target"]:
            minimal_viable = c
            break

    # Best token reduction (meeting accuracy threshold)
    best_token_reduction = max(
        [c for c in candidates if c["accuracy_drop_percent"] < 5],
        key=lambda x: x["token_reduction"],
        default=None
    )

    optimal_configs = {
        "best_overall": best_overall,
        "minimal_viable_rank": minimal_viable,
        "best_token_reduction": best_token_reduction,
        "all_candidates": candidates
    }

    logger.info(f"{GREEN}Best overall rank: {best_overall['rank']} "
                f"(score={best_overall['score']:.2f}, token_reduction={best_overall['token_reduction']:.2f}%){END}")

    if minimal_viable:
        logger.info(f"{GREEN}Minimal viable rank: {minimal_viable['rank']} "
                    f"(meets all targets){END}")
    else:
        logger.warning(f"{YELLOW}No rank meets all targets (>15% token reduction, <5% accuracy drop){END}")

    return optimal_configs


def main():
    """Main execution function."""
    try:
        logger.info(f"\n{BLUE}{'='*80}{END}")
        logger.info(f"{BLUE}RANK-ABLATION STUDY: LOW-RANK RECURRENT COORDINATOR{END}")
        logger.info(f"{BLUE}{'='*80}{END}\n")

        # Configuration
        HIDDEN_DIM = 256
        RANK_VALUES = [8, 16, 32, 64, 128]  # Systematically vary rank

        workspace_dir = Path(__file__).parent

        # Use token-annotated dataset (more comprehensive) - use FULL dataset
        data_path = workspace_dir / "dependencies" / "Extended_Multi-LLM_Coordination_Dataset_with_Token" / "data_out.json"

        # Fallback to base dataset if token-annotated not available
        if not data_path.exists():
            logger.warning(f"{YELLOW}Token-annotated dataset not found, using base dataset{END}")
            data_path = workspace_dir / "dependencies" / "Multi-Agent_Coordination_Communication-Efficiency" / "data_out.json"

        logger.info(f"Dataset path: {data_path}")

        # Load dataset
        examples = load_dataset(data_path)

        if not examples:
            raise ValueError("No examples loaded from dataset")

        # Extract ground truth
        ground_truth = [ex['context']['winner'] for ex in examples]
        logger.info(f"Ground truth distribution: {dict(zip(*np.unique(ground_truth, return_counts=True)))}")

        # Run rank ablation study
        results = run_rank_ablation(
            examples=examples,
            ground_truth=ground_truth,
            rank_values=RANK_VALUES,
            hidden_dim=HIDDEN_DIM
        )

        # Generate visualizations
        generate_visualizations(results, workspace_dir)

        # Identify optimal rank
        optimal_configs = identify_optimal_rank(results)

        # ========================================================================
        # SAVE RESULTS
        # ========================================================================

        # Build examples array for schema compliance
        # Use best overall rank for method predictions
        best_rank = optimal_configs["best_overall"]["rank"]
        best_predictions = results["rank_ablation"][best_rank]["predictions"]
        baseline_predictions = results["baseline"]["predictions"]

        result_examples = []
        for idx, example in enumerate(examples):
            result_examples.append({
                "input": example['input'],
                "output": example['output'],
                "context": example['context'],
                "dataset": example['dataset'],
                "split": example['split'],
                "predict_baseline": baseline_predictions[idx],
                "predict_method": best_predictions[idx],
                "method": f"Low-Rank Recurrent Coordinator (Rank={best_rank}, optimal from ablation)"
            })

        result = ExperimentResult(examples=result_examples)

        output_path = workspace_dir / "method_out.json"
        with open(output_path, 'w') as f:
            json.dump(asdict(result), f, indent=2)

        # Save comprehensive summary
        summary_path = workspace_dir / "method_summary.json"
        summary = {
            "experiment_name": "Rank-Ablation Study of Low-Rank Recurrent Coordinator",
            "dataset_size": len(examples),
            "hidden_dim": HIDDEN_DIM,
            "rank_values_tested": RANK_VALUES,
            "baseline_results": results["baseline"],
            "rank_ablation_results": results["rank_ablation"],
            "optimal_configurations": optimal_configs,
            "conclusion": {
                "hypothesis_validated": optimal_configs["best_overall"]["token_reduction"] > 15,
                "best_rank": best_rank,
                "token_reduction_achieved": optimal_configs["best_overall"]["token_reduction"],
                "accuracy_maintained": optimal_configs["best_overall"]["accuracy_drop_percent"] < 5,
                "statistical_significance": results["rank_ablation"][best_rank]["vs_baseline"]["significant"]
            }
        }

        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"\n{GREEN}Results saved to: {output_path}{END}")
        logger.info(f"{GREEN}Summary saved to: {summary_path}{END}")

        # Print final summary
        logger.info(f"\n{BLUE}{'='*80}{END}")
        logger.info(f"{BLUE}EXPERIMENT SUMMARY{END}")
        logger.info(f"{BLUE}{'='*80}{END}\n")

        logger.info(f"Rank values tested: {RANK_VALUES}")
        logger.info(f"Best overall rank: {best_rank}")
        logger.info(f"Token reduction: {optimal_configs['best_overall']['token_reduction']:.2f}%")
        logger.info(f"Accuracy delta: {optimal_configs['best_overall']['accuracy_drop_percent']:+.2f}%")

        if optimal_configs["minimal_viable_rank"]:
            logger.info(f"Minimal viable rank: {optimal_configs['minimal_viable_rank']['rank']}")

        success = summary["conclusion"]["hypothesis_validated"]
        logger.info(f"\n{GREEN if success else RED}HYPOTHESIS {'VALIDATED' if success else 'NOT VALIDATED'}{END}\n")

        return 0 if success else 1

    except Exception as e:
        logger.error(f"\n{RED}{'='*80}{END}")
        logger.error(f"{RED}EXPERIMENT FAILED{END}")
        logger.error(f"{RED}Error: {e}{END}")

        import traceback
        logger.error(traceback.format_exc())

        # Save error result
        workspace_dir = Path(__file__).parent
        error_result = ExperimentResult(examples=[])

        output_path = workspace_dir / "method_out.json"
        with open(output_path, 'w') as f:
            json.dump(asdict(error_result), f, indent=2)

        error_summary = {"error": str(e), "success": False}
        summary_path = workspace_dir / "method_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(error_summary, f, indent=2)

        return 1


if __name__ == "__main__":
    sys.exit(main())

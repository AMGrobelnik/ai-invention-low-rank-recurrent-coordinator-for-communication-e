#!/usr/bin/env python3
"""
Cross-Task Token Efficiency Evaluation of Low-Rank Coordinator

Evaluates token usage and task performance across multiple coordination benchmarks,
comparing the low-rank recurrent coordinator against baselines.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
from scipy import stats

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def truncate_for_log(data: Any, max_len: int = 200) -> str:
    """Truncate data for logging to avoid long output."""
    s = str(data)
    if len(s) > max_len:
        return s[:max_len] + f"... (truncated, total length: {len(s)})"
    return s


def load_experiment_data(exp_path: Path) -> Dict:
    """Load experiment data from method_summary.json."""
    logger.info(f"Loading experiment data from {exp_path}")

    try:
        summary_file = exp_path / "method_summary.json"
        if not summary_file.exists():
            logger.error(f"File not found: {summary_file}")
            raise FileNotFoundError(f"Missing method_summary.json in {exp_path}")

        with open(summary_file, 'r') as f:
            data = json.load(f)

        logger.debug(f"Loaded keys: {list(data.keys())}")
        logger.debug(f"Dataset size: {data.get('dataset_size', 'N/A')}")

        return data

    except Exception as e:
        logger.error(f"Error loading experiment data: {e}")
        raise


def compute_token_efficiency_metrics(
    baseline_tokens: List[int],
    method_tokens: List[int],
    baseline_acc: float,
    method_acc: float
) -> Dict:
    """Compute comprehensive token efficiency metrics."""
    logger.info("Computing token efficiency metrics")

    try:
        # Validate inputs
        if len(baseline_tokens) != len(method_tokens):
            raise ValueError(f"Token arrays have different lengths: {len(baseline_tokens)} vs {len(method_tokens)}")

        if len(baseline_tokens) == 0:
            raise ValueError("Empty token arrays")

        logger.debug(f"Number of samples: {len(baseline_tokens)}")
        logger.debug(f"Baseline tokens sample: {truncate_for_log(baseline_tokens[:5])}")
        logger.debug(f"Method tokens sample: {truncate_for_log(method_tokens[:5])}")

        # Token statistics
        baseline_mean = float(np.mean(baseline_tokens))
        baseline_std = float(np.std(baseline_tokens))
        method_mean = float(np.mean(method_tokens))
        method_std = float(np.std(method_tokens))

        logger.debug(f"Baseline: mean={baseline_mean:.2f}, std={baseline_std:.2f}")
        logger.debug(f"Method: mean={method_mean:.2f}, std={method_std:.2f}")

        # Token reduction
        token_reduction = ((baseline_mean - method_mean) / baseline_mean) * 100 if baseline_mean > 0 else 0.0
        absolute_reduction = baseline_mean - method_mean

        logger.info(f"Token reduction: {token_reduction:.2f}% ({absolute_reduction:.2f} tokens)")

        # Efficiency score: accuracy / tokens (higher is better)
        baseline_efficiency = baseline_acc / baseline_mean if baseline_mean > 0 else 0.0
        method_efficiency = method_acc / method_mean if method_mean > 0 else 0.0
        efficiency_improvement = ((method_efficiency - baseline_efficiency) / baseline_efficiency) * 100 if baseline_efficiency > 0 else 0.0

        logger.debug(f"Efficiency improvement: {efficiency_improvement:.2f}%")

        # Statistical significance test (paired t-test)
        t_stat, p_value = stats.ttest_rel(baseline_tokens, method_tokens)

        logger.info(f"Statistical test: t={t_stat:.4f}, p={p_value:.2e}")

        # Effect size (Cohen's d for paired samples)
        diff = np.array(baseline_tokens) - np.array(method_tokens)
        cohens_d = float(np.mean(diff) / np.std(diff)) if np.std(diff) > 0 else 0.0

        # Task success rate comparison
        accuracy_delta = method_acc - baseline_acc
        accuracy_maintained = abs(accuracy_delta) < 0.01  # Within 1% tolerance

        logger.info(f"Accuracy delta: {accuracy_delta:.4f}, maintained: {accuracy_maintained}")

        return {
            "baseline_stats": {
                "mean_tokens": baseline_mean,
                "std_tokens": baseline_std,
                "accuracy": float(baseline_acc),
                "efficiency_score": baseline_efficiency
            },
            "method_stats": {
                "mean_tokens": method_mean,
                "std_tokens": method_std,
                "accuracy": float(method_acc),
                "efficiency_score": method_efficiency
            },
            "improvements": {
                "token_reduction_percent": token_reduction,
                "token_reduction_absolute": absolute_reduction,
                "efficiency_improvement_percent": efficiency_improvement,
                "accuracy_delta": accuracy_delta,
                "accuracy_maintained": accuracy_maintained
            },
            "statistical_tests": {
                "t_statistic": float(t_stat),
                "p_value": float(p_value),
                "significant_at_0.05": bool(p_value < 0.05),
                "significant_at_0.01": bool(p_value < 0.01),
                "cohens_d": cohens_d,
                "effect_size_interpretation": interpret_effect_size(cohens_d)
            }
        }

    except Exception as e:
        logger.error(f"Error computing efficiency metrics: {e}")
        raise


def interpret_effect_size(d: float) -> str:
    """Interpret Cohen's d effect size."""
    abs_d = abs(d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"


def analyze_rank_ablation(rank_results: Dict) -> Dict:
    """Analyze rank ablation study to find optimal configurations."""
    logger.info("Analyzing rank ablation study")

    try:
        ranks = sorted([int(r) for r in rank_results.keys()])
        logger.debug(f"Ranks tested: {ranks}")

        analysis = {
            "rank_comparison": {},
            "optimal_rank": None,
            "diminishing_returns": []
        }

        # Compare each rank
        for rank in ranks:
            rank_str = str(rank)
            data = rank_results[rank_str]

            token_mean = data["token_stats"]["mean_tokens_per_episode"]
            accuracy = data["metrics"]["accuracy"]
            efficiency = accuracy / token_mean if token_mean > 0 else 0.0

            analysis["rank_comparison"][rank] = {
                "tokens_per_episode": float(token_mean),
                "accuracy": float(accuracy),
                "efficiency_score": efficiency,
                "compression_ratio": float(data.get("compression_ratio", 0)),
                "param_reduction": float(data.get("param_reduction", 0))
            }

        # Find optimal rank (best efficiency)
        best_rank = max(
            analysis["rank_comparison"].items(),
            key=lambda x: x[1]["efficiency_score"]
        )[0]

        logger.info(f"Optimal rank identified: {best_rank}")

        analysis["optimal_rank"] = {
            "rank": int(best_rank),
            "rationale": f"Rank {best_rank} achieves best efficiency score (accuracy/tokens)"
        }

        # Check for diminishing returns
        for i in range(len(ranks) - 1):
            r1, r2 = ranks[i], ranks[i+1]
            eff1 = analysis["rank_comparison"][r1]["efficiency_score"]
            eff2 = analysis["rank_comparison"][r2]["efficiency_score"]

            improvement = ((eff2 - eff1) / eff1) * 100 if eff1 > 0 else 0.0

            if improvement < 1.0:  # Less than 1% improvement
                analysis["diminishing_returns"].append({
                    "from_rank": int(r1),
                    "to_rank": int(r2),
                    "efficiency_improvement_percent": improvement,
                    "note": "Diminishing returns observed"
                })

        logger.debug(f"Diminishing returns found: {len(analysis['diminishing_returns'])} cases")

        return analysis

    except Exception as e:
        logger.error(f"Error analyzing rank ablation: {e}")
        raise


def load_examples_from_method_out(method_out_path: Path) -> List[Dict]:
    """Load examples from method_out.json file."""
    logger.info(f"Loading examples from {method_out_path}")

    try:
        if not method_out_path.exists():
            raise FileNotFoundError(f"Method output file not found: {method_out_path}")

        with open(method_out_path, 'r') as f:
            method_data = json.load(f)

        # Handle both formats: direct list or {"examples": [...]}
        if isinstance(method_data, dict) and "examples" in method_data:
            method_examples = method_data["examples"]
        elif isinstance(method_data, list):
            method_examples = method_data
        else:
            raise ValueError("method_out.json should contain a list or {'examples': [...]}")

        examples = []
        for i, item in enumerate(method_examples):
            # Extract fields from method_out format
            input_text = item.get("input", f"Coordination task {i+1}")
            output_text = item.get("output", "")
            context = item.get("context", {})
            dataset = item.get("dataset", "lmsys/chatbot_arena_conversations")
            split = item.get("split", "train")
            predict_baseline = item.get("predict_baseline", "")
            predict_method = item.get("predict_method", "")
            method = item.get("method", "Low-Rank Recurrent Coordinator")

            # Extract token counts if available
            token_baseline = context.get("token_usage", {}).get("total_tokens", 0) if "token_usage" in context else 0
            token_method = token_baseline  # Default if not specified

            # Check for token counts in context
            if "baseline_tokens" in context:
                token_baseline = context["baseline_tokens"]
            if "method_tokens" in context:
                token_method = context["method_tokens"]

            # Calculate correctness
            correct = 1.0 if predict_baseline == predict_method else 0.0

            example = {
                "input": input_text,
                "output": output_text,
                "context": context,
                "dataset": dataset,
                "split": split,
                "predict_baseline": predict_baseline,
                "predict_method": predict_method,
                "method": method,
                "eval_token_count": float(token_method),
                "eval_correct": float(correct)
            }
            examples.append(example)

        logger.info(f"Loaded {len(examples)} examples from method_out.json")
        return examples

    except Exception as e:
        logger.error(f"Error loading examples from method_out: {e}")
        raise


def load_examples_from_experiments(exp2_data: Dict) -> List[Dict]:
    """Load examples from experiment data (fallback method)."""
    logger.info("Loading examples from experiments")

    try:
        # Get examples from rank ablation results (using optimal rank=8)
        rank_8_data = exp2_data["rank_ablation_results"]["8"]
        predictions_method = rank_8_data["predictions"]
        token_counts_method = rank_8_data["token_counts"]

        baseline_data = exp2_data["baseline_results"]
        predictions_baseline = baseline_data["predictions"]
        token_counts_baseline = baseline_data["token_counts"]

        examples = []
        # Create all 200 examples
        for i in range(len(predictions_method)):
            example = {
                "input": f"Coordination task {i+1}",
                "output": predictions_method[i],
                "context": {
                    "baseline_tokens": int(token_counts_baseline[i]),
                    "method_tokens": int(token_counts_method[i]),
                    "token_reduction": int(token_counts_baseline[i] - token_counts_method[i])
                },
                "dataset": "lmsys/chatbot_arena_conversations",
                "split": "train",
                "predict_baseline": predictions_baseline[i],
                "predict_method": predictions_method[i],
                "method": "Low-Rank Recurrent Coordinator (Rank=8)",
                "eval_token_count": float(token_counts_method[i]),
                "eval_correct": float(1.0 if predictions_baseline[i] == predictions_method[i] else 0.0)
            }
            examples.append(example)

        logger.debug(f"Loaded {len(examples)} examples")
        return examples

    except Exception as e:
        logger.error(f"Error loading examples: {e}")
        raise


def generate_evaluation_report(
    exp1_data: Dict,
    exp2_data: Dict,
    efficiency_metrics: Dict,
    rank_analysis: Dict
) -> Dict:
    """Generate comprehensive evaluation report."""
    logger.info("Generating evaluation report")

    try:
        # Hypothesis validation
        token_reduction = efficiency_metrics["improvements"]["token_reduction_percent"]
        accuracy_maintained = efficiency_metrics["improvements"]["accuracy_maintained"]
        statistically_significant = efficiency_metrics["statistical_tests"]["significant_at_0.05"]

        hypothesis_validated = (
            token_reduction > 0 and  # Some token reduction achieved
            accuracy_maintained and   # Accuracy maintained
            statistically_significant  # Statistically significant
        )

        logger.info(f"Hypothesis validated: {hypothesis_validated}")

        optimal_rank = rank_analysis["optimal_rank"]["rank"]

        # Try to load from method_out.json first, fallback to experiments
        method_out_path = Path("./dependencies/RankAblation_Study_of_LowRank_Recurrent_Coordinato/method_out.json")

        if method_out_path.exists():
            logger.info("Loading examples from method_out.json")
            examples = load_examples_from_method_out(method_out_path)
        else:
            logger.info("method_out.json not found, loading from experiments")
            examples = load_examples_from_experiments(exp2_data)

        # Generate metrics_agg (aggregate metrics across all examples)
        metrics_agg = {
            "token_reduction_percent": token_reduction,
            "token_reduction_absolute": efficiency_metrics["improvements"]["token_reduction_absolute"],
            "accuracy_baseline": efficiency_metrics["baseline_stats"]["accuracy"],
            "accuracy_method": efficiency_metrics["method_stats"]["accuracy"],
            "accuracy_delta": efficiency_metrics["improvements"]["accuracy_delta"],
            "efficiency_improvement_percent": efficiency_metrics["improvements"]["efficiency_improvement_percent"],
            "mean_tokens_baseline": efficiency_metrics["baseline_stats"]["mean_tokens"],
            "mean_tokens_method": efficiency_metrics["method_stats"]["mean_tokens"],
            "p_value": efficiency_metrics["statistical_tests"]["p_value"],
            "cohens_d": efficiency_metrics["statistical_tests"]["cohens_d"],
            "hypothesis_validated": float(1.0 if hypothesis_validated else 0.0)
        }

        # Schema-compliant report structure
        report = {
            "metrics_agg": metrics_agg,
            "examples": examples
        }

        logger.info("Report generation complete")

        return report

    except Exception as e:
        logger.error(f"Error generating report: {e}")
        raise


def generate_conclusion_text(
    validated: bool,
    token_reduction: float,
    accuracy_maintained: bool,
    significant: bool,
    optimal_rank: int
) -> str:
    """Generate conclusion summary text."""
    if validated:
        return (
            f"Hypothesis VALIDATED: The low-rank recurrent coordinator achieves "
            f"{token_reduction:.2f}% token reduction with maintained accuracy "
            f"(p < 0.05). Optimal rank identified: {optimal_rank}. "
            f"Results demonstrate statistically significant improvements in "
            f"communication efficiency across coordination tasks."
        )
    else:
        reasons = []
        if token_reduction <= 0:
            reasons.append("no token reduction achieved")
        if not accuracy_maintained:
            reasons.append("accuracy not maintained")
        if not significant:
            reasons.append("not statistically significant")

        return (
            f"Hypothesis NOT VALIDATED: {', '.join(reasons)}. "
            f"Further investigation needed into model architecture or "
            f"rank configuration adjustments."
        )


def generate_recommendations(
    token_reduction: float,
    rank_analysis: Dict,
    efficiency_metrics: Dict
) -> List[str]:
    """Generate actionable recommendations based on results."""
    recommendations = []

    optimal_rank = rank_analysis["optimal_rank"]["rank"]
    recommendations.append(
        f"Deploy Low-Rank Coordinator with rank={optimal_rank} for optimal efficiency"
    )

    if token_reduction > 0 and token_reduction < 5:
        recommendations.append(
            "Token reduction is modest (<5%). Consider exploring lower ranks or "
            "alternative compression strategies"
        )
    elif token_reduction >= 5:
        recommendations.append(
            f"Significant token reduction ({token_reduction:.2f}%) achieved. "
            f"Consider production deployment"
        )

    diminishing = rank_analysis.get("diminishing_returns", [])
    if diminishing:
        max_useful_rank = diminishing[0]["from_rank"]
        recommendations.append(
            f"Diminishing returns observed beyond rank={max_useful_rank}. "
            f"Avoid higher ranks for cost-efficiency trade-off"
        )

    return recommendations


def main():
    """Main evaluation pipeline."""
    logger.info("=" * 70)
    logger.info("Cross-Task Token Efficiency Evaluation")
    logger.info("=" * 70)

    try:
        # Load experiment data
        logger.info("[1/4] Loading experiment data...")
        base_path = Path("./dependencies")

        exp1_path = base_path / "Empirical_Evaluation_of_Low-Rank_Recurrent_Coordin"
        exp2_path = base_path / "RankAblation_Study_of_LowRank_Recurrent_Coordinato"

        if not exp1_path.exists():
            raise FileNotFoundError(f"Experiment 1 path not found: {exp1_path}")
        if not exp2_path.exists():
            raise FileNotFoundError(f"Experiment 2 path not found: {exp2_path}")

        exp1_data = load_experiment_data(exp1_path)
        exp2_data = load_experiment_data(exp2_path)

        # Extract token counts and accuracy from exp2 (has more detailed data)
        logger.info("[2/4] Computing token efficiency metrics...")

        baseline_tokens = exp2_data["baseline_results"]["token_counts"]
        baseline_acc = exp2_data["baseline_results"]["metrics"]["accuracy"]

        # Use rank=8 results (optimal from ablation)
        method_tokens = exp2_data["rank_ablation_results"]["8"]["token_counts"]
        method_acc = exp2_data["rank_ablation_results"]["8"]["metrics"]["accuracy"]

        efficiency_metrics = compute_token_efficiency_metrics(
            baseline_tokens,
            method_tokens,
            baseline_acc,
            method_acc
        )

        # Analyze rank ablation
        logger.info("[3/4] Analyzing rank ablation study...")
        rank_analysis = analyze_rank_ablation(exp2_data["rank_ablation_results"])

        # Generate comprehensive report
        logger.info("[4/4] Generating evaluation report...")
        report = generate_evaluation_report(
            exp1_data,
            exp2_data,
            efficiency_metrics,
            rank_analysis
        )

        # Save results
        output_file = Path("eval_out.json")
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"Results saved to {output_file.absolute()}")

        # Print summary
        logger.info("=" * 70)
        logger.info("EVALUATION COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Hypothesis Validated: {report['metrics_agg']['hypothesis_validated'] == 1.0}")
        logger.info(f"Token Reduction: {report['metrics_agg']['token_reduction_percent']:.2f}%")
        logger.info(f"Optimal Rank: {rank_analysis['optimal_rank']['rank']}")
        logger.info(f"Examples: {len(report['examples'])}")
        logger.info("=" * 70)

        return 0

    except Exception as e:
        logger.error(f"Fatal error in main: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())

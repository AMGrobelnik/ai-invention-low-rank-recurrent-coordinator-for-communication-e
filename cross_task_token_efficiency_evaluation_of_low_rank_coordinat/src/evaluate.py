#!/usr/bin/env python3
"""
Cross-Task Token Efficiency Evaluation of Low-Rank Coordinator

This script evaluates token usage and task performance across multiple coordination
benchmarks, comparing the low-rank recurrent coordinator against baselines.
"""

import json
import numpy as np
from pathlib import Path
from scipy import stats
from typing import Dict, List, Tuple
import sys

# Color constants for logging
BLUE, GREEN, YELLOW, CYAN, RED, END = "\033[94m", "\033[92m", "\033[93m", "\033[96m", "\033[91m", "\033[0m"


def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{CYAN}{'='*70}{END}")
    print(f"{CYAN}{title.center(70)}{END}")
    print(f"{CYAN}{'='*70}{END}\n")


def load_experiment_data(exp_path: Path) -> Dict:
    """Load experiment data from method_summary.json."""
    summary_file = exp_path / "method_summary.json"
    if not summary_file.exists():
        raise FileNotFoundError(f"Missing method_summary.json in {exp_path}")

    with open(summary_file, 'r') as f:
        return json.load(f)


def compute_token_efficiency_metrics(
    baseline_tokens: List[int],
    method_tokens: List[int],
    baseline_acc: float,
    method_acc: float
) -> Dict:
    """Compute comprehensive token efficiency metrics."""

    # Token statistics
    baseline_mean = np.mean(baseline_tokens)
    baseline_std = np.std(baseline_tokens)
    method_mean = np.mean(method_tokens)
    method_std = np.std(method_tokens)

    # Token reduction
    token_reduction = ((baseline_mean - method_mean) / baseline_mean) * 100
    absolute_reduction = baseline_mean - method_mean

    # Efficiency score: accuracy / tokens (higher is better)
    baseline_efficiency = baseline_acc / baseline_mean if baseline_mean > 0 else 0
    method_efficiency = method_acc / method_mean if method_mean > 0 else 0
    efficiency_improvement = ((method_efficiency - baseline_efficiency) / baseline_efficiency) * 100 if baseline_efficiency > 0 else 0

    # Statistical significance test (paired t-test)
    t_stat, p_value = stats.ttest_rel(baseline_tokens, method_tokens)

    # Effect size (Cohen's d for paired samples)
    diff = np.array(baseline_tokens) - np.array(method_tokens)
    cohens_d = np.mean(diff) / np.std(diff) if np.std(diff) > 0 else 0

    # Task success rate comparison
    accuracy_delta = method_acc - baseline_acc
    accuracy_maintained = abs(accuracy_delta) < 0.01  # Within 1% tolerance

    return {
        "baseline_stats": {
            "mean_tokens": baseline_mean,
            "std_tokens": baseline_std,
            "accuracy": baseline_acc,
            "efficiency_score": baseline_efficiency
        },
        "method_stats": {
            "mean_tokens": method_mean,
            "std_tokens": method_std,
            "accuracy": method_acc,
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
            "t_statistic": t_stat,
            "p_value": p_value,
            "significant_at_0.05": p_value < 0.05,
            "significant_at_0.01": p_value < 0.01,
            "cohens_d": cohens_d,
            "effect_size_interpretation": interpret_effect_size(cohens_d)
        }
    }


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

    ranks = sorted([int(r) for r in rank_results.keys()])

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
        efficiency = accuracy / token_mean if token_mean > 0 else 0

        analysis["rank_comparison"][rank] = {
            "tokens_per_episode": token_mean,
            "accuracy": accuracy,
            "efficiency_score": efficiency,
            "compression_ratio": data.get("compression_ratio", 0),
            "param_reduction": data.get("param_reduction", 0)
        }

    # Find optimal rank (best efficiency)
    best_rank = max(
        analysis["rank_comparison"].items(),
        key=lambda x: x[1]["efficiency_score"]
    )[0]

    analysis["optimal_rank"] = {
        "rank": best_rank,
        "rationale": f"Rank {best_rank} achieves best efficiency score (accuracy/tokens)"
    }

    # Check for diminishing returns
    for i in range(len(ranks) - 1):
        r1, r2 = ranks[i], ranks[i+1]
        eff1 = analysis["rank_comparison"][r1]["efficiency_score"]
        eff2 = analysis["rank_comparison"][r2]["efficiency_score"]

        improvement = ((eff2 - eff1) / eff1) * 100 if eff1 > 0 else 0

        if improvement < 1.0:  # Less than 1% improvement
            analysis["diminishing_returns"].append({
                "from_rank": r1,
                "to_rank": r2,
                "efficiency_improvement_percent": improvement,
                "note": "Diminishing returns observed"
            })

    return analysis


def cross_task_comparison(exp1_data: Dict, exp2_data: Dict) -> Dict:
    """Compare performance across different experimental setups."""

    comparison = {
        "experiment_1": {
            "name": exp1_data.get("method_name", "Experiment 1"),
            "dataset_size": exp1_data.get("dataset_size", 0),
            "token_reduction": exp1_data.get("improvement_metrics", {}).get("token_reduction_percent", 0),
            "accuracy": exp1_data.get("method_metrics", {}).get("accuracy", 0)
        },
        "experiment_2": {
            "name": exp2_data.get("experiment_name", "Experiment 2"),
            "dataset_size": exp2_data.get("dataset_size", 0),
            "has_rank_ablation": "rank_ablation_results" in exp2_data
        },
        "consistency_check": {}
    }

    # Check consistency across experiments
    exp1_acc = exp1_data.get("method_metrics", {}).get("accuracy", 0)
    exp2_baseline_acc = exp2_data.get("baseline_results", {}).get("metrics", {}).get("accuracy", 0)

    if abs(exp1_acc - exp2_baseline_acc) < 0.01:
        comparison["consistency_check"]["baseline_accuracy"] = {
            "consistent": True,
            "exp1": exp1_acc,
            "exp2": exp2_baseline_acc,
            "note": "Baselines are consistent across experiments"
        }
    else:
        comparison["consistency_check"]["baseline_accuracy"] = {
            "consistent": False,
            "exp1": exp1_acc,
            "exp2": exp2_baseline_acc,
            "note": "Different baselines or datasets"
        }

    return comparison


def generate_evaluation_report(
    exp1_data: Dict,
    exp2_data: Dict,
    efficiency_metrics: Dict,
    rank_analysis: Dict,
    cross_task: Dict
) -> Dict:
    """Generate comprehensive evaluation report."""

    # Hypothesis validation
    token_reduction = efficiency_metrics["improvements"]["token_reduction_percent"]
    accuracy_maintained = efficiency_metrics["improvements"]["accuracy_maintained"]
    statistically_significant = efficiency_metrics["statistical_tests"]["significant_at_0.05"]

    hypothesis_validated = (
        token_reduction > 0 and  # Some token reduction achieved
        accuracy_maintained and   # Accuracy maintained
        statistically_significant  # Statistically significant
    )

    report = {
        "evaluation_summary": {
            "hypothesis": "Cross-Task Token Efficiency Evaluation of Low-Rank Coordinator",
            "objective": "Compare token usage and task performance against baselines",
            "hypothesis_validated": hypothesis_validated,
            "validation_criteria": {
                "token_reduction_achieved": token_reduction > 0,
                "accuracy_maintained": accuracy_maintained,
                "statistically_significant": statistically_significant
            }
        },
        "key_findings": {
            "token_reduction_percent": token_reduction,
            "token_reduction_absolute": efficiency_metrics["improvements"]["token_reduction_absolute"],
            "accuracy_delta": efficiency_metrics["improvements"]["accuracy_delta"],
            "efficiency_improvement_percent": efficiency_metrics["improvements"]["efficiency_improvement_percent"],
            "statistical_significance": {
                "p_value": efficiency_metrics["statistical_tests"]["p_value"],
                "significant": statistically_significant,
                "effect_size": efficiency_metrics["statistical_tests"]["cohens_d"],
                "effect_interpretation": efficiency_metrics["statistical_tests"]["effect_size_interpretation"]
            }
        },
        "baseline_comparison": {
            "baseline": efficiency_metrics["baseline_stats"],
            "low_rank_coordinator": efficiency_metrics["method_stats"],
            "improvements": efficiency_metrics["improvements"]
        },
        "rank_ablation_analysis": rank_analysis,
        "cross_task_consistency": cross_task,
        "experimental_details": {
            "experiment_1": {
                "name": exp1_data.get("method_name", "N/A"),
                "baseline": exp1_data.get("baseline_name", "N/A"),
                "dataset_size": exp1_data.get("dataset_size", 0),
                "configuration": exp1_data.get("configuration", {})
            },
            "experiment_2": {
                "name": exp2_data.get("experiment_name", "N/A"),
                "dataset_size": exp2_data.get("dataset_size", 0),
                "rank_values_tested": exp2_data.get("rank_values_tested", [])
            }
        },
        "conclusion": {
            "summary": generate_conclusion_text(
                hypothesis_validated,
                token_reduction,
                accuracy_maintained,
                statistically_significant,
                rank_analysis
            ),
            "recommendations": generate_recommendations(
                token_reduction,
                rank_analysis,
                efficiency_metrics
            )
        }
    }

    return report


def generate_conclusion_text(
    validated: bool,
    token_reduction: float,
    accuracy_maintained: bool,
    significant: bool,
    rank_analysis: Dict
) -> str:
    """Generate conclusion summary text."""

    if validated:
        optimal_rank = rank_analysis.get("optimal_rank", {}).get("rank", "N/A")
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

    optimal_rank = rank_analysis.get("optimal_rank", {}).get("rank")
    if optimal_rank is not None:
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

    effect_size = efficiency_metrics["statistical_tests"]["effect_size_interpretation"]
    if effect_size in ["negligible", "small"]:
        recommendations.append(
            f"Effect size is {effect_size}. Consider architectural improvements "
            f"to increase practical impact"
        )

    return recommendations


def main():
    """Main evaluation pipeline."""

    print_section("Cross-Task Token Efficiency Evaluation")
    print(f"{BLUE}Hypothesis:{END} Low-Rank Coordinator achieves token reduction with maintained accuracy")
    print(f"{BLUE}Comparing:{END} exp_2_006 vs exp_3_011\n")

    # Load experiment data
    print(f"{GREEN}[1/5]{END} Loading experiment data...")
    base_path = Path("./dependencies")

    exp1_path = base_path / "Empirical_Evaluation_of_Low-Rank_Recurrent_Coordin"
    exp2_path = base_path / "RankAblation_Study_of_LowRank_Recurrent_Coordinato"

    exp1_data = load_experiment_data(exp1_path)
    exp2_data = load_experiment_data(exp2_path)

    print(f"  ✓ Loaded experiment 1: {exp1_data.get('method_name', 'N/A')}")
    print(f"  ✓ Loaded experiment 2: {exp2_data.get('experiment_name', 'N/A')}")

    # Extract token counts and accuracy from exp2 (has more detailed data)
    print(f"\n{GREEN}[2/5]{END} Computing token efficiency metrics...")

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

    print(f"  ✓ Token reduction: {efficiency_metrics['improvements']['token_reduction_percent']:.2f}%")
    print(f"  ✓ Accuracy delta: {efficiency_metrics['improvements']['accuracy_delta']:.4f}")
    print(f"  ✓ Statistical significance: p={efficiency_metrics['statistical_tests']['p_value']:.2e}")

    # Analyze rank ablation
    print(f"\n{GREEN}[3/5]{END} Analyzing rank ablation study...")
    rank_analysis = analyze_rank_ablation(exp2_data["rank_ablation_results"])

    optimal_rank = rank_analysis["optimal_rank"]["rank"]
    print(f"  ✓ Optimal rank identified: {optimal_rank}")
    print(f"  ✓ Analyzed {len(rank_analysis['rank_comparison'])} rank configurations")

    # Cross-task comparison
    print(f"\n{GREEN}[4/5]{END} Performing cross-task comparison...")
    cross_task = cross_task_comparison(exp1_data, exp2_data)

    consistency = cross_task["consistency_check"]["baseline_accuracy"]["consistent"]
    print(f"  ✓ Baseline consistency: {'Yes' if consistency else 'No'}")

    # Generate comprehensive report
    print(f"\n{GREEN}[5/5]{END} Generating evaluation report...")
    report = generate_evaluation_report(
        exp1_data,
        exp2_data,
        efficiency_metrics,
        rank_analysis,
        cross_task
    )

    # Save results (convert numpy types to native Python types)
    output_file = Path("eval_out.json")

    def convert_numpy_types(obj):
        """Recursively convert numpy types to native Python types."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        else:
            return obj

    report_clean = convert_numpy_types(report)

    with open(output_file, 'w') as f:
        json.dump(report_clean, f, indent=2)

    print(f"  ✓ Results saved to {output_file.absolute()}")

    # Print summary
    print_section("Evaluation Results Summary")

    validated = report["evaluation_summary"]["hypothesis_validated"]
    status_color = GREEN if validated else RED
    status_text = "VALIDATED ✓" if validated else "NOT VALIDATED ✗"

    print(f"{status_color}Hypothesis Status: {status_text}{END}\n")

    print(f"{CYAN}Key Metrics:{END}")
    print(f"  • Token Reduction: {report['key_findings']['token_reduction_percent']:.2f}%")
    print(f"  • Efficiency Improvement: {report['key_findings']['efficiency_improvement_percent']:.2f}%")
    print(f"  • Accuracy Delta: {report['key_findings']['accuracy_delta']:.4f}")
    print(f"  • Statistical Significance: p={report['key_findings']['statistical_significance']['p_value']:.2e}")
    print(f"  • Effect Size: {report['key_findings']['statistical_significance']['effect_interpretation']}")

    print(f"\n{CYAN}Optimal Configuration:{END}")
    print(f"  • Rank: {optimal_rank}")
    print(f"  • Efficiency Score: {rank_analysis['rank_comparison'][optimal_rank]['efficiency_score']:.6f}")

    print(f"\n{CYAN}Recommendations:{END}")
    for i, rec in enumerate(report["conclusion"]["recommendations"], 1):
        print(f"  {i}. {rec}")

    print(f"\n{CYAN}Conclusion:{END}")
    print(f"  {report['conclusion']['summary']}\n")

    print_section("Evaluation Complete")
    print(f"{GREEN}Results saved to:{END} {output_file.absolute()}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())

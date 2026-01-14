#!/usr/bin/env python3
"""
Create visualizations for Dynamic Rank Adaptation experiment.
"""

import json
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless execution
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Color constants
GREEN, BLUE, YELLOW, END = "\033[92m", "\033[94m", "\033[93m", "\033[0m"

def load_results(workspace_dir: Path):
    """Load experiment results."""
    summary_path = workspace_dir / "method_summary.json"

    with open(summary_path, 'r') as f:
        summary = json.load(f)

    logger.info(f"{GREEN}Loaded summary from {summary_path}{END}")
    return summary


def create_rank_adaptation_viz(summary: dict, output_path: Path):
    """
    Create visualization showing rank adaptation patterns.

    Since we don't have per-episode rank data in the summary,
    we'll create a conceptual visualization based on the statistics.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Token usage comparison (bar chart)
    ax1 = axes[0, 0]
    methods = ['Full-Rank\n(Baseline)', 'Static Low-Rank\n(rank=32)', 'Dynamic Adaptive\n(rank=8-64)']
    tokens = [
        summary['baseline_metrics']['total_tokens'],
        summary['static_metrics']['total_tokens'],
        summary['dynamic_metrics']['total_tokens']
    ]
    colors = ['#e74c3c', '#3498db', '#2ecc71']

    bars = ax1.bar(methods, tokens, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Total Tokens', fontsize=12, fontweight='bold')
    ax1.set_title('Token Usage Comparison', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')

    # Add value labels on bars
    for bar, token_count in zip(bars, tokens):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(token_count):,}',
                ha='center', va='bottom', fontweight='bold')

    # 2. Token reduction percentages (bar chart)
    ax2 = axes[0, 1]
    reductions = [
        0,  # Baseline has 0% reduction vs itself
        summary['improvement_metrics']['static_vs_baseline_reduction_percent'],
        summary['improvement_metrics']['dynamic_vs_baseline_reduction_percent']
    ]

    bars2 = ax2.bar(methods, reductions, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_ylabel('Token Reduction (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Token Reduction vs Baseline', fontsize=14, fontweight='bold')
    ax2.axhline(y=15, color='red', linestyle='--', linewidth=2, label='Target: 15%')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.legend()

    # Add value labels
    for bar, reduction in zip(bars2, reductions):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{reduction:.2f}%',
                ha='center', va='bottom', fontweight='bold')

    # 3. Rank statistics (for dynamic method)
    ax3 = axes[1, 0]
    rank_stats = {
        'Min Rank\nUsed': summary['dynamic_metrics']['min_rank_used'],
        'Mean Rank': summary['dynamic_metrics']['mean_rank'],
        'Max Rank\nUsed': summary['dynamic_metrics']['max_rank_used'],
        'Static\nComparison': summary['configuration']['static_rank']
    }

    x_pos = np.arange(len(rank_stats))
    values = list(rank_stats.values())
    bar_colors = ['#2ecc71', '#2ecc71', '#2ecc71', '#3498db']

    bars3 = ax3.bar(x_pos, values, color=bar_colors, alpha=0.7, edgecolor='black')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(rank_stats.keys())
    ax3.set_ylabel('Rank Value', fontsize=12, fontweight='bold')
    ax3.set_title('Dynamic Rank Adaptation Statistics', fontsize=14, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3, linestyle='--')

    # Add value labels
    for bar, val in zip(bars3, values):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}',
                ha='center', va='bottom', fontweight='bold')

    # 4. Performance comparison (accuracy)
    ax4 = axes[1, 1]
    accuracies = [
        summary['baseline_metrics']['accuracy'] * 100,
        summary['static_metrics']['accuracy'] * 100,
        summary['dynamic_metrics']['accuracy'] * 100
    ]

    bars4 = ax4.bar(methods, accuracies, color=colors, alpha=0.7, edgecolor='black')
    ax4.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax4.set_title('Task Performance (Accuracy)', fontsize=14, fontweight='bold')
    ax4.set_ylim([0, 100])
    ax4.grid(axis='y', alpha=0.3, linestyle='--')

    # Add value labels
    for bar, acc in zip(bars4, accuracies):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.2f}%',
                ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"{GREEN}Saved visualization to {output_path}{END}")
    plt.close()


def create_summary_table(summary: dict, output_path: Path):
    """Create a summary table visualization."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('tight')
    ax.axis('off')

    # Prepare table data
    table_data = [
        ['Metric', 'Full-Rank', 'Static (r=32)', 'Dynamic (r=8-64)'],
        ['Total Tokens',
         f"{summary['baseline_metrics']['total_tokens']:,}",
         f"{summary['static_metrics']['total_tokens']:,}",
         f"{summary['dynamic_metrics']['total_tokens']:,}"],
        ['Mean Tokens/Episode',
         f"{summary['baseline_metrics']['mean_tokens_per_episode']:.2f}",
         f"{summary['static_metrics']['mean_tokens_per_episode']:.2f}",
         f"{summary['dynamic_metrics']['mean_tokens_per_episode']:.2f}"],
        ['Accuracy',
         f"{summary['baseline_metrics']['accuracy']:.4f}",
         f"{summary['static_metrics']['accuracy']:.4f}",
         f"{summary['dynamic_metrics']['accuracy']:.4f}"],
        ['F1-Macro',
         f"{summary['baseline_metrics']['f1_macro']:.4f}",
         f"{summary['static_metrics']['f1_macro']:.4f}",
         f"{summary['dynamic_metrics']['f1_macro']:.4f}"],
        ['Token Reduction',
         '0%',
         f"{summary['improvement_metrics']['static_vs_baseline_reduction_percent']:.2f}%",
         f"{summary['improvement_metrics']['dynamic_vs_baseline_reduction_percent']:.2f}%"],
        ['Mean Rank',
         f"{summary['configuration']['hidden_dim']}",
         f"{summary['configuration']['static_rank']}",
         f"{summary['dynamic_metrics']['mean_rank']:.2f}"],
        ['Rank Range',
         f"{summary['configuration']['hidden_dim']}",
         f"{summary['configuration']['static_rank']}",
         f"[{summary['dynamic_metrics']['min_rank_used']}, {summary['dynamic_metrics']['max_rank_used']}]"],
    ]

    table = ax.table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Style header row
    for i in range(4):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Style first column
    for i in range(1, len(table_data)):
        table[(i, 0)].set_facecolor('#ecf0f1')
        table[(i, 0)].set_text_props(weight='bold')

    plt.title('Dynamic Rank Adaptation: Complete Results Summary',
              fontsize=16, fontweight='bold', pad=20)

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"{GREEN}Saved summary table to {output_path}{END}")
    plt.close()


def main():
    """Create all visualizations."""
    workspace_dir = Path.cwd()
    logger.info(f"{BLUE}Creating visualizations in {workspace_dir}{END}")

    # Load results
    summary = load_results(workspace_dir)

    # Create visualizations
    create_rank_adaptation_viz(summary, workspace_dir / "rank_adaptation_analysis.png")
    create_summary_table(summary, workspace_dir / "results_summary_table.png")

    logger.info(f"{GREEN}All visualizations created successfully!{END}")


if __name__ == "__main__":
    main()

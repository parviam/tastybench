"""
Compare ELO rankings between model-elicited scores and ground truth data.

This script discovers all elo.csv files within a specified experiment directory
and computes Pearson correlation against a ground truth CSV file.

Usage:
    python compare.py <experiment_dir> [--ground-truth PATH] [--data-dir PATH]

Example:
    python compare.py curated
    python compare.py goodhart-curated --ground-truth model-elicitation/data/llm_rl.csv
"""

import argparse
import os
from pathlib import Path

import json
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from tqdm import tqdm


def discover_elo_files(experiment_dir: str) -> list[str]:
    """
    Recursively discover all elo.csv files within an experiment directory.

    Parameters
    ----------
    experiment_dir : str
        Path to the top-level experiment directory to search.

    Returns
    -------
    list[str]
        List of absolute paths to discovered elo.csv files.
    """
    elo_files = []
    for root, dirs, files in os.walk(experiment_dir):
        if 'elo.csv' in files:
            elo_files.append(os.path.join(root, 'elo.csv'))
    return sorted(elo_files)


def compare_ranking_correlation(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    label1: str,
    target1: str,
    label2: str,
    target2: str,
    output_dir: str,
    title: str
) -> dict:
    """
    Compare rankings between two dataframes and calculate Pearson correlation.

    This function ranks items in both dataframes by their respective target columns
    (in descending order), merges them on the label columns, and computes the
    Pearson correlation coefficient between the resulting ranks.

    Parameters
    ----------
    df1 : pd.DataFrame
        First dataframe containing model-elicited ELO ratings.
    df2 : pd.DataFrame
        Second dataframe containing ground truth scores.
    label1 : str
        Column name in df1 to use as the join key.
    target1 : str
        Column name in df1 to sort by (descending order) for ranking.
    label2 : str
        Column name in df2 to use as the join key.
    target2 : str
        Column name in df2 to sort by (descending order) for ranking.
    output_dir : str
        Directory path where results (JSON and plot) will be saved.
    title : str
        Title to display on the correlation scatter plot.

    Returns
    -------
    dict
        Dictionary containing:
        - correlation: Pearson correlation coefficient
        - p_value: Statistical p-value
        - n_samples: Number of matched samples
        - label1, label2, target1, target2: Input parameter values
    """
    # Sort dataframes by target columns in descending order
    df1_sorted = df1.sort_values(by=target1, ascending=False).reset_index(drop=True)
    df2_sorted = df2.sort_values(by=target2, ascending=False).reset_index(drop=True)
    
    # Create ranking columns (rank 0 is highest value)
    df1_sorted['rank'] = range(len(df1_sorted))
    df2_sorted['rank'] = range(len(df2_sorted))
    
    # Merge on the label columns to align the data
    merged = pd.merge(
        df1_sorted[[label1, 'rank']], 
        df2_sorted[[label2, 'rank']], 
        left_on=label1, 
        right_on=label2,
        suffixes=('_df1', '_df2')
    )
    
    # Calculate Pearson correlation between the rankings
    pearson = stats.pearsonr(merged['rank_df1'], merged['rank_df2'])
    correlation = pearson.statistic
    p_value = pearson.pvalue

    # Prepare results
    results = {
        'correlation': float(correlation),
        'p_value': float(p_value),
        'n_samples': len(merged),
        'label1': label1,
        'label2': label2,
        'target1': target1,
        'target2': target2
    }
    
    # Save to JSON file
    with open(output_dir + 'correlation.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Create scatter plot
    plt.figure(figsize=(10, 8))
    plt.scatter(merged['rank_df1'], merged['rank_df2'], alpha=0.6)
    plt.xlabel(f'{label1} Rank (by {target1})')
    plt.ylabel(f'{label2} Rank (by {target2})')
    plt.title(title + f'\nPearson r = {correlation:.3f}, p = {p_value:.3e}')
    
    # Add diagonal line for perfect correlation
    max_rank = max(merged['rank_df1'].max(), merged['rank_df2'].max())
    plt.plot([0, max_rank], [0, max_rank], 'r--', alpha=0.5, label='Perfect correlation')
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_dir + 'correlation_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return results


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments with experiment_dir, ground_truth, and data_dir.
    """
    parser = argparse.ArgumentParser(
        description='Compare ELO rankings from model elicitation against ground truth data.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python compare.py curated
    python compare.py goodhart-curated --ground-truth model-elicitation/data/llm_rl.csv
    python compare.py intro-methods-curated --data-dir model-elicitation/data
        """
    )
    parser.add_argument(
        'experiment_dir',
        type=str,
        help='Name of the experiment directory within data_dir (e.g., "curated", "goodhart-curated")'
    )
    parser.add_argument(
        '--ground-truth',
        type=str,
        default='model-elicitation/data/llm_rl_yix_curate.csv',
        help='Path to the ground truth CSV file (default: model-elicitation/data/llm_rl_yix_curate.csv)'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='model-elicitation/data',
        help='Base data directory containing experiment folders (default: model-elicitation/data)'
    )
    return parser.parse_args()


def main() -> None:
    """
    Main entry point for the compare script.

    Discovers all elo.csv files in the specified experiment directory and
    computes ranking correlations against the ground truth data.
    """
    args = parse_args()

    # Construct full path to experiment directory
    experiment_path = os.path.join(args.data_dir, args.experiment_dir)

    if not os.path.isdir(experiment_path):
        raise FileNotFoundError(f"Experiment directory not found: {experiment_path}")

    # Load ground truth data
    if not os.path.isfile(args.ground_truth):
        raise FileNotFoundError(f"Ground truth file not found: {args.ground_truth}")

    ground_truth_df = pd.read_csv(args.ground_truth)
    print(f"Loaded ground truth from: {args.ground_truth}")

    # Discover all elo.csv files
    elo_files = discover_elo_files(experiment_path)

    if not elo_files:
        raise FileNotFoundError(f"No elo.csv files found in: {experiment_path}")

    print(f"Found {len(elo_files)} elo.csv files in {experiment_path}")

    # Process each elo.csv file
    for elo_file in tqdm(elo_files, desc="Processing models"):
        elo_dir = os.path.dirname(elo_file)
        # Extract a meaningful title from the path (model name + epochs)
        rel_path = os.path.relpath(elo_dir, experiment_path)
        title = rel_path.replace(os.sep, '/')

        compare_ranking_correlation(
            df1=pd.read_csv(elo_file),
            df2=ground_truth_df,
            label1='paper_id',
            target1='elo_rating',
            label2='paperId',
            target2='b',
            title=title,
            output_dir=elo_dir + '/'
        )


if __name__ == "__main__":
    main()
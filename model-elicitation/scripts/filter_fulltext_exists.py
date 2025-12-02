"""
Script to filter llm_rl_yix_curate_fulltext.csv by removing papers 
that don't have an intro and methods section in llm_rl_yix_curate_intro_methods.json.

Saves the filtered result as llm_rl_yix_curate_fulltext_exists.csv
"""

import json
import pandas as pd
from pathlib import Path


def main():
    # Define paths
    data_dir = Path(__file__).parent.parent / "data"
    json_path = data_dir / "llm_rl_yix_curate_intro_methods.json"
    csv_path = data_dir / "llm_rl_yix_curate_fulltext.csv"
    output_path = data_dir / "llm_rl_yix_curate_fulltext_exists.csv"

    # Load the JSON file to get paper IDs with successful intro_and_methods
    with open(json_path, "r", encoding="utf-8") as f:
        intro_methods_data = json.load(f)

    # Get paper IDs where success is True (intro_and_methods exists)
    successful_paper_ids = {
        paper["paperId"] for paper in intro_methods_data if paper["success"]
    }

    # Get paper IDs where success is False (intro_and_methods does not exist)
    failed_paper_ids = {
        paper["paperId"] for paper in intro_methods_data if not paper["success"]
    }

    print(f"Total papers in JSON: {len(intro_methods_data)}")
    print(f"Papers with intro_and_methods: {len(successful_paper_ids)}")
    print(f"Papers without intro_and_methods: {len(failed_paper_ids)}")
    print(f"\nPapers to be removed:")
    for paper in intro_methods_data:
        if not paper["success"]:
            print(f"  - {paper['paperId']}: {paper.get('title', 'N/A')}")

    # Load the fulltext CSV
    df = pd.read_csv(csv_path)
    original_count = len(df)
    print(f"\nOriginal CSV row count: {original_count}")

    # Filter to keep only papers with successful intro_and_methods
    df_filtered = df[df["paperId"].isin(successful_paper_ids)]
    filtered_count = len(df_filtered)
    print(f"Filtered CSV row count: {filtered_count}")
    print(f"Papers removed: {original_count - filtered_count}")

    # Save the filtered CSV
    df_filtered.to_csv(output_path, index=False)
    print(f"\nSaved filtered CSV to: {output_path}")


if __name__ == "__main__":
    main()

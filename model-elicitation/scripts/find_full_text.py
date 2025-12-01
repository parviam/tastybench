#!/usr/bin/env python3
"""
Add an arXiv HTML link column to a Semantic Scholar CSV.

The CSV's leftmost column is assumed to be the Semantic Scholar paperId.
By default the script edits the input file in place; pass --output to
write to a different path.

Example:
  python "find full text.py" \
      --input model-elicitation/data/llm_rl_yix_curate_fulltext.csv \
      --column-name arxiv_html
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from typing import Dict, Optional

import requests
from tqdm.auto import tqdm
API_URL = "https://api.semanticscholar.org/graph/v1/paper/"
# Default delay per request. Semantic Scholar free tier allows ~100 requests / 5 minutes (~3s/request).
DEFAULT_DELAY = 3.0
DEFAULT_FIELDS = "externalIds,title"


def fetch_arxiv_link(
    paper_id: str,
    session: requests.Session,
    retries: int = 3,
    backoff: float = 1.0,
) -> tuple[Optional[str], Optional[str]]:
    """Return (arXiv HTML link, title) for a paperId, or (None, None) if not found."""
    for attempt in range(retries):
        try:
            resp = session.get(
                f"{API_URL}{paper_id}",
                params={"fields": DEFAULT_FIELDS},
                timeout=10,
            )
            if resp.status_code == 429:
                # Respect rate limit; exponential backoff.
                time.sleep(backoff)
                backoff *= 2
                continue
            if resp.status_code == 404:
                return None, None
            resp.raise_for_status()

            data: Dict = resp.json() or {}
            title = data.get("title")
            external = data.get("externalIds") or {}
            # Key varies in capitalization; check a few common variants.
            for key in ("ArXiv", "arXiv", "ARXIV", "arxiv"):
                arxiv_id = external.get(key)
                if arxiv_id:
                    return f"https://arxiv.org/html/{arxiv_id}", title
            return None, title
        except requests.RequestException as exc:  # Network issues, timeouts, 5xx, etc.
            if attempt == retries - 1:
                print(f"[warn] {paper_id}: {exc}", file=sys.stderr)
                return None, None
            time.sleep(backoff)
            backoff *= 2
    return None, None


def process_csv(
    input_path: str,
    output_path: str,
    id_column: Optional[str],
    column_name: str,
    delay: float,
):
    with open(input_path, newline="", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        if not reader.fieldnames:
            raise ValueError("Input CSV has no header row.")

        id_col = id_column or reader.fieldnames[0]
        fieldnames = list(reader.fieldnames)
        if column_name not in fieldnames:
            fieldnames.append(column_name)
        if "title" not in fieldnames:
            fieldnames.append("title")

        rows = list(reader)

    session = requests.Session()

    for idx, row in enumerate(tqdm(rows, desc="Fetching arXiv links"), 1):
        paper_id = row.get(id_col, "").strip()
        if not paper_id:
            row[column_name] = ""
            row["title"] = ""
            continue

        # Skip fetch if already present.
        if row.get(column_name) and row.get("title"):
            continue

        link, title = fetch_arxiv_link(paper_id, session=session)
        row[column_name] = link or ""
        row["title"] = title or ""

        # Rate limiting
        if idx < len(rows):
            time.sleep(delay)

    with open(output_path, "w", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Append an arXiv HTML link column to a Semantic Scholar CSV."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the input CSV (first column should be Semantic Scholar paperId).",
    )
    parser.add_argument(
        "--output",
        help="Path for the updated CSV. Defaults to overwrite the input file.",
    )
    parser.add_argument(
        "--id-column",
        help="Column name containing the Semantic Scholar paperId. Defaults to the leftmost column.",
    )
    parser.add_argument(
        "--column-name",
        default="arxiv_html",
        help="Name of the new column to store arXiv HTML links.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=DEFAULT_DELAY,
        help=f"Seconds to sleep between requests (default {DEFAULT_DELAY}).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_path = args.output or args.input
    try:
        process_csv(
            input_path=args.input,
            output_path=output_path,
            id_column=args.id_column,
            column_name=args.column_name,
            delay=max(args.delay, 0.0),
        )
    except Exception as exc:  # Catch-all to provide a clean CLI message.
        print(f"[error] {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

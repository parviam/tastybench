#!/usr/bin/env python3
"""
Extract introduction and methods text from arXiv HTML pages listed in a CSV.

Input CSV requirements:
  - Leftmost column (or --id-column) contains the Semantic Scholar paperId.
  - A column (default: arxiv_html) contains the arXiv HTML link for the paper.

Output:
  A JSON file keyed by paperId with:
    {
      "paperId": "...",
      "intro_and_methods": "<extracted text>",
      "success": true/false,
      "error": "<message if any>",
      "source_url": "<arxiv_html>"
    }

The script is intentionally verbose (print + tqdm) and is resilient to failures;
errors for individual papers are recorded instead of aborting the whole run.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from typing import Dict, Iterable, List, Optional, Tuple

import requests
from tqdm.auto import tqdm

try:
    from bs4 import BeautifulSoup, Tag, NavigableString
except ImportError as exc:  # Graceful message if dependency is missing.
    print(
        "[error] BeautifulSoup not found. Please install with `pip install bs4`.",
        file=sys.stderr,
    )
    raise

# Heuristic heading labels
INTRO_TERMS = (
    "introduction",
    "background",
    "overview",
    "preliminaries",
)
METHOD_TERMS = (
    "methods",
    "method",
    "methodology",
    "materials and methods",
    "materials & methods",
    "approach",
    "approaches",
    "experimental setup",
    "experiments",
    "model",
    "models",
    "data and methods",
    "data & methods",
)
HEADING_TAGS = ("h1", "h2", "h3", "h4")


def normalize_text(text: str) -> str:
    """Collapse whitespace for cleaner output."""
    return " ".join(text.split())


def matches_any(label: str, terms: Iterable[str]) -> bool:
    label_l = label.lower()
    return any(term in label_l for term in terms)


def collect_section_text(start_heading: Tag) -> str:
    """
    Grab text following a heading until the next heading of the same/greater level.
    This is robust to arXiv's HTML structure where content sits as siblings.
    """
    pieces: List[str] = []
    for sib in start_heading.next_siblings:
        if isinstance(sib, Tag) and sib.name in HEADING_TAGS:
            break  # Stop at the next section heading.
        if isinstance(sib, (NavigableString, Tag)):
            text = normalize_text(get_text_from_node(sib))
            if text:
                pieces.append(text)
    return "\n".join(pieces).strip()


def get_text_from_node(node) -> str:
    """Extract visible text from a BeautifulSoup node."""
    if isinstance(node, NavigableString):
        return str(node)
    if isinstance(node, Tag):
        # Skip scripts/styles to avoid noise.
        if node.name in ("script", "style"):
            return ""
        return node.get_text(separator=" ", strip=True)
    return ""


def extract_intro_methods(html: str) -> Tuple[str, bool, Optional[str]]:
    """
    Return (text, success, error). Text combines introduction + methods sections.
    """
    soup = BeautifulSoup(html, "html.parser")

    headings: List[Tag] = soup.find_all(HEADING_TAGS)
    intro_chunks: List[str] = []
    method_chunks: List[str] = []

    for heading in headings:
        title = heading.get_text(" ", strip=True)
        if not title:
            continue

        target: Optional[List[str]] = None
        if matches_any(title, INTRO_TERMS):
            target = intro_chunks
        elif matches_any(title, METHOD_TERMS):
            target = method_chunks

        if target is None:
            continue

        section_text = collect_section_text(heading)
        if section_text:
            target.append(section_text)

    combined = "\n\n".join(intro_chunks + method_chunks).strip()
    if combined:
        return combined, True, None

    return "", False, "No matching Introduction/Methods sections found."


def fetch_html(url: str, timeout: float = 15.0) -> Tuple[Optional[str], Optional[str]]:
    """
    Fetch HTML content. Returns (html, error_message).
    """
    try:
        resp = requests.get(url, timeout=timeout)
        if resp.status_code != 200:
            return None, f"HTTP {resp.status_code}"
        content_type = resp.headers.get("content-type", "")
        if "html" not in content_type:
            return None, f"Unexpected content-type: {content_type}"
        return resp.text, None
    except requests.RequestException as exc:
        return None, str(exc)


def process_csv(
    input_path: str,
    output_path: str,
    id_column: Optional[str],
    arxiv_column: str,
    verbose: bool = False,
):
    with open(input_path, newline="", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        if not reader.fieldnames:
            raise ValueError("Input CSV has no header row.")

        id_col = id_column or reader.fieldnames[0]
        rows = list(reader)

    results: List[Dict] = []

    for row in tqdm(rows, desc="Extracting intro/methods"):
        paper_id = row.get(id_col, "").strip()
        url = row.get(arxiv_column, "").strip()

        record = {
            "paperId": paper_id,
            "intro_and_methods": "",
            "success": False,
            "error": None,
            "source_url": url,
        }

        if not paper_id:
            record["error"] = f"Missing paperId column '{id_col}'."
            results.append(record)
            if verbose:
                print(f"[warn] row missing paperId in column '{id_col}'.")
            continue

        if not url:
            record["error"] = f"No arXiv HTML link in column '{arxiv_column}'."
            results.append(record)
            if verbose:
                print(f"[warn] {paper_id}: missing arxiv_html link.")
            continue

        if verbose:
            print(f"[info] Fetching {paper_id} -> {url}")

        html, fetch_err = fetch_html(url)
        if fetch_err:
            record["error"] = f"Fetch failed: {fetch_err}"
            results.append(record)
            if verbose:
                print(f"[warn] {paper_id}: {fetch_err}")
            continue

        text, success, parse_err = extract_intro_methods(html)
        record["intro_and_methods"] = text
        record["success"] = success
        record["error"] = parse_err

        if verbose:
            status = "ok" if success else f"fail ({parse_err})"
            print(f"[info] {paper_id}: extraction {status}")

        results.append(record)

    with open(output_path, "w", encoding="utf-8") as outfile:
        json.dump(results, outfile, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create JSON with introduction and methods text from arXiv HTML links."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input CSV with paperId and arxiv_html columns.",
    )
    parser.add_argument(
        "--output",
        help="Path to output JSON (default: <input>.intro_methods.json).",
    )
    parser.add_argument(
        "--id-column",
        help="Column containing Semantic Scholar paperId (default: first column).",
    )
    parser.add_argument(
        "--arxiv-column",
        default="arxiv_html",
        help="Column name that stores arXiv HTML links (default: arxiv_html).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose progress alongside tqdm.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_path = args.output or f"{args.input}.intro_methods.json"
    try:
        process_csv(
            input_path=args.input,
            output_path=output_path,
            id_column=args.id_column,
            arxiv_column=args.arxiv_column,
            verbose=args.verbose,
        )
        if args.verbose:
            print(f"[done] wrote results to {output_path}")
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

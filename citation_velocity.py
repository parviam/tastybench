import sys
import time
import json
import datetime as dt
from pathlib import Path

import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import random
import argparse

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
BASE_URL = "https://api.semanticscholar.org/graph/v1"
SEARCH_ENDPOINT = f"{BASE_URL}/paper/search"
FIELDS = [
    "paperId",
    "title",
    "year",
    "publicationDate",
    "citationCount",
    # citationHistory is not always available; we fall back to citation data from
    # citing papers (or zero series if no data is available)
]
# ----------------------------------------------------------------------
# Command‑line arguments
# ----------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Citation Velocity Analyzer")
parser.add_argument(
    "--query",
    default="machine learning",
    help="Search query for Semantic Scholar",
)
parser.add_argument(
    "--start-year",
    type=int,
    default=2024,
    help="Start year for paper publication date filter",
)
parser.add_argument(
    "--end-year",
    type=int,
    default=2024,
    help="End year for paper publication date filter",
)
parser.add_argument(
    "--start-month",
    type=int,
    default=1,
    help="Start month (1‑12) for publication date filter",
)
parser.add_argument(
    "--end-month",
    type=int,
    default=3,
    help="End month (1‑12) for publication date filter",
)

parser.add_argument(
    "--max-paper-size",
    type=int,
    default=100,
    help="Maximum number of papers to retrieve (default: 100)",
)

parser.add_argument(
    "--top-n-plots",
    type=int,
    default=10,
    help="Number of top papers to plot (default: 10)",
)
parser.add_argument(
    "--min-citations",
    type=int,
    default=10,
    help="The minimum number of citations a paper needs to be considered.",
)
args = parser.parse_args()

# Compute date range based on arguments
START_DATE = dt.date(args.start_year, args.start_month, 1)
# Use the last day of the end month (approximate as 28‑31)
END_DATE = dt.date(args.end_year, args.end_month, 28)
# ----------------------------------------------------------------------
# TOP_N_PLOTS is now controlled via the --top-n-plots command‑line argument
REQUESTS_PER_MIN = 20  # reduced to respect rate limits and avoid 429 errors

# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------


def fetch_papers():
    """Fetch up to `args.max_paper_size` machine‑learning papers from the
    Semantic Scholar API, respecting the provided date range (`START_DATE` to
    `END_DATE`). Includes simple progress logging so the user can see fetching
    status and respects rate‑limit handling with exponential back‑off."""
    papers = []
    offset = 0
    limit = 100  # API max per request
    MAX_RETRIES = 5
    total_fetched = 0
    while len(papers) < args.max_paper_size:
        params = {
            "query": args.query,
            "year": args.start_year,
            "offset": offset,
            "limit": limit,
            "fields": ",".join(FIELDS),
            "fieldsOfStudy": "Computer Science",
            "minCitationCount": args.min_citations,
            "publicationTypes": 'Conference,JournalArticle,Study'  
        }
        retries = 0
        while True:
            resp = requests.get(SEARCH_ENDPOINT, params=params, timeout=30)
            if resp.status_code == 200:
                break
            if resp.status_code == 429 and retries < MAX_RETRIES:
                # exponential back‑off
                wait = (2 ** retries) + (random.random())
                time.sleep(wait)
                retries += 1
                continue
            raise RuntimeError(
                f"Semantic Scholar request failed ({resp.status_code}): {resp.text}"
            )
        data = resp.json()
        batch_count = 0
        for entry in data.get("data", []):
            # Filter by publication date range
            pub_date_str = entry.get("publicationDate")
            if not pub_date_str:
                continue
            try:
                pub_date = dt.datetime.strptime(pub_date_str, "%Y-%m-%d").date()
            except ValueError:
                continue
            if not (START_DATE <= pub_date <= END_DATE):
                continue
            papers.append(
                {
                    "paperId": entry.get("paperId"),
                    "title": entry.get("title", "").replace("\n", " ").strip(),
                    "pub_date": pub_date,
                    "total_citations": entry.get("citationCount", 0),
                    "citation_history": entry.get("citationHistory", []),
                }
            )
            batch_count += 1
            total_fetched += 1
            if len(papers) >= args.max_paper_size:
                break
        print(f"Fetched {batch_count} papers in this batch; total fetched so far: {total_fetched}")
        if not data.get("data"):
            break  # no more results
        offset += limit
        # Respect rate limits
        time.sleep(60 / REQUESTS_PER_MIN)
    print(f"Finished fetching papers. Total papers collected: {len(papers)}")
    return papers[:args.max_paper_size]

def fetch_citing_papers(paper_id):
    """
    Fetch citing papers for a given paper ID using the Semantic Scholar
    citations endpoint. Returns a list of dicts with at least a
    ``publicationDate`` field (as ``datetime.date``). If the request fails
    or no citing papers are found, an empty list is returned.
    """
    citations = []
    offset = 0
    limit = 1000
    endpoint = f"{BASE_URL}/paper/{paper_id}/citations"
    MAX_RETRIES = 5
    while True:
        params = {
            "offset": offset,
            "limit": limit,
            "fields": "citingPaper.paperId,citingPaper.title,citingPaper.year,citingPaper.publicationDate",
        }
        retries = 0
        while True:
            resp = requests.get(endpoint, params=params, timeout=30)
            if resp.status_code == 200:
                break
            if resp.status_code == 429 and retries < MAX_RETRIES:
                # exponential back‑off for rate limiting
                wait = (2 ** retries) + random.random()
                print(f"Citations endpoint rate‑limited (429). Back‑off {wait:.2f}s and retry {retries+1}/{MAX_RETRIES}")
                time.sleep(wait)
                retries += 1
                continue
            # Other errors – abort fetching citations for this paper
            print(f"Failed to fetch citations for paper {paper_id}: HTTP {resp.status_code}")
            return citations
        data = resp.json()
        batch_count = 0
        for entry in data.get("data", []):
            citing = entry.get("citingPaper", {})
            pub_date_str = citing.get("publicationDate")
            if not pub_date_str:
                continue
            try:
                pub_date = dt.datetime.strptime(pub_date_str, "%Y-%m-%d").date()
            except ValueError:
                continue
            citations.append({"publicationDate": pub_date})
            batch_count += 1
        print(f"Fetched {batch_count} citing papers for {paper_id} (offset {offset})")
        # If fewer results than the limit, we are at the last page
        if len(data.get("data", [])) < limit:
            break
        offset += limit
        # Respect rate limits
        time.sleep(60 / REQUESTS_PER_MIN)
    return citations


def build_time_series(paper):
    """
    Return a pandas Series indexed by month (datetime) containing cumulative
    citation counts from publication up to today.
    If the API provides a citationHistory, use it; otherwise approximate
    linearly.
    """
    today = dt.date.today()
    months = pd.date_range(
        start=paper["pub_date"], end=today, freq="MS"
    )  # month start frequency
    if paper["citation_history"]:
        # citation_history is a list of dicts: {"date": "YYYY-MM-DD", "citationCount": int}
        hist = pd.DataFrame(paper["citation_history"])
        hist["date"] = pd.to_datetime(hist["date"])
        hist = hist.set_index("date").sort_index()
        # Reindex to month start, forward‑fill, then fill missing with last known value
        series = hist["citationCount"].reindex(months, method="ffill")
        series = series.fillna(method="ffill").fillna(0).astype(int)
    else:
        # Build citation series from citing papers' publication dates
        citing_papers = fetch_citing_papers(paper["paperId"])
        if citing_papers:
            # Count cumulative citations up to each month
            counts = []
            for month_start in months:
                month_date = month_start.date()
                count = sum(1 for c in citing_papers if c["publicationDate"] <= month_date)
                counts.append(count)
            series = pd.Series(counts, index=months)
        else:
            # Fallback to a series of zeros when no citation data is available
            series = pd.Series([0] * len(months), index=months)
            print("No citation data available – using zero series")
    # Ensure monotonicity (cumulative)
    series = series.cummax()
    return series


def exponential(t, a, b):
    """C(t) = a * exp(b * t)"""
    return a * np.exp(b * t)


def fit_exponential(series):
    """Fit exponential to a citation series. Returns a, b, R²."""
    t = np.arange(len(series))
    y = series.values
    # Guard against all‑zero series
    if np.all(y == 0):
        return 0.0, 0.0, 0.0
    # Initial guess: a = first value (or 1), b = small positive
    p0 = [max(y[0], 1.0), 0.1]
    try:
        popt, pcov = curve_fit(exponential, t, y, p0=p0, maxfev=10000)
        a, b = popt
        # Compute R²
        residuals = y - exponential(t, *popt)
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0.0
        return float(a), float(b), float(r2)
    except Exception:
        # Fallback to zeros if fitting fails
        return 0.0, 0.0, 0.0


def plot_citations(papers_series, top_n):
    """Plot cumulative citation curves for the top_n papers by total citations."""
    # Determine top papers
    sorted_papers = sorted(
        papers_series,
        key=lambda x: x["total_citations"],
        reverse=True,
    )[:top_n]

    plt.figure(figsize=(12, 8))
    for entry in sorted_papers:
        series = entry["series"]
        plt.plot(
            series.index,
            series.values,
            label=entry["title"][:60] + ("…" if len(entry["title"]) > 60 else ""),
            linewidth=1.5,
        )
    plt.xlabel("Date")
    plt.ylabel("Cumulative Citations")
    plt.title(f"Citation Velocity (Top {top_n} Papers \"{args.query}\", {START_DATE:%b %Y}–{END_DATE:%b %Y})")
    plt.legend(loc="upper left", fontsize="small", ncol=2)
    plt.tight_layout()
    plt.savefig("citation_velocity.png")
    plt.close()


def save_ranking(results):
    """Write ranking CSV ordered by exponent b (descending)."""
    df = pd.DataFrame(results)
    df = df.sort_values(by="b", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1
    # Serialize citation timestamps and counts as JSON strings for CSV storage
    cols = [
        "rank",
        "paperId",
        "title",
        "a",
        "b",
        "R2",
        "total_citations",
        "citation_dates",
        "citation_counts",
    ]
    # Convert list columns to JSON strings before writing
    df["citation_dates"] = df["citation_dates"].apply(json.dumps)
    df["citation_counts"] = df["citation_counts"].apply(json.dumps)
    df[cols].to_csv("citation_ranking.csv", index=False)


def main():
    """Entry point for the citation velocity analysis script.

    The function orchestrates the workflow:
    1. Fetch papers matching the query and date range.
    2. Build cumulative citation time‑series for each paper.
    3. Fit an exponential growth model to each series.
    4. Plot the top‑N citation trajectories.
    5. Save a ranking CSV with model parameters and full time‑series data.

    The script writes two output files in the current directory:
    - ``citation_velocity.png`` – the plot of citation trajectories.
    - ``citation_ranking.csv`` – the ranking of papers by growth rate.
    """
    print("Fetching papers from Semantic Scholar …")
    papers = fetch_papers()
    if not papers:
        print("No papers found for the given criteria.", file=sys.stderr)
        sys.exit(1)

    results = []
    total_papers = len(papers)
    for idx, paper in enumerate(papers, start=1):
        series = build_time_series(paper)
        a, b, r2 = fit_exponential(series)
        # Extract citation timestamps and cumulative counts for CSV export
        citation_dates = [d.strftime("%Y-%m-%d") for d in series.index]
        citation_counts = series.tolist()
        results.append(
            {
                "paperId": paper["paperId"],
                "title": paper["title"],
                "total_citations": paper["total_citations"],
                "a": a,
                "b": b,
                "R2": r2,
                "series": series,
                "citation_dates": citation_dates,
                "citation_counts": citation_counts,
            }
        )
        print(f"Processed paper {idx}/{total_papers}")

    print("Plotting citation trajectories …")
    plot_citations(results, top_n=args.top_n_plots)

    print("Saving ranking CSV …")
    save_ranking(results)

    print("Done. Files generated:")
    print(" - citation_velocity.png")
    print(" - citation_ranking.csv")


if __name__ == "__main__":
    main()

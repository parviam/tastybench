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
import tqdm
from typing import List

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
    "authors",
    "citations.publicationDate",
]
# ----------------------------------------------------------------------
# Command‑line arguments
# ----------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Citation Velocity Analyzer")
parser.add_argument(
    "--query",
    default="reinforcement learning large language model llm rl",
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
    default=200,
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
    default=20,
    help="The minimum number of citations a paper needs to be considered.",
)
parser.add_argument(
    "--max-months-cited",
    type=int,
    default=15,
    help="How many months to consider for citation history.",
)
args = parser.parse_args()

# Compute date range based on arguments
START_DATE = f"{args.start_year}-{args.start_month:02d}"
# Use the last day of the end month (approximate as 28‑31)
END_DATE = f"{args.end_year}-{args.end_month:02d}"
# ----------------------------------------------------------------------
REQUESTS_PER_MIN = 10
MAX_RETRIES = 10
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
    total_fetched = 0
    while len(papers) < args.max_paper_size:
        params = {
            "query": args.query,
            "publicationDateorYear": f"{START_DATE}:{END_DATE}",
            "offset": offset,
            "limit": limit,
            "fields": ",".join(FIELDS),
            "fieldsOfStudy": "Computer Science",
            "minCitationCount": args.min_citations,
            "publicationTypes": 'Conference,JournalArticle,Study'  
        }
        retries = 0
        while retries < MAX_RETRIES:
            resp = requests.get(SEARCH_ENDPOINT, params=params, timeout=30)
            if resp.status_code == 200:
                time.sleep(60 / REQUESTS_PER_MIN)
                break
            if resp.status_code == 429:
                # exponential back‑off
                wait = (2 ** retries) + (random.random())
                print(f"Citations endpoint rate‑limited (429). Back‑off {wait:.2f}s and retry {retries+1}/{MAX_RETRIES}")
                time.sleep(wait)
                retries += 1
                continue
            raise RuntimeError(
                f"Semantic Scholar request failed ({resp.status_code}): {resp.text}"
            )
        
        data = resp.json()
        batch_count = 0
        for entry in data.get("data", []):
            pub_date_str = entry.get("publicationDate")
            citations = entry.get("citations", [])
            if not pub_date_str or not citations:
                continue
            try:
                pub_date = dt.datetime.strptime(pub_date_str, "%Y-%m-%d").date()
                if pub_date.month == 1 and pub_date.day == 1:
                    continue # Skip papers with only year known
            except ValueError:
                continue
            papers.append(
                {
                    "paperId": entry.get("paperId"),
                    "title": entry.get("title", "").replace("\n", " ").strip(),
                    "pub_date": pub_date,
                    "total_citations": entry.get("citationCount", 0),
                    "lead_author_id": entry.get("authors", [{}])[0].get("authorId") if entry.get("authors") else None,
                    "last_author_id": entry.get("authors", [{}])[-1].get("authorId") if entry.get("authors") else None,
                    "citations": citations,
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
    print(f"Finished fetching papers. Total papers collected: {len(papers)}")
    return papers[:args.max_paper_size]

def fetch_citing_papers(citations):
    """
    Fetch citing papers for a given paper ID using the Semantic Scholar
    citations endpoint. Returns a list of dicts with at least a
    ``publicationDate`` field (as ``datetime.date``). If the request fails
    or no citing papers are found, an empty list is returned.
    """
    citations_dict = []
    for entry in citations:
        pub_date_str = entry.get("publicationDate")
        if not pub_date_str:
            continue
        try:
            pub_date = dt.datetime.strptime(pub_date_str, "%Y-%m-%d").date()
        except ValueError:
            continue
        citations_dict.append({"publicationDate": pub_date})
    return citations_dict

def fetch_all_author_h_index(author_id: pd.Series) -> pd.Series:
    """
    Retrieve the h‑index for a given author ID via the Semantic Scholar API.
    Returns an integer h‑index or None if unavailable.
    """
    if author_id is None:
        return None
    endpoint = f"{BASE_URL}/author/batch"
    params = {"fields": "hIndex"}
    data = {
        "ids": author_id.tolist()
    }
    retries = 0
    while True:
        resp = requests.post(endpoint, params=params, json=data)
        if resp.status_code == 200:
            resp = resp.json()
            return pd.Series([entry.get('hIndex') if entry else None for entry in resp])
        if resp.status_code == 429 and retries < MAX_RETRIES:
            wait = (2 ** retries) + random.random()
            time.sleep(wait)
            print(f"Authors endpoint rate‑limited (429). Back‑off {wait:.2f}s and retry {retries+1}/{MAX_RETRIES}")
            retries += 1
            continue
        # On other errors, give up and return None
        print(data)
        raise Exception(f"Failed to fetch h‑index for authors: HTTP {resp.status_code}")

def build_time_series(paper):
    """
    Return a pandas Series indexed by month (datetime) containing cumulative
    citation counts from publication up to today.
    If the API provides a citationHistory, use it; otherwise approximate
    linearly.
    """
    today = dt.date.today()
    months = pd.date_range(
        start=paper["pub_date"], end=paper["pub_date"] + dt.timedelta(weeks=4*args.max_months_cited), freq="MS"
    )  # month start frequency
    # Build citation series from citing papers' publication dates
    citing_papers = fetch_citing_papers(paper["citations"])
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


def fit_exponential(series):
    """Fit exponential to a citation series. Returns a, b, R²."""
    t = np.arange(len(series))
    y = series.values
    # Guard against all‑zero series
    if np.all(y == 0):
        return 0.0, 0.0, 0.0
    # Initial guess: a = first value (or 1), b = small positive
    p0 = [max(y[0], 1.0), 0.1]
    def exponential(t, a, b):
        """C(t) = a * exp(b * t)"""
        return a * np.exp(b * t)
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

def fit_linear(series):
    """Fit linear (c + m*t) to a citation series. Returns intercept (c), slope (m), R²."""
    t = np.arange(len(series))
    y = series.values.astype(float)
    # Guard against all‑zero series
    if np.all(y == 0):
        return 0.0, 0.0, 0.0
    try:
        # np.polyfit returns [slope, intercept] for degree 1
        m, c = np.polyfit(t, y, 1)
        # Predictions
        y_pred = m * t + c
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0.0
        return float(c), float(m), float(r2)
    except Exception:
        return 0.0, 0.0, 0.0


def plot_citations(papers_series, top_n, output_file):
    """Plot cumulative citation curves for the top_n papers by velocity."""
    # Determine top papers
    sorted_papers = sorted(
        papers_series,
        key=lambda x: x["b"],
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
    plt.title(f"Citation Velocity (Top {top_n} Papers \"{args.query}\", {START_DATE}:{END_DATE})")
    plt.legend(loc="upper left", fontsize="small", ncol=2)
    plt.tight_layout()
    plt.savefig(fname=output_file, dpi=500)
    plt.close()


def save_ranking(results: pd.DataFrame, output_file: str) -> None:
    """Write ranking CSV ordered by b (descending)."""
    df = results.sort_values(by="b", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1
    # Convert list columns to JSON strings before writing
    df["citation_dates"] = df["citation_dates"].apply(json.dumps)
    df["citation_counts"] = df["citation_counts"].apply(json.dumps)
    df.to_csv(output_file, index=False)
    return None


def main(output_dir: str='results/'):
    """Entry point for the citation velocity analysis script.

    The function orchestrates the workflow:
    1. Fetch papers matching the query and date range.
    2. Build cumulative citation time‑series for each paper.
    3. Fit a linear growth model to each series.
    4. Plot the top‑N citation trajectories.
    5. Save a ranking CSV with model parameters and full time‑series data.

    The script writes two output files in the current directory:
    - ``citation_velocity.png`` – the plot of citation trajectories.
    - ``citation_ranking.csv`` – the ranking of papers by growth rate.
    """
    output_dir += dt.datetime.now().strftime("%Y%m%d_%H%M%S") + '/'
    print("Starting citation velocity analysis in", output_dir)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    json.dump(vars(args), open(output_dir + "config.json", "w"), indent=2)

    print("Fetching papers from Semantic Scholar …")
    papers = fetch_papers()
    if not papers:
        print("No papers found for the given criteria.", file=sys.stderr)
        sys.exit(1)

    results = []
    for paper in tqdm.tqdm(papers[1:], desc="Processing papers"):
        series = build_time_series(paper)
        a, b, r2 = fit_linear(series)
        # Extract citation timestamps and cumulative counts for CSV export
        citation_dates = [d.strftime("%Y-%m-%d") for d in series.index]
        citation_counts = series.tolist()
        results.append(
            {
                "paperId": paper["paperId"],
                "title": paper["title"],
                "pub_date": paper["pub_date"].strftime("%Y-%m-%d"),
                "total_citations": paper["total_citations"],
                "a": a,
                "b": b,
                "R2": r2,
                "lead_author_id": paper.get("lead_author_id"),
                "last_author_id": paper.get("last_author_id"),
                "series": series,
                "citation_dates": citation_dates,
                "citation_counts": citation_counts,
            }
        )
    
    print("Running stupid proxies...")
    df = pd.DataFrame(results)
    df["lead_author_h_index"] = fetch_all_author_h_index(df["lead_author_id"])
    df["last_author_h_index"] = fetch_all_author_h_index(df["last_author_id"])

    print("Plotting citation trajectories …")
    results_dict = df.to_dict(orient="records")
    plot_citations(results_dict, top_n=args.top_n_plots, output_file=output_dir+"citation_velocity.png")

    print("Saving ranking CSV …")
    df.drop(columns=["series"], inplace=True)
    save_ranking(df, output_file=output_dir+"citation_ranking.csv")

    print("Done. Files generated:")
    print(output_dir + "citation_velocity.png")
    print(output_dir + "citation_ranking.csv")


if __name__ == "__main__":
    main()

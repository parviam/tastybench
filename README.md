# Citation Velocity Analyzer

This repository provides a Python script, **`citation_velocity.py`**, that fetches recent machine‑learning papers from the [Semantic Scholar API](https://www.semanticscholar.org/), builds cumulative citation time‑series for each paper, fits an exponential growth model, and ranks the papers by their citation velocity.

## Features

- **Automated data collection** – Retrieves up to a configurable number of papers within a user‑specified date range.
- **Citation time‑series** – Constructs monthly cumulative citation counts, using either the paper’s `citationHistory` or citing‑paper dates.
- **Exponential model fitting** – Estimates parameters `a` and `b` for `C(t) = a·e^(b·t)` and provides an R² goodness‑of‑fit metric.
- **Visualization** – Generates `citation_velocity.png` showing the top‑N citation trajectories.
- **Ranking CSV** – Outputs `citation_ranking.csv` with model parameters and the full time‑series (JSON‑encoded) for each paper.

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd tastybench

# (Optional) Create a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install required packages
pip install -r requirements.txt   # if a requirements file is provided
# Or install manually:
pip install requests pandas numpy matplotlib scipy
```

## Usage

```bash
python citation_velocity.py [options]
```

### Command‑line options

| Option            | Default               | Description                                   |
|-------------------|-----------------------|-----------------------------------------------|
| `--query`         | `"machine learning"`  | Search query for Semantic Scholar.            |
| `--start-year`    | `2024`                | Start year for paper publication filter.      |
| `--end-year`      | `2024`                | End year for paper publication filter.        |
| `--start-month`   | `1`                   | Start month (1‑12) for publication filter.    |
| `--end-month`     | `3`                   | End month (1‑12) for publication filter.      |
| `--max-paper-size`| `100`                 | Maximum number of papers to retrieve.         |
| `--top-n-plots`   | `10`                  | Number of top papers to plot.                 |
| `--min_citations` | `10`                  | Minimum citations for inclusion in set.       |
The script writes two output files in the current directory:

- `citation_velocity.png` – Plot of citation trajectories for the top‑N papers.
- `citation_ranking.csv` – Ranking of papers by exponential growth rate, including JSON‑encoded time‑series data.

The tests use `unittest` and `unittest.mock` to isolate external API calls and file I/O.
# \[ONGOING\] TastyBench: Research Taste Evaluation Benchmark

A benchmark for evaluating AI models' ability to identify high-impact research directions to better understand paths to for AI R&D automation.

**Authors:** Parv Mahajan, Yilin Huang, Yixiong Hao

**Post**: TBA

**Note:** This benchmark is under active development. Results should be interpreted very cautiously given the limitations outlined above.

## Repository Structure

```
tastybench/
├── acceleration-proxies/       # Citation velocity analysis
│   ├── citation_velocity.py   # Fetch papers & compute citation growth
│   ├── test_stupid_proxies.py # Script used to validate proxy isn't confounded
│   └── results/               # Citation rankings and configs
├── model-elicitation/         # Model taste evaluation
│   ├── elicit_elo.py         # Pairwise comparison & Elo ranking
│   ├── compare.py            # Correlation analysis between rankings
│   ├── util.py               # Inference utilities (OpenAI, Anthropic, Gemini, Ollama)
│   ├── prompts/              # Prompt templates
│   └── data/                 # Model ranking results
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Setup

### Prerequisites

- Python 3.11+
- API keys (set as environment variables):
  - `OPENAI_API_KEY` (for OpenAI models)
  - `ANTHROPIC_API_KEY` (for Claude models)
  - `GEMINI_API_KEY` (for Gemini models)

- Optional: Start up server for local models

### Installation

```bash
# Clone the repository
cd tastybench

# Set up environment
uv sync
```

## Usage

### Step 1: Generate Citation Velocity Rankings

Fetch papers and compute citation velocity scores:

```bash
cd acceleration-proxies
python citation_velocity.py \
  --query "reinforcement learning large language model llm rl" \
  --start-year 2024 \
  --end-year 2024 \
  --start-month 1 \
  --end-month 3 \
  --max-paper-size 200 \
  --min-citations 20 \
  --max-months-cited 15
```

**Parameters:**
- `--query`: Search query for Semantic Scholar
- `--start-year/--end-year`: Publication date range
- `--start-month/--end-month`: Publication month range (1-12)
- `--max-paper-size`: Maximum number of papers to retrieve
- `--min-citations`: Minimum citation count threshold
- `--max-months-cited`: Citation history window (months)

**Outputs:**
- `results/YYYYMMDD_HHMMSS/citation_ranking.csv` - Papers ranked by citation velocity
- `results/YYYYMMDD_HHMMSS/citation_velocity.png` - Citation trajectory visualization
- `results/YYYYMMDD_HHMMSS/config.json` - Run configuration

### Step 2: Test for Confounding Variables

Validate that citation velocity isn't simply measuring author prestige:

```bash
uv run test_stupid_proxies.py [path/to/citation_ranking.csv]
```

If no path is provided, defaults to `citation_ranking.csv` in the current directory. Use `--help` for more options.

This generates correlation matrices showing relationships between citation velocity and potential confounds (author h-indices, total citations, etc.).

### Step 3: Set Up and Run Experiments

Create a paper dataset and run experiments with multiple models using the `Experiment` interface:

```python
from model_elicitation.elicit_elo import PaperDataset, Experiment
import pandas as pd
import asyncio

# Load papers from citation velocity output
citation_df = pd.read_csv('acceleration-proxies/results/.../citation_ranking.csv')
paper_ids = citation_df['paperId'].tolist()

# Create a paper dataset (extracts ideas from a list of paper ids)
paper_dataset = PaperDataset(
    name="llm_rl_2024",
    paper_ids=paper_ids,
    model="gpt-oss:120b",  # Model used for idea extraction
    extract_prompt='model-elicitation/prompts/extract_idea.md'
)

# Optional: Export the dataset with extracted ideas
paper_dataset.export_to_csv('model-elicitation/data/llm_rl_2024_with_ideas.csv')

# Define models to evaluate (model_id, output_folder_name)
models = [
    ('claude-sonnet-4-5-20250929', 'claude-sonnet-4-5'),
    ('gpt-5.1', 'gpt-5-1'),
    ('gemini-2.5-pro', 'gemini-2-5-pro'),
]

# Create an experiment
experiment = Experiment(
    name='my-experiment',
    paper_dataset=paper_dataset,
    models=models,
    epochs=[20],  # Number of comparison epochs per model
    ranking_prompt='model-elicitation/prompts/judge_ideas.md',
    extract_choice_prompt='model-elicitation/prompts/extract_choice.md'
)

# Run the experiment asynchronously (evaluates all models in parallel)
async def run_experiment():
    async with asyncio.TaskGroup() as tg:
        await experiment.run(tg)

asyncio.run(run_experiment())
```

**Outputs:**
- `model-elicitation/data/my-experiment/{model_name}/{epochs}-epochs/rankings.csv` - Pairwise comparison results
- `model-elicitation/data/my-experiment/{model_name}/{epochs}-epochs/elo.csv` - Elo rankings
- `model-elicitation/data/my-experiment/{model_name}/{epochs}-epochs/ranking.log` - Detailed logs
- `model-elicitation/data/my-experiment/metadata.json` - Experiment configuration

### Step 4: Compare Rankings

Compute correlation between model rankings and citation velocity:

```python
from model_elicitation.compare import compare_ranking_correlation
import pandas as pd

# Load citation velocity rankings
citation_df = pd.read_csv('acceleration-proxies/results/.../citation_ranking.csv')

# Load model Elo rankings (from experiment output)
elo_df = pd.read_csv('model-elicitation/data/my-experiment/claude-sonnet-4-5/20-epochs/elo.csv')

results = compare_ranking_correlation(
    df1=citation_df,
    df2=elo_df,
    label1='paperId',
    target1='b',  # Citation velocity parameter
    label2='paper_id',
    target2='elo_rating',
    output_dir='results/',
    title='Claude Sonnet 4.5 vs Citation Velocity'
)

print(f"Spearman correlation: {results['spearman_correlation']:.3f}")
print(f"P-value: {results['spearman_pvalue']:.4f}")
```

## Supported Models

- **OpenAI**: GPT models via OpenAI API
- **Anthropic**: Claude models (Sonnet, Opus, etc.)
- **Google**: Gemini models
- **Ollama**: Any locally-hosted models (Llama, Mistral, etc.)
- There is also a LiteLLM interface we used on an internal cluster.

Model identifiers follow the format:
- OpenAI models must begin with `gpt`, e.g., `gpt-4o`
- Anthropic models must begin with `claude`, e.g., `claude-sonnet-4-5`
- Google models must begin with `gemini`, e.g., `gemini-2-5-pro`
- You can use Ollama's regular format, e.g., `llama3:70b`
- Our internal endpoint usually frames things like `openai/gpt-oss-120b`

## Data

The `model-elicitation/data/` directory contains the expirements we've run so far!

## Contributing

We welcome feedback! Please open an issue, fork, submit a pull request, or get in touch!

| Name |
|------|
| Parv Mahajan |
| Yilin Huang |
| Yixiong Hao |
from elicit_elo import PaperDataset, RankingDataset, get_elo_rankings
import pandas as pd
import json
from scipy import stats
import matplotlib.pyplot as plt
from tqdm import tqdm

def compare_ranking_correlation(df1, df2, label1, target1, label2, target2, output_dir, title):
    """
    Compare rankings between two dataframes and calculate Pearson correlation.
    
    Parameters:
    -----------
    df1 : pd.DataFrame
        First dataframe
    df2 : pd.DataFrame
        Second dataframe
    label1 : str
        Column name in df1 to use for correlation (must be present in both dataframes)
    target1 : str
        Column name in df1 to sort by (descending order)
    label2 : str
        Column name in df2 to use for correlation (must be present in both dataframes)
    target2 : str
        Column name in df2 to sort by (descending order)
    output_dir : str
        Path to save the JSON results
        
    Returns:
    --------
    dict : Dictionary containing correlation coefficient and p-value
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


model_dirs = [
    'model-elicitation/data/curated/gpt-oss-20b/20-epochs',
    'model-elicitation/data/curated/gpt-oss-20b/50-epochs',
    'model-elicitation/data/curated/gpt-oss-120b/20-epochs',
    'model-elicitation/data/curated/gpt-oss-120b/50-epochs',
    'model-elicitation/data/curated/llama-3-70b/20-epochs',
    'model-elicitation/data/curated/llama-3-70b/50-epochs',
    'model-elicitation/data/goodhart-curated/gpt-oss-20b/20-epochs',
    'model-elicitation/data/goodhart-curated/gpt-oss-20b/50-epochs',
    'model-elicitation/data/goodhart-curated/gpt-oss-120b/20-epochs',
    'model-elicitation/data/goodhart-curated/gpt-oss-120b/50-epochs',
    'model-elicitation/data/goodhart-curated/llama-3-70b/20-epochs',
    'model-elicitation/data/goodhart-curated/llama-3-70b/50-epochs',
    'model-elicitation/data/max-goodhart-curated/gpt-oss-20b/20-epochs',
    'model-elicitation/data/max-goodhart-curated/gpt-oss-20b/50-epochs',
    'model-elicitation/data/max-goodhart-curated/gpt-oss-120b/20-epochs',
    'model-elicitation/data/max-goodhart-curated/gpt-oss-120b/50-epochs',
    'model-elicitation/data/max-goodhart-curated/llama-3-70b/20-epochs',
    'model-elicitation/data/max-goodhart-curated/llama-3-70b/50-epochs',
]

for model_dir in tqdm(model_dirs):
    compare_ranking_correlation(
        df1=pd.read_csv(model_dir + '/elo.csv'),
        df2=pd.read_csv('model-elicitation/data/llm_rl_yix_curate.csv'),
        label1='paper_id',
        target1='elo_rating',
        label2='paperId',
        target2='b',
        title=model_dir.split('/')[-2],
        output_dir=model_dir + '/'
    )
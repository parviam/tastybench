import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt

def _pairwise_pvalues(df, method):
    """Calculate pairwise p-values for Pearson or Spearman correlation."""
    cols = df.columns
    pvals = pd.DataFrame(np.zeros((len(cols), len(cols))), index=cols, columns=cols)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            if method == 'pearson':
                _, p = stats.pearsonr(df.iloc[:, i], df.iloc[:, j])
            else:
                _, p = stats.spearmanr(df.iloc[:, i], df.iloc[:, j])
            pvals.iloc[i, j] = p
            pvals.iloc[j, i] = p
    return pvals

def plot_heatmap(corr, pvals, title, filename):
    """Plot a heatmap of the correlation matrix, bolding values with p < 0.05."""
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(corr, cmap='coolwarm', cbar=True, square=True,
                     linewidths=.5, linecolor='gray', annot=False)
    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            val = corr.iloc[i, j]
            p = pvals.iloc[i, j]
            txt = f"{val:.2f}"
            weight = 'bold' if p < 0.05 else 'normal'
            ax.text(j + 0.5, i + 0.5, txt,
                    ha='center', va='center',
                    color='black', fontsize=8,
                    fontweight=weight)
    ax.set_title(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Load the CSV file
    df = pd.read_csv('citation_ranking.csv')

    # Columns of interest
    cols = ['lead_author_h_index', 'total_citations', 'b', 'a', 'last_author_h_index']

    # Compute rankings (higher values get lower rank numbers)
    rank_df = df[cols].rank(ascending=False, method='average')

    # Pearson correlation of rankings
    pearson_corr = rank_df.corr(method='pearson')

    # Spearman correlation (equivalent to Pearson on ranks, but using pandas built‑in)
    spearman_corr = df[cols].corr(method='spearman')


    # Compute p-values
    pearson_pvals = _pairwise_pvalues(rank_df, 'pearson')
    spearman_pvals = _pairwise_pvalues(df[cols], 'spearman')

    # Print results
    print("Pearson correlation of rankings:")
    print(pearson_corr)
    print("\nPearson p-values:")
    print(pearson_pvals)

    print("\nSpearman correlation of original values:")
    print(spearman_corr)
    print("\nSpearman p-values:")
    print(spearman_pvals)

    plot_heatmap(pearson_corr, pearson_pvals, "Pearson correlation of rankings", "pearson_heatmap.png")
    plot_heatmap(spearman_corr, spearman_pvals, "Spearman correlation of original values", "spearman_heatmap.png")

    pearson_corr.to_csv('stupid_proxies/pearson_correlation.csv')
    pearson_pvals.to_csv('stupid_proxies/pearson_pvalues.csv')
    spearman_corr.to_csv('stupid_proxies/spearman_correlation.csv')
    spearman_pvals.to_csv('stupid_proxies/spearman_pvalues.csv')

if __name__ == "__main__":
    main()

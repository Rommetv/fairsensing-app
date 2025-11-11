import pandas as pd
import numpy as np

# FINAL 

def compute_combined_optimization_scores(combined_df, columns=None, top_n=10):
    """
    Compute combined ranking scores from multiple optimization results.

    Parameters:
    - combined_df : DataFrame with optimization results (columns of ranked vehicle IDs).
    - columns     : Optional list of column names to use for scoring. If None, uses default set.
    - top_n       : Number of top vehicle IDs to return in the final output.

    Returns:
    - scores_df   : Full score table with optimization memberships.
    - top_n_df    : Top N vehicle IDs in a one-column DataFrame named 'combined_opt'.
    """
    if columns is None:
        columns = ['max_spatial', 'max_A_inhab', 'max_count', 'closest_simple']

    df_sel = combined_df[columns].copy()
    n = len(df_sel)

    scores = {}
    for method in df_sel.columns:
        for rank, vid in enumerate(df_sel[method]):
            points = n - rank
            scores[vid] = scores.get(vid, 0) + points

    methods_map = {
        vid: [method for method in df_sel.columns if vid in df_sel[method].values]
        for vid in scores
    }

    scores_df = pd.DataFrame.from_dict(scores, orient='index', columns=['score'])
    scores_df['optimizations'] = scores_df.index.map(methods_map)
    scores_df = scores_df.sort_values('score', ascending=False)

    top_n_df = scores_df.head(top_n).reset_index()[['index']].rename(columns={'index': 'combined_opt'})
    return scores_df, top_n_df

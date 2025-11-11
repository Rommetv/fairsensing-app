import pandas as pd

def combine_optimized_dfs(spatial_df, temp_df, pop_df, fair_df):
    df = pd.concat([spatial_df, temp_df, pop_df, fair_df], axis=1)
    return df
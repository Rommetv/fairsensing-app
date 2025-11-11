import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import euclidean
import numpy as np
from typing import Dict, List, Tuple
import geopandas as gpd


def compute_top10(gdf: gpd.GeoDataFrame, n: int = 10) -> dict:
    """
    Return a dict of the top-n rows for various metrics,
    computing required percentages first.
    Keys are 'max_<metric>'.
    """
    df = gdf.copy()

    # compute percentages
    df['P_65+'] = df['A_65+'] / df['A_inhab'] * 100
    df['P_n_west_m'] = df['A_n_west_m'] / df['A_inhab'] * 100

    metrics = [
        'A_inhab',
        'A_65+',
        'A_0_15',
        'A_nederlan',
        'A_n_west_m',
        'P_65+',
        'P_n_west_m',
        'count'
    ]

    return {f'max_{m}': df.nlargest(n, m) for m in metrics}



def analyze_tops_with_cbs(
    tops: Dict[str, gpd.GeoDataFrame],
    cbs_gdf: gpd.GeoDataFrame,
    list_col: str = 'crs28922_list',
    code_col: str = 'crs28992'
) -> Tuple[
    Dict[str, List[str]],       # cbs_lists
    Dict[str, int],             # max_number
    Dict[str, gpd.GeoDataFrame] # gdf_filtered
]:
    """
    For each optimization in `tops`:
      1) extract unique CBS codes from its top‐n GeoDataFrame
      2) filter cbs_gdf by those codes (always returns CBS cells)
      3) sum the appropriate column:
         - for 'max_point_count' → sum the vehicle 'count'
         - otherwise → sum the matching A_… column in CBS

    Returns three dicts keyed by metric name:
      - cbs_lists    : list of unique CRS codes
      - max_number   : summed total (int)
      - gdf_filtered : GeoDataFrame of matched CBS cells
    """
    cbs_lists:    Dict[str, List[str]]           = {}
    max_number:   Dict[str, int]                 = {}
    gdf_filtered: Dict[str, gpd.GeoDataFrame]    = {}

    for metric, df in tops.items():
        raw_col = metric.replace('max_', '')              # e.g. 'A_inhab', 'P_65+', or 'count'
        sum_col = raw_col if not raw_col.startswith('P_') else raw_col.replace('P_', 'A_')

        # 1) collect all unique CBS codes
        codes = set()
        for val in df[list_col]:
            if isinstance(val, list):
                codes.update(val)
            elif isinstance(val, str):
                inner = val.strip("[]")
                parts = [p.strip(" '\"") for p in inner.replace("', '", ",").split(",") if p.strip()]
                codes.update(parts)
        cbs_lists[metric] = list(codes)

        # 2) filter CBS cells (always)
        cells = cbs_gdf[cbs_gdf[code_col].isin(codes)].copy()
        gdf_filtered[metric] = cells

        # 3) compute total
        if raw_col == 'count':
            total = int(df['count'].sum())
        else:
            # sum the A_… column in the filtered CBS cells
            total = int(cells[sum_col].sum())

        max_number[metric] = total

    return cbs_lists, max_number, gdf_filtered


def report_length_metrics(df):
    """
    Compute per-cell metrics, print averages±std, and return key columns.
    """
    d = df.copy()
    d['len'] = d['crs28922_list'].apply(len)
    d['A_inhab_per_len_crs28992']     = d['A_inhab'] / d['len']
    d['point_count_per_len_crs28992'] = d['count']   / d['len']

    ai, pc = d['A_inhab_per_len_crs28992'], d['point_count_per_len_crs28992']
    print(f"Avg A_inhab/cell: {ai.mean():.2f} (Std {ai.std():.2f})")
    print(f"Avg points/cell:   {pc.mean():.2f} (Std {pc.std():.2f})")

    return d[['uni_id','A_inhab_per_len_crs28992','point_count_per_len_crs28992']]

def create_summary_df_from_tops(tops: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Build a summary table where each column is an optimization name
    and each row is one of the selected uni_id values.

    Parameters:
        tops: dict mapping optimization names (e.g. 'max_A_inhab')
              to DataFrames that contain a 'uni_id' column

    Returns:
        summary_df: DataFrame with one column per optimization,
                    rows are the uni_id lists lined up by index.
    """
    # build a dict of lists
    data = {opt: df['uni_id'].tolist() for opt, df in tops.items()}
    # create the DataFrame
    summary_df = pd.DataFrame(data)
    return summary_df

# # Usage:
# tops = {
#     'max_A_inhab':         max_A_inhab,
#     'max_A_old':           max_A_old,
#     'max_A_young':         max_A_young,
#     'max_A_dutch':         max_A_dutch,
#     'max_A_non_western':   max_A_non_western,
#     'max_P_old':           max_P_old,
#     'max_P_non_western':   max_P_non_western,
#     'max_point_count':     max_point_count,
# }

# FINAL SUMMARY

def run_max_coverage_pipeline(
    gdf: gpd.GeoDataFrame,
    cbs_gdf: gpd.GeoDataFrame,
    n: int = 10
):
    """
    1) Compute top-n vehicles per max_* metric
    2) Analyze CBS coverage for each top-n set
    3) Build a summary table of uni_id selections

    Parameters:
      - gdf     : vehicles GeoDataFrame
      - cbs_gdf : CBS cells GeoDataFrame with 'crs28992' and A_* columns
      - n       : number of top vehicles to pick for each metric

    Returns:
      - tops           : dict of GeoDataFrames (one per 'max_*')
      - cbs_lists      : dict of lists of crs28992 codes
      - max_number     : dict of summed totals
      - gdf_filtered   : dict of filtered CBS GeoDataFrames
      - summary_df     : DataFrame, rows = rank, cols = 'max_*' with uni_id
    """
    # 1) pick top-n vehicles per metric
    tops = compute_top10(gdf, n=n)

    # 2) get CBS coverage info
    cbs_lists, max_number, gdf_filtered = analyze_tops_with_cbs(tops, cbs_gdf)

    # 3) make a summary table of uni_id lists
    summary_df = create_summary_df_from_tops(tops)

    return tops, cbs_lists, max_number, gdf_filtered, summary_df

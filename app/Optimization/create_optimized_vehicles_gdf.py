import pandas as pd
import numpy as np
import geopandas as gpd 

# FINAL 

def prepare_selected_vehicles_from_combined(
    gdf: gpd.GeoDataFrame,
    combined_df: pd.DataFrame,
    column: str
) -> gpd.GeoDataFrame:
    """
    Filters and prepares a GeoDataFrame of selected vehicles using a column from combined_df.

    Parameters:
    - gdf         : Full GeoDataFrame of vehicles.
    - combined_df : DataFrame with selection columns (e.g., combined strategies).
    - column      : Column name in combined_df to use as selection list (e.g., 'max_spatial').

    Returns:
    - GeoDataFrame with percentage columns computed.
    """
    selected_ids = combined_df[column].dropna().astype(str).tolist()

    # Age percentages
    age_cols = ["A_0_15", "A_15_25", "A_25_45", "A_45_65", "A_65+"]
    for col in age_cols:
        pct_col = f"P_{col.split('_')[1]}" if col != "A_65+" else "P_65+"
        if pct_col in gdf.columns:
            gdf = gdf.drop(columns=[pct_col])
        gdf[pct_col] = (gdf[col] / gdf["A_inhab"] * 100).round(2)

    # Migration group percentages
    mig_map = {
        "A_nederlan": "P_nederlan",
        "A_west_mig": "P_west_mig",
        "A_n_west_m": "P_n_west_m"
    }
    for a_col, p_col in mig_map.items():
        if p_col in gdf.columns:
            gdf = gdf.drop(columns=[p_col])
        gdf[p_col] = (gdf[a_col] / gdf["A_inhab"] * 100).round(2)

    # Reorder geometry to the end
    gdf = gdf[[c for c in gdf.columns if c != "geometry"] + ["geometry"]]

    return gdf[gdf["uni_id"].astype(str).isin(selected_ids)].copy()

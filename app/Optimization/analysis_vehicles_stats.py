import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import mapclassify
import plotly.express as px
from shapely.geometry import MultiPoint
import ast


def create_vehicles_gdf(points_df):
    """
    Aggregate points_df by uni_id into vehicles_1503 GeoDataFrame.
    """

    # 1. stats aggregation
    stats = points_df.groupby("uni_id").agg(
        route_id=("route_id_left", lambda x: list(set(x))),
        crs28992_list=("crs28992_list", lambda x: list(set(x))),
        min_new_timest=("new_timest", "min"),
        max_new_timest=("new_timest", "max"),
        count=("id", "count")
    ).reset_index()

    # 2. build MultiPoint geometry
    geom = points_df.groupby("uni_id")["geometry"] \
        .apply(lambda geoms: MultiPoint(geoms.tolist())) \
        .reset_index()

    # 3. merge stats + geometry
    df = stats.merge(geom, on="uni_id")

    # 4. flatten crs28922_list per row
    df["crs28922_list"] = df["crs28992_list"].apply(
        lambda rows: list(dict.fromkeys(
            item
            for row in rows
            for item in ast.literal_eval(row)
        ))
    )

    # 5. delete crs28992_list and rename correctly
    df.drop(columns=["crs28992_list"], inplace=True)
    df.rename(columns={"crs28922_list": "crs28922_list"}, inplace=True)

    # 6. count unique values in crs28922_list
    df["crs28992_uniq"] = df["crs28922_list"].apply(lambda x: len(set(x)))

    # 7. count measured values per unique crs28922
    df["avg_count_per_crs28992"] = df.apply(
        lambda x: x["count"] / x["crs28992_uniq"] if x["crs28992_uniq"] > 0 else 0,
        axis=1
    ).round(2)

    # 8. count unique measurement days from original points_df
    unique_days = points_df.copy()
    unique_days["date"] = pd.to_datetime(unique_days["new_timest"], unit="s").dt.date
    day_counts = unique_days.groupby("uni_id")["date"].nunique().reset_index(name="n_unique_days")
    df = df.merge(day_counts, on="uni_id", how="left")

    # 9. reorder columns
    df = df[[
        "uni_id", "route_id", "crs28922_list",
        "min_new_timest", "max_new_timest", "count",
        "crs28992_uniq", "avg_count_per_crs28992", "n_unique_days",
        "geometry"
    ]]

    # 10. return GeoDataFrame
    gdf_vehicles = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:28992")
    return gdf_vehicles


def enrich_vehicles_with_cbs_and_routes(vehicles_gdf, points_gdf, cbs_gdf):
    """
    Enriches vehicle-level data with aggregated CBS stats and route_type from points.

    Parameters:
    - vehicles_gdf : GeoDataFrame with 'uni_id', 'crs28922_list', and other metadata
    - points_gdf   : GeoDataFrame with 'uni_id' and 'route_type'
    - cbs_gdf      : GeoDataFrame with 'crs28992' and CBS demographic columns

    Returns:
    - vehicles_gdf enriched with CBS sums, mean WOZ, route_type, sorted by 'A_inhab'
    """

    sum_cols = ['A_inhab', 'A_0_15', 'A_15_25', 'A_25_45', 'A_45_65', 'A_65+', 'A_nederlan', 'A_west_mig', 'A_n_west_m']
    mean_col = 'G_woz_woni'

    # Explode CRS list to individual rows
    exploded = vehicles_gdf[['uni_id', 'crs28922_list']].explode('crs28922_list').rename(columns={'crs28922_list': 'crs28992'})

    # Merge CBS data
    merged = exploded.merge(
        cbs_gdf[['crs28992'] + sum_cols + [mean_col]],
        on='crs28992',
        how='left'
    )

    # Aggregate CBS stats per vehicle
    agg_map = {col: 'sum' for col in sum_cols}
    agg_map[mean_col] = 'mean'
    agg = merged.groupby('uni_id').agg(agg_map).reset_index()

    # Merge aggregated CBS stats
    vehicles_gdf = vehicles_gdf.merge(agg, on='uni_id', how='left')

    # Merge route_type from points
    vehicles_gdf = vehicles_gdf.merge(
        points_gdf[['uni_id', 'route_type_left']].drop_duplicates('uni_id'),
        on='uni_id',
        how='left'
    )

    # Round G_woz_woni to 1 decimal
    vehicles_gdf[mean_col] = vehicles_gdf[mean_col].round(1)

    # Sort and reorder columns
    ordered_cols = ['uni_id', 'route_id', 'route_type_left', 'crs28922_list', 'min_new_timest', 'max_new_timest', 'count',  "crs28992_uniq", "avg_count_per_crs28992", "n_unique_days"] + sum_cols + [mean_col, 'geometry']
    vehicles_gdf = vehicles_gdf[ordered_cols]

    vehicles_gdf.sort_values(by='A_inhab', ascending=False, inplace=True)

    return vehicles_gdf


# FINAL FUNCTION

def prepare_vehicles_with_stats(points_grouped, cbs_full):
    """
    Full pipeline:
    1. Creates vehicle-level GeoDataFrame from grouped points.
    2. Enriches with CBS statistics and route types.

    Parameters:
    - points_grouped : GeoDataFrame grouped by 'uni_id' with 'route_type' and other data.
    - cbs_full       : CBS GeoDataFrame with 'crs28992' and demographic data.

    Returns:
    - vehicles_stats : GeoDataFrame enriched with CBS data and route_type.
    """
    vehicles_gdf = create_vehicles_gdf(points_grouped)
    vehicles_stats = enrich_vehicles_with_cbs_and_routes(vehicles_gdf, points_grouped, cbs_full)
    return vehicles_stats

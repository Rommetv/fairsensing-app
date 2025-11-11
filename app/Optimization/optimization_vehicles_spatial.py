import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import euclidean
import numpy as np
import math

# STEP 1 ADD TITUS OPTIMIYATION AND FINISh

# Define a function to identify the group with the largest increase in coverage

def find_next_group(gdf, covered_poly_ids):


    """ The find_next_group function is designed to identify the next group of data that maximizes the increase in coverage based on a set of previously covered polygons.
    The polygons are IDs in the table. """ 

    max_increase = -1
    next_group = None

    # Group by 'uni_id' and iterate over groups
    for uni_id, group in gdf.groupby('uni_id'):
        # Get unique poly_ids for current uni_id
        unique_poly_ids = set(group['id'].unique())

        # Calculate non-overlapping poly_ids
        non_overlapping_poly_ids = unique_poly_ids.difference(covered_poly_ids)

        # Calculate the increase in coverage
        increase = len(non_overlapping_poly_ids)

        # Update next_group if this group has a larger increase in coverage
        if increase > max_increase:
            max_increase = increase
            next_group = uni_id

    return next_group, max_increase


def prepare_vehicle_unique_ids(points_gdf, cbs_gdf):
    """
    Joins CBS cells to points and calculates the number of unique CBS cell IDs ('crs28992')
    for each vehicle ('uni_id').

    Parameters:
    - points_gdf : GeoDataFrame of points with 'id' and 'uni_id'.
    - cbs_gdf    : GeoDataFrame of CBS cells with 'crs28992' and 'geometry'.

    Returns:
    - vehicle_unique_ids : DataFrame with 'unique_id_count' per vehicle, sorted descending.
    - points_gdf         : Updated GeoDataFrame with CBS IDs added.
    """

    # Spatial join: points to CBS cells
    points_gdf = gpd.sjoin(
        points_gdf,
        cbs_gdf[['crs28992', 'geometry']],
        how='left',
        predicate='intersects'
    ).drop(columns='index_right') \
     .rename(columns={'crs28992': 'crs28992_orig'})

    # Rename for clarity
    points_gdf.rename(columns={'id': 'id_point', 'crs28992_orig': 'id'}, inplace=True)

    # Count unique CBS cells per vehicle
    vehicle_unique_ids = points_gdf.groupby('uni_id')['id'].nunique().reset_index()
    vehicle_unique_ids.rename(columns={'id': 'unique_id_count'}, inplace=True)
    vehicle_unique_ids.set_index('uni_id', inplace=True)
    vehicle_unique_ids.sort_values(by='unique_id_count', ascending=False, inplace=True)

    return vehicle_unique_ids, points_gdf


# def select_vehicles_for_max_coverage(points_gdf, vehicle_unique_ids, coverage_threshold=3):
#     """
#     Selects vehicles (uni_ids) iteratively to maximize CBS cell coverage.

#     Parameters:
#     - points_gdf          : GeoDataFrame with 'uni_id' and 'id' (CBS cell ID).
#     - vehicle_unique_ids  : DataFrame with 'uni_id' and 'unique_id_count' (number of unique cells per vehicle).
#     - coverage_threshold  : Minimum increase in coverage to continue selection (default = 3).

#     Returns:
#     - selected_uni_ids_df : DataFrame of selected vehicles with their 'unique_id_count'.
#     """

#     covered_poly_ids = []
#     selected_uni_ids_df = pd.DataFrame(columns=['uni_id'])

#     while True:
#         # Use helper function to find the next best vehicle
#         next_group, increase = find_next_group(points_gdf, covered_poly_ids)

#         if increase <= coverage_threshold:
#             break

#         # Update covered CBS cells
#         covered_poly_ids.extend(points_gdf.loc[points_gdf['uni_id'] == next_group, 'id'].unique())

#         # Add selected vehicle
#         selected_uni_ids_df = pd.concat(
#             [selected_uni_ids_df, pd.DataFrame({'uni_id': [next_group]})],
#             ignore_index=True
#         )

#         # Merge unique counts into the selection list
#         selected_uni_ids_df = selected_uni_ids_df.merge(
#             vehicle_unique_ids, on='uni_id', how='left'
#         )

#     return selected_uni_ids_df

def select_vehicles_for_max_coverage(points_gdf, vehicle_unique_ids, coverage_threshold=3):
    """
    Selects vehicles (uni_ids) iteratively to maximize CBS cell coverage.

    Returns:
    - selected_uni_ids_df : DataFrame of selected vehicles with their 'unique_id_count'.
    """

    covered_poly_ids = []
    selected_uni_ids = []  # just collect uni_ids here

    while True:
        next_group, increase = find_next_group(points_gdf, covered_poly_ids)

        if increase <= coverage_threshold:
            break

        covered_poly_ids.extend(points_gdf.loc[points_gdf['uni_id'] == next_group, 'id'].unique())
        selected_uni_ids.append(next_group)

    # Convert list to DataFrame once, THEN merge
    selected_uni_ids_df = pd.DataFrame({'uni_id': selected_uni_ids}).merge(
        vehicle_unique_ids, on='uni_id', how='left'
    )

    return selected_uni_ids_df

# def extract_top_spatial_selection(selected_uni_ids_df, vehicles_df, top_n=10):
#     """
#     Extracts top-N spatially optimized vehicles.

#     Parameters:
#     - selected_uni_ids_df : DataFrame with 'uni_id' column
#     - vehicles_df         : GeoDataFrame with 'uni_id'
#     - top_n               : number of top vehicles to select

#     Returns:
#     - optimized_ids       : list of selected uni_ids (as 'max_spatial')
#     - filtered_vehicles   : GeoDataFrame filtered to those IDs
#     """

#     top_selected = selected_uni_ids_df.head(top_n).copy()
#     top_selected.rename(columns={'uni_id': 'max_spatial'}, inplace=True)

#     optimized_ids = top_selected['max_spatial'].to_list()
#     filtered_vehicles = vehicles_df[vehicles_df['uni_id'].isin(optimized_ids)].copy()

#     return optimized_ids, filtered_vehicles

def extract_top_spatial_selection(selected_uni_ids_df, vehicles_df, top_n=10, seed=None):
    """
    Extracts top-N spatially optimized vehicles, padding with random ones if needed.

    Parameters:
    - selected_uni_ids_df : DataFrame with 'uni_id' column
    - vehicles_df         : GeoDataFrame with 'uni_id'
    - top_n               : desired number of vehicles
    - seed                : random seed for reproducibility

    Returns:
    - optimized_ids       : list of selected uni_ids (as 'max_spatial')
    - filtered_vehicles   : GeoDataFrame filtered to those IDs
    """
    top_selected = selected_uni_ids_df.head(top_n).copy()
    top_selected.rename(columns={'uni_id': 'max_spatial'}, inplace=True)
    optimized_ids = top_selected['max_spatial'].tolist()

    if len(optimized_ids) < top_n:
        remaining = vehicles_df[~vehicles_df['uni_id'].isin(optimized_ids)]
        additional = remaining.sample(n=top_n - len(optimized_ids), random_state=seed)
        extra_ids = additional['uni_id'].tolist()
        optimized_ids.extend(extra_ids)

    filtered_vehicles = vehicles_df[vehicles_df['uni_id'].isin(optimized_ids)].copy()
    return optimized_ids, filtered_vehicles


# FINAL FUNCTION

# def spatial_optimization_pipeline(points_gdf, cbs_gdf, vehicles_df, coverage_threshold=3, top_n=10):
#     """
#     Full pipeline for spatial optimization:
#     1. Prepares unique vehicle coverage.
#     2. Selects vehicles to maximize CBS cell coverage.
#     3. Extracts top-N optimized vehicles.
#     4. Returns a one-column DataFrame ('max_spatial') listing the selected vehicle IDs.

#     Parameters:
#     - points_gdf          : GeoDataFrame of measurement points.
#     - cbs_gdf             : CBS GeoDataFrame with 'crs28992' and geometry.
#     - vehicles_df         : GeoDataFrame of vehicles with 'uni_id'.
#     - coverage_threshold  : Minimum coverage increase to continue selection.
#     - top_n               : Number of top optimized vehicles to select.

#     Returns:
#     - optimized_ids       : List of selected vehicle IDs.
#     - filtered_vehicles   : GeoDataFrame of the selected vehicles.
#     - df_max_spatial      : DataFrame with one column ('max_spatial') listing selected vehicle IDs.
#     """

#     # Step 1: Prepare vehicle coverage
#     vehicle_unique_ids, points_gdf_prepared = prepare_vehicle_unique_ids(points_gdf, cbs_gdf)

#     # Step 2: Select vehicles to maximize coverage
#     selected = select_vehicles_for_max_coverage(points_gdf_prepared, vehicle_unique_ids, coverage_threshold)

#     # Step 3: Extract top-N optimized vehicles
#     optimized_ids, filtered_vehicles = extract_top_spatial_selection(selected, vehicles_df, top_n=top_n)

#     # Step 4: Format selected IDs into a one-column DataFrame
#     df_max_spatial = pd.DataFrame({'max_spatial': optimized_ids})

#     return optimized_ids, filtered_vehicles, df_max_spatial

def spatial_optimization_pipeline(vehicles_df, selected, top_n=10):
    """
    Speedy pipeline for spatial optmization
    1. Loads selected to speed up the process
    2. Extracts top-N optimized vehicles.
    3. Returns a one-column DataFrame ('max_spatial') listing the selected vehicle IDs.

    Parameters:
    - vehicles_df         : GeoDataFrame of vehicles with 'uni_id'.
    - top_n               : Number of top optimized vehicles to select.

    Returns:
    - optimized_ids       : List of selected vehicle IDs.
    - filtered_vehicles   : GeoDataFrame of the selected vehicles.
    - df_max_spatial      : DataFrame with one column ('max_spatial') listing selected vehicle IDs.
    """

    # Step 3: Extract top-N optimized vehicles
    optimized_ids, filtered_vehicles = extract_top_spatial_selection(selected, vehicles_df, top_n=top_n)

    # Step 4: Format selected IDs into a one-column DataFrame
    df_max_spatial = pd.DataFrame({'max_spatial': optimized_ids})

    return optimized_ids, filtered_vehicles, df_max_spatial
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import euclidean
import numpy as np
import math
import ast


def calculate_percentages_from_vehicles(gdf_vehicles):
    age_cols = ["A_0_15", "A_15_25", "A_25_45", "A_45_65", "A_65+"]
    for col in age_cols:
        pct_col = f"P_{col.split('_')[1]}" if col != "A_65+" else "P_65+"
        if pct_col in gdf_vehicles.columns:
            gdf_vehicles = gdf_vehicles.drop(columns=[pct_col])
        gdf_vehicles[pct_col] = (gdf_vehicles[col] / gdf_vehicles["A_inhab"] * 100).round(2)

    mig_map = {
        "A_nederlan": "P_nederlan",
        "A_west_mig": "P_west_mig",
        "A_n_west_m": "P_n_west_m"
    }
    for a_col, p_col in mig_map.items():
        if p_col in gdf_vehicles.columns:
            gdf_vehicles = gdf_vehicles.drop(columns=[p_col])
        gdf_vehicles[p_col] = (gdf_vehicles[a_col] / gdf_vehicles["A_inhab"] * 100).round(2)

    gdf_vehicles.rename(columns={
        'P_0': 'P_0_15', 'P_15': 'P_15_25',
        'P_25': 'P_25_45', 'P_45': 'P_45_65',
        'P_65+': 'P_65+'
    }, inplace=True)

    cols = [c for c in gdf_vehicles.columns if c != 'geometry'] + ['geometry']
    return gdf_vehicles[cols]


### FUNCTION 1 - make lists 
def extract_string_lists(df, prefixes=None):
    """
    Extracts non-null values from all DataFrame columns whose names
    start with any of the given prefixes, returning a dict of lists.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        prefixes (tuple): Column-name prefixes to include.
            Defaults to ('max_', 'fairest_', 'combined_', 'random').

    Returns:
        dict: { column_name: [str(id1), str(id2), …], … }
    """
    if prefixes is None:
        prefixes = ('max_','closest', 'fairest_', 'combined_', 'random', 'all_')

    strategy_cols = [
        col for col in df.columns
        if any(col.startswith(pref) for pref in prefixes)
    ]

    return {
        col: df[col].dropna().astype(str).tolist()
        for col in strategy_cols
    }


### FUNCTION 2 - compute sums (connect to vehicles)
def compute_and_export_sums(
    gdf,
    lists_dict,
    id_col='uni_id',
    value_cols=['A_inhab', 'A_0_15', 'A_65+', 'A_nederlan','A_n_west_m','count'],
):
    """
    Filters gdf by each list in lists_dict, sums only value_cols,
    builds a summary DataFrame (rows=value_cols, cols=groups),
    exports it to CSV, and returns the DataFrame.
    """
    # compute sums
    results = {
        label: {
            col: gdf[gdf[id_col].astype(str).isin(ids)][col].sum()
            for col in value_cols
        }
        for label, ids in lists_dict.items()
    }
    # build summary DataFrame
    df_summary = pd.DataFrame(results, index=value_cols).astype(int)
   
    return df_summary


def extract_unique_crs_codes(vehicles_gdf, column='crs28922_list'):
    """
    Extracts all unique CRS codes from list‑type or stringified‑list column.
    """
    unique_cells = set()
    for val in vehicles_gdf[column]:
        if isinstance(val, list):
            unique_cells.update(val)
        else:
            inner = val.strip("[]")
            parts = inner.split("', '")
            parts = [p.strip(" '\"") for p in parts if p.strip(" '\"")]
            unique_cells.update(parts)
    return list(unique_cells)

def compute_cbs_summaries(
    vehicles_gdf,
    cbs_gdf,
    lists_dict,
    id_col='uni_id',
    vehicle_crs_col='crs28922_list',
    cbs_crs_col='crs28992',
    sum_cols=['A_inhab','A_0_15','A_65+','A_nederlan','A_n_west_m'],
    export_path=None
):
    """
    For each key in lists_dict:
      1. Filter vehicles_gdf by matching IDs.
      2. Extract unique CRS codes from vehicle_crs_col.
      3. Filter cbs_gdf by those codes.
      4. Sum the sum_cols in that filtered CBS.
    Builds a summary DataFrame (rows=sum_cols_uniq plus cells_unique, cols=list keys),
    optionally exports it, and returns it.
    """
    results = {}
    codes_counts = {}
    for label, vid_list in lists_dict.items():
        # subset vehicles
        subset = vehicles_gdf[vehicles_gdf[id_col].astype(str).isin(vid_list)]
        # extract unique CRS codes and count them
        codes = extract_unique_crs_codes(subset, column=vehicle_crs_col)
        codes_counts[label] = len(codes)
        # filter CBS and sum desired columns
        cb = cbs_gdf[cbs_gdf[cbs_crs_col].isin(codes)]
        results[label] = {col: int(cb[col].sum()) for col in sum_cols}

    # build summary DataFrame
    df_summary = pd.DataFrame(results, index=sum_cols).astype(int)
    df_summary.index = [f"{idx}_uniq" for idx in df_summary.index]

    # add row counting unique cells
    df_summary.loc['cells_unique'] = pd.Series(codes_counts, dtype=int)

    if export_path:
        df_summary.to_csv(export_path)

    return df_summary

def compute_cbs_summaries(
    vehicles_gdf,
    cbs_gdf,
    lists_dict,
    id_col='uni_id',
    vehicle_crs_col='crs28922_list',
    cbs_crs_col='crs28992',
    sum_cols=['A_inhab', 'A_0_15', 'A_65+', 'A_nederlan', 'A_n_west_m'],
    export_path=None
):
    """
    For each key in lists_dict:
      1. Filter vehicles_gdf by matching IDs.
      2. Extract unique CRS codes from vehicle_crs_col.
      3. Filter cbs_gdf by those codes.
      4. Sum the sum_cols in that filtered CBS.
    Builds a summary DataFrame (rows=sum_cols_uniq plus cells_unique, cols=list keys),
    optionally exports it, and returns it.
    """
    results = {}
    codes_counts = {}

    for label, vid_list in lists_dict.items():
        subset = vehicles_gdf[vehicles_gdf[id_col].astype(str).isin(vid_list)]

        # --- Inline extract_unique_crs_codes ---
        unique_cells = set()
        for val in subset[vehicle_crs_col]:
            if isinstance(val, list):
                unique_cells.update(val)
            else:
                inner = val.strip("[]")
                parts = inner.split("', '")
                parts = [p.strip(" '\"") for p in parts if p.strip(" '\"")]
                unique_cells.update(parts)
        codes = list(unique_cells)
        # ----------------------------------------

        codes_counts[label] = len(codes)
        cb = cbs_gdf[cbs_gdf[cbs_crs_col].isin(codes)]
        results[label] = {col: int(cb[col].sum()) for col in sum_cols}

    df_summary = pd.DataFrame(results, index=sum_cols).astype(int)
    df_summary.index = [f"{idx}_uniq" for idx in df_summary.index]
    df_summary.loc['cells_unique'] = pd.Series(codes_counts, dtype=int)

    if export_path:
        df_summary.to_csv(export_path)

    return df_summary

def compute_and_merge_summaries(
    summary_df1,
    summary_df2,
    gdf,
    lists_dict,
    id_col='uni_id',
    woz_col='G_woz_woni'
):
    """
    Merges summaries from spatial/pop/fair/etc groups,
    adds demographic percentages and G_woz_woni means.
    """

    # Compute G_woz_woni means as summary_df3
    summary_df3 = pd.DataFrame([{
        label: gdf[gdf[id_col].astype(str).isin(ids)][woz_col].mean()
        for label, ids in lists_dict.items()
    }], index=[woz_col])

    # Merge all summaries
    merged_df = pd.concat([summary_df1, summary_df2, summary_df3], axis=0)

    # Add demographic share percentages
    merged_df.loc['dutch_share_pct']       = (merged_df.loc['A_nederlan_uniq'] / merged_df.loc['A_inhab_uniq'] * 100).round(2)
    merged_df.loc['non_western_share_pct'] = (merged_df.loc['A_n_west_m_uniq'] / merged_df.loc['A_inhab_uniq'] * 100).round(2)
    merged_df.loc['young_share_pct']       = (merged_df.loc['A_0_15_uniq']     / merged_df.loc['A_inhab_uniq'] * 100).round(2)
    merged_df.loc['old_share_pct']         = (merged_df.loc['A_65+_uniq']      / merged_df.loc['A_inhab_uniq'] * 100).round(2)

    return merged_df

# FUNCTION 6 - add euclidean distances 

def add_euclidean_distances(merged_df, ams_stats):
    """
    Adds a single new row 'euclidean_distance' with Euclidean distances
    from group stats to Amsterdam stats, rounded to 2 decimals.
    """
    if isinstance(ams_stats, pd.DataFrame):
        ams = ams_stats.iloc[0]
    else:
        ams = ams_stats

    metrics = [
        ('dutch_share_pct',       'P_nederlan'),
        ('non_western_share_pct', 'P_n_west_m'),
        ('young_share_pct',       'P_0_15'),
        ('old_share_pct',         'P_65+'),
        ('G_woz_woni',            'G_woz_woni')
    ]

    target = np.array([float(ams[p]) for _, p in metrics])
    dist_vals = {}

    for col in merged_df.columns:
        vec = [float(merged_df.loc[m, col]) for m, _ in metrics]
        dist_vals[col] = round(euclidean(target, vec), 2)

    return pd.concat([merged_df, pd.DataFrame(dist_vals, index=['euclidean_distance'])])

def add_bus_tram_counts(vehicles_gdf, lists_dict, summary_df,
                   id_col='uni_id', route_type_col='route_type_left'):
    """
    Adds a row 'buses_count' to summary_df, counting how many selected vehicles
    in each group are buses (route_type == 3).

    Parameters:
        vehicles_gdf   : GeoDataFrame with vehicle data (must include id_col & route_type_col)
        lists_dict     : dict {label: list of uni_id strings}
        summary_df     : DataFrame with groups as columns
        id_col         : column name for vehicle ID (default 'uni_id')
        route_type_col : column name for route type (default 'route_type')

    Returns:
        summary_df with an extra row 'buses_count'
    """
    
    bus_counts = {}
    for label, ids in lists_dict.items():
        subset = vehicles_gdf[vehicles_gdf[id_col].astype(str).isin(ids)]
        bus_counts[label] = int((subset[route_type_col] == 3).sum())

    tram_counts = {}
    for label, ids in lists_dict.items():
        subset = vehicles_gdf[vehicles_gdf[id_col].astype(str).isin(ids)]
        tram_counts[label] = int((subset[route_type_col] == 0).sum())

    # append the counts as a new row
    summary_df.loc['buses_count'] = pd.Series(bus_counts, dtype=int)
    summary_df.loc['trams_count'] = pd.Series(tram_counts, dtype=int)
    return summary_df



def add_unique_route_counts(
    vehicles_gdf,
    lists_dict,
    summary_df,
    id_col='uni_id',
    route_col='route_id'
):
    """
    Appends a row 'routes_unique' to summary_df, counting the number of distinct
    route IDs across all vehicles in each group.

    Parameters:
        vehicles_gdf : GeoDataFrame with vehicle data (must include id_col & route_col)
        lists_dict   : dict {label: list of uni_id strings}
        summary_df   : DataFrame with groups as columns
        id_col       : column name for vehicle ID (default 'uni_id')
        route_col    : column name for route list (default 'route_id')

    Returns:
        summary_df with an extra row 'routes_unique'
    """
    route_counts = {}
    for label, ids in lists_dict.items():
        subset = vehicles_gdf[vehicles_gdf[id_col].astype(str).isin(ids)]
        uniq_routes = set()
        for val in subset[route_col]:
            if isinstance(val, list):
                uniq_routes.update(val)
            else:
                # parse stringified list
                try:
                    lst = ast.literal_eval(val)
                    if isinstance(lst, list):
                        uniq_routes.update(lst)
                        continue
                except:
                    pass
                # fallback: split on commas
                uniq_routes.update([v.strip() for v in str(val).strip("[]").split(",") if v.strip()])
        route_counts[label] = len(uniq_routes)

    summary_df.loc['routes_unique'] = pd.Series(route_counts, dtype=int)
    return summary_df


def add_city_column(merged_df, ams_stats, cbs_gdf, vehicles_gdf, city_name='Amsterdam'):
    """
    Adds a new column to merged_df with values for a city (default: Amsterdam).
    Fills key indicators from stats, CBS, and vehicle data.
    """

    ams = ams_stats.iloc[0] if isinstance(ams_stats, pd.DataFrame) else ams_stats
    merged_df[city_name] = pd.NA

    mapping = {
        'A_inhab_uniq': 'A_inhab',
        'A_0_15_uniq': 'A_0_15',
        'A_65+_uniq': 'A_65+',
        'A_nederlan_uniq': 'A_nederlan',
        'A_n_west_m_uniq': 'A_n_west_m',
        'dutch_share_pct': 'P_nederlan',
        'non_western_share_pct': 'P_n_west_m',
        'young_share_pct': 'P_0_15',
        'old_share_pct': 'P_65+',
        'G_woz_woni': 'G_woz_woni'
    }

    for row, ams_col in mapping.items():
        if row in merged_df.index and ams_col in ams:
            merged_df.at[row, city_name] = ams[ams_col]

    if 'euclidean_distance' in merged_df.index:
        merged_df.at['euclidean_distance', city_name] = 0

    if 'cells_unique' in merged_df.index:
        merged_df.at['cells_unique', city_name] = len(cbs_gdf)

    # Fill technical values
    merged_df.at['buses_count', city_name] = int((vehicles_gdf['route_type_left'] == 3).sum())
    merged_df.at['trams_count', city_name] = int((vehicles_gdf['route_type_left'] == 0).sum())

    # Unique routes
    unique_routes = set()
    for val in vehicles_gdf['route_id']:
        if isinstance(val, list):
            unique_routes.update(val)
        elif isinstance(val, str):
            try:
                parsed = ast.literal_eval(val)
                if isinstance(parsed, list):
                    unique_routes.update(parsed)
                else:
                    unique_routes.add(parsed)
            except:
                unique_routes.update([v.strip() for v in val.strip("[]").split(",") if v.strip()])
        else:
            unique_routes.add(val)

    merged_df.at['routes_unique', city_name] = len(unique_routes)
    merged_df.at['count', city_name] = int(vehicles_gdf['count'].sum())

    # Fill unused or not-available rows with dash
    dash_rows = [
        'A_inhab', 'A_0_15', 'A_65+', 'A_nederlan', 'A_n_west_m',
        'count', 'buses_count', 'trams_count', 'routes_unique', 'avg_points_per_cell'
    ]

    for row in dash_rows:
        if row in merged_df.index:
            merged_df.at[row, city_name] = '-'

    return merged_df




# Updated function: adds both `cells_unique` and `avg_points_per_cell` to summary_df

def add_cells_unique_counts(
    summary_df,
    vehicles_gdf,
    lists_dict,
    id_col='uni_id',
    crs_col='crs28922_list',
    count_col='count'
):
    """
    Adds 'cells_unique' and 'avg_points_per_cell' rows to summary_df.

    Parameters:
        summary_df   : DataFrame with strategy columns + 'Amsterdam'
        vehicles_gdf : GeoDataFrame with vehicle and CRS data
        lists_dict   : dict {strategy_label: [uni_id, ...]}
        id_col       : column with vehicle ID
        crs_col      : column with list of CRS codes
        count_col    : column with total measurement count

    Returns:
        summary_df with new rows added.
    """
    counts = {}
    avg_per_cell = {}

    # Strategy-specific counts and ratios
    for label, vid_list in lists_dict.items():
        subset = vehicles_gdf[vehicles_gdf[id_col].astype(str).isin(vid_list)]
        crs_codes = extract_unique_crs_codes(subset, column=crs_col)
        counts[label] = len(crs_codes)
        total_count = subset[count_col].sum()
        avg_per_cell[label] = round(total_count / len(crs_codes), 2) if len(crs_codes) > 0 else 0

    # Amsterdam totals
    ams_crs = extract_unique_crs_codes(vehicles_gdf, column=crs_col)
    counts['Amsterdam'] = len(ams_crs)
    ams_total_count = vehicles_gdf[count_col].sum()
    avg_per_cell['Amsterdam'] = round(ams_total_count / len(ams_crs), 2) if len(ams_crs) > 0 else 0

    # Insert new rows
    summary_df.loc['cells_unique'] = pd.Series(counts, dtype=int)
    summary_df.loc['avg_points_per_cell'] = pd.Series(avg_per_cell)

    return summary_df





def create_combined_vehicle_df(*dfs):
    """
    Concatenates multiple DataFrames column-wise and renames 'closest_' to 'fairest_' in columns.
    """
    combined_df = pd.concat(dfs, axis=1)
    combined_df.columns = combined_df.columns.str.replace('closest_', 'fairest_')
    return combined_df


# FINAL 1  Random Function 
def select_random_vehicles(gdf, n=10, seed=None):
    df = gdf.sample(n=n, random_state=seed)[['uni_id']].reset_index(drop=True)
    df.rename(columns={'uni_id': 'random'}, inplace=True)
    return df

# FINAL 2 

def vehicle_optimization_stats_pipeline(
    gdf,
    cbs,
    ams_stats,
    max_space_vehicles,
    max_temp_vehicles,
    max_pop_vehicles,
    fair_vehicles,
    combined_vehicles,
    random_vehicles, 
    all_vehicles
):
    """
    Full pipeline for computing and comparing vehicle optimization strategies.

    Parameters:
        gdf                : GeoDataFrame of vehicles
        cbs                : GeoDataFrame of CBS grid
        ams_stats          : DataFrame of Amsterdam statistics
        max_space_vehicles : DataFrame with 'max_spatial' column
        max_pop_vehicles   : DataFrame with 'max_point_count' column
        fair_vehicles      : DataFrame with 'fairest_' columns
        combined_vehicles  : DataFrame with 'combined_opt' column
        random_vehicles    : DataFrame with 'random' column

    Returns:
        final_df_cells     : Final summary DataFrame
    """
    # Process vehicles
    gdf_p = calculate_percentages_from_vehicles(gdf)

    # Combine all strategy outputs
    combined_df = create_combined_vehicle_df(
        max_space_vehicles,
        max_temp_vehicles,
        max_pop_vehicles,
        fair_vehicles,
        combined_vehicles,
        random_vehicles,
        all_vehicles
    )

    # Extract dict {strategy: [vehicle_ids]}
    lists_dict = extract_string_lists(combined_df)

    # Compute summary tables
    summary_df_1 = compute_and_export_sums(gdf_p, lists_dict)
    summary_df_2 = compute_cbs_summaries(gdf_p, cbs, lists_dict)

    # Merge and enrich
    merged_df = compute_and_merge_summaries(summary_df_1, summary_df_2, gdf_p, lists_dict)
    euclidean_df = add_euclidean_distances(merged_df, ams_stats)
    trams_buses_df = add_bus_tram_counts(gdf_p, lists_dict, euclidean_df)
    final_df = add_unique_route_counts(gdf_p, lists_dict, trams_buses_df)
    final_df_city = add_city_column(final_df, ams_stats, cbs, gdf_p)
    final_df_cells = add_cells_unique_counts(final_df_city, gdf_p, lists_dict)

    return final_df_cells


# FINAL 3 

def plot_vehicles_by_group(gdf, lists_dict, geo_boundary, color='#9EC043', alpha=0.7, markersize=0.1):
    """
    Plots subsets of vehicles based on lists_dict.
    
    Parameters:
    - gdf: GeoDataFrame of all vehicles with 'uni_id'.
    - lists_dict: dict {group_name: [uni_id, ...]}.
    - geo_boundary: GeoDataFrame for the boundary.
    - color, alpha, markersize: plotting parameters.
    """
    for group_name, ids in lists_dict.items():
        subset = gdf[gdf['uni_id'].astype(str).isin(ids)]
        ax = subset.plot(figsize=(8, 6), color=color, alpha=alpha, markersize=markersize)
        geo_boundary.boundary.plot(ax=ax, color='black', linewidth=1)
        ax.set_title(group_name, fontsize=14)
        ax.set_axis_off()
        plt.show()
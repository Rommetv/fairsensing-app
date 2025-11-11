import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import euclidean
import numpy as np

def calculate_percentages(gdf):
    """
    Calculate age group and migration background percentages in-place.
    Keeps the same logic and structure as provided.
    """
    age_cols = ["A_0_15", "A_15_25", "A_25_45", "A_45_65", "A_65+"]
    for col in age_cols:
        pct_col = f"P_{col.split('_')[1]}" if col != "A_65+" else "P_65+"
        gdf[pct_col] = (gdf[col] / gdf["A_inhab"] * 100).round(2)

    mig_map = {
        "A_nederlan": "P_nederlan",
        "A_west_mig": "P_west_mig",
        "A_n_west_m": "P_n_west_m"
    }
    for a_col, p_col in mig_map.items():
        gdf[p_col] = (gdf[a_col] / gdf["A_inhab"] * 100).round(2)

    gdf.rename(columns={
        'P_0':    'P_0_15',
        'P_15':   'P_15_25',
        'P_25':   'P_25_45',
        'P_45':   'P_45_65',
        'P_65+':  'P_65+'
    }, inplace=True)

    cols = [c for c in gdf.columns if c != 'geometry'] + ['geometry']
    gdf = gdf[cols]
    
    return gdf

def calculate_closest_vehicle(gdf, ams_gdf):
    
    ams_values = ams_gdf[['P_nederlan', 'P_west_mig', 'P_n_west_m', 'P_0_15', 'P_15_25','P_25_45','P_45_65','P_65+', 'G_woz_woni']].values[0]
    gdf_values = gdf[['P_nederlan', 'P_west_mig', 'P_n_west_m', 'P_0_15', 'P_15_25','P_25_45','P_45_65', 'P_65+', 'G_woz_woni']].values

    distances = [euclidean(ams_values, row) for row in gdf_values]

    gdf['distance'] = distances
    gdf.sort_values('distance', ascending=True).head(10)

    # Closest vehicle
    closest_vehicle = gdf.loc[gdf['distance'].idxmin()]
    closest_vehicle_df = closest_vehicle.to_frame().T
    closest_vehicle_df.reset_index(drop=True, inplace=True)

    return gdf

def select_top_n_vehicles(gdf_closest, n=10):
    vehicles_simplest = gdf_closest.sort_values('distance', ascending=True).head(n)['uni_id'].values
    vehicles_simplest = vehicles_simplest.tolist()
    closest_vehicle_closest_df = gdf_closest.sort_values('distance', ascending=True).head(n)
    return closest_vehicle_closest_df, vehicles_simplest


def iterative_closest_vehicles(gdf_closest, ams_gdf, target_n=10):
    i = 2
    ams_values = ams_gdf[['P_nederlan', 'P_west_mig', 'P_n_west_m', 'P_0_15', 'P_15_25', 
                          'P_25_45', 'P_45_65', 'P_65+', 'G_woz_woni']].values[0]
    gdf_values = gdf_closest[['P_nederlan', 'P_west_mig', 'P_n_west_m', 'P_0_15', 'P_15_25', 
                              'P_25_45', 'P_45_65', 'P_65+', 'G_woz_woni']].values
    distances = [euclidean(ams_values, row) for row in gdf_values]
    gdf_closest['distance'] = distances

    closest_vehicle_df = gdf_closest.loc[[gdf_closest['distance'].idxmin()]].reset_index(drop=True)

    while len(closest_vehicle_df) < target_n:
        target_values = np.array([
            ams_gdf['P_nederlan'].values[0] * (len(closest_vehicle_df) + 1) - closest_vehicle_df['P_nederlan'].sum(),
            ams_gdf['P_west_mig'].values[0] * (len(closest_vehicle_df) + 1) - closest_vehicle_df['P_west_mig'].sum(),
            ams_gdf['P_n_west_m'].values[0] * (len(closest_vehicle_df) + 1) - closest_vehicle_df['P_n_west_m'].sum(),
            ams_gdf['P_0_15'].values[0] * (len(closest_vehicle_df) + 1) - closest_vehicle_df['P_0_15'].sum(),
            ams_gdf['P_15_25'].values[0] * (len(closest_vehicle_df) + 1) - closest_vehicle_df['P_15_25'].sum(),
            ams_gdf['P_25_45'].values[0] * (len(closest_vehicle_df) + 1) - closest_vehicle_df['P_25_45'].sum(),
            ams_gdf['P_45_65'].values[0] * (len(closest_vehicle_df) + 1) - closest_vehicle_df['P_45_65'].sum(),
            ams_gdf['P_65+'].values[0] * (len(closest_vehicle_df) + 1) - closest_vehicle_df['P_65+'].sum(),
            ams_gdf['G_woz_woni'].values[0] * (len(closest_vehicle_df) + 1) - closest_vehicle_df['G_woz_woni'].sum()
        ])

        gdf_values = gdf_closest[['P_nederlan', 'P_west_mig', 'P_n_west_m', 'P_0_15', 'P_15_25', 
                                  'P_25_45', 'P_45_65', 'P_65+', 'G_woz_woni']].values
        distances = [euclidean(target_values, row) for row in gdf_values]
        distance_col = f'distance_{i}'
        gdf_closest[distance_col] = distances

        existing_vehicle_ids = closest_vehicle_df['uni_id'].values
        filtered_gdf = gdf_closest[~gdf_closest['uni_id'].isin(existing_vehicle_ids)]

        if not filtered_gdf.empty:
            closest_row = filtered_gdf.loc[filtered_gdf[distance_col].idxmin()]
            closest_row_df = closest_row.to_frame().T
            closest_row_df['target_values'] = [np.round(target_values, 2).tolist()]
            closest_vehicle_df = pd.concat([closest_vehicle_df, closest_row_df], ignore_index=True)
        else:
            break

        i += 1
    
    vehicles_relative = closest_vehicle_df.sort_values('distance', ascending=True).head(target_n)['uni_id'].values
    vehicles_relative = list(vehicles_relative)  

    return closest_vehicle_df, vehicles_relative



def iterative_closest_vehicles_absolute(gdf, ams_gdf, target_n=10):
    metrics = ['P_nederlan', 'P_west_mig', 'P_n_west_m', 'P_0_15', 
               'P_15_25', 'P_25_45', 'P_45_65', 'P_65+', 'G_woz_woni']
    ams_vals = ams_gdf[metrics].iloc[0].values

    gdf['distance_1'] = [euclidean(ams_vals, row) for row in gdf[metrics].values]
    closest_vehicle_df = gdf.loc[[gdf['distance_1'].idxmin()]].copy()

    i = 2
    while len(closest_vehicle_df) < target_n:
        total_pop = closest_vehicle_df['A_inhab'].sum()
        real_pct = {
            m: (closest_vehicle_df[m] * closest_vehicle_df['A_inhab']).sum() / total_pop
            for m in metrics[:-1]
        }
        real_prop = closest_vehicle_df['G_woz_woni'].mean()

        target = np.array([
            ams_vals[j]*2 - (real_prop if m == 'G_woz_woni' else real_pct[m])
            for j, m in enumerate(metrics)
        ])

        dist_col = f'distance_{i}'
        gdf[dist_col] = [euclidean(target, row) for row in gdf[metrics].values]

        remaining = gdf[~gdf['uni_id'].isin(closest_vehicle_df['uni_id'])]
        if remaining.empty:
            break

        next_row = remaining.loc[remaining[dist_col].idxmin()].copy()
        next_row['target_values'] = [np.round(target, 2).tolist()]

        closest_vehicle_df = pd.concat(
            [closest_vehicle_df, next_row.to_frame().T],
            ignore_index=True
        )
        i += 1

    vehicles_absolute = closest_vehicle_df.sort_values('distance', ascending=True).head(target_n)['uni_id'].values
    vehicles_absolute = vehicles_absolute.tolist()

    return closest_vehicle_df, vehicles_absolute


def create_area_comparison_statistics(ams_gdf, df_closest, df_rel, df_abs):
    """
    Creates a DataFrame comparing Amsterdam average with optimized sensing stats
    using absolute, relative, and closest vehicle methods.

    Parameters:
    - ams_gdf   : GeoDataFrame with Amsterdam reference stats
    - df_abs    : GeoDataFrame of absolute optimization vehicles
    - df_rel    : GeoDataFrame of relative optimization vehicles
    - df_closest: GeoDataFrame of closest match optimization vehicles

    Returns:
    - df_area_statistics : DataFrame with percentages + Euclidean distances
    """

    def calculate_percentages2(gdf):
        sums = gdf[[
            'A_inhab', 'A_nederlan', 'A_west_mig', 'A_n_west_m',
            'A_0_15', 'A_15_25', 'A_25_45', 'A_45_65', 'A_65+', 'G_woz_woni'
        ]].sum()

        return {
            'P_nederlan': float(sums['A_nederlan'] / sums['A_inhab'] * 100),
            'P_west_mig': float(sums['A_west_mig'] / sums['A_inhab'] * 100),
            'P_n_west_m': float(sums['A_n_west_m'] / sums['A_inhab'] * 100),
            'P_0_15': float(sums['A_0_15'] / sums['A_inhab'] * 100),
            'P_15_25': float(sums['A_15_25'] / sums['A_inhab'] * 100),
            'P_25_45': float(sums['A_25_45'] / sums['A_inhab'] * 100),
            'P_45_65': float(sums['A_45_65'] / sums['A_inhab'] * 100),
            'P_65+': float(sums['A_65+'] / sums['A_inhab'] * 100),
            'G_woz_woni': float(sums['G_woz_woni'] / len(gdf))
        }

    # Compute all percentage dicts
    percentages_abs = calculate_percentages2(df_abs)
    percentages_rel = calculate_percentages2(df_rel)
    percentages_closest = calculate_percentages2(df_closest)

    # Compose the data table
    data = {
        'Area': ['Amsterdam_Average', 'percentages_abs', 'percentages_rel', 'percentages_closest'],
        'P_nederlan': [ams_gdf['P_nederlan'].iloc[0], percentages_abs['P_nederlan'], percentages_rel['P_nederlan'], percentages_closest['P_nederlan']], 
        'P_west_mig': [ams_gdf['P_west_mig'].iloc[0], percentages_abs['P_west_mig'], percentages_rel['P_west_mig'], percentages_closest['P_west_mig']],
        'P_n_west_m': [ams_gdf['P_n_west_m'].iloc[0], percentages_abs['P_n_west_m'], percentages_rel['P_n_west_m'], percentages_closest['P_n_west_m']],
        'P_0_15': [ams_gdf['P_0_15'].iloc[0], percentages_abs['P_0_15'], percentages_rel['P_0_15'], percentages_closest['P_0_15']],
        'P_15_25': [ams_gdf['P_15_25'].iloc[0], percentages_abs['P_15_25'], percentages_rel['P_15_25'], percentages_closest['P_15_25']],
        'P_25_45': [ams_gdf['P_25_45'].iloc[0], percentages_abs['P_25_45'], percentages_rel['P_25_45'], percentages_closest['P_25_45']],
        'P_45_65': [ams_gdf['P_45_65'].iloc[0], percentages_abs['P_45_65'], percentages_rel['P_45_65'], percentages_closest['P_45_65']],
        'P_65+': [ams_gdf['P_65+'].iloc[0], percentages_abs['P_65+'], percentages_rel['P_65+'], percentages_closest['P_65+']],
        'G_woz_woni': [ams_gdf['G_woz_woni'].iloc[0], percentages_abs['G_woz_woni'], percentages_rel['G_woz_woni'], percentages_closest['G_woz_woni']],
        'Date': ["CBS Average 2022", "15th of March", "15th of March", "15th of March"]
    }

    # Compute Euclidean distances
    amsterdam_avg = np.array([
        data['P_nederlan'][0], data['P_west_mig'][0], data['P_n_west_m'][0],
        data['P_0_15'][0], data['P_15_25'][0], data['P_25_45'][0],
        data['P_45_65'][0], data['P_65+'][0], data['G_woz_woni'][0]
    ])
    percentages_list = [
        [data[k][1] for k in ['P_nederlan','P_west_mig','P_n_west_m','P_0_15','P_15_25','P_25_45','P_45_65','P_65+','G_woz_woni']],
        [data[k][2] for k in ['P_nederlan','P_west_mig','P_n_west_m','P_0_15','P_15_25','P_25_45','P_45_65','P_65+','G_woz_woni']],
        [data[k][3] for k in ['P_nederlan','P_west_mig','P_n_west_m','P_0_15','P_15_25','P_25_45','P_45_65','P_65+','G_woz_woni']]
    ]
    distances = [euclidean(amsterdam_avg, p) for p in percentages_list]
    data['Distance'] = [0] + distances

    return pd.DataFrame(data).round(2)


def generate_optimization_vehicle_table(closest_absolute, closest_relative, closest_closest, n):
    """
    Create tables of vehicle IDs from different optimization strategies.

    Parameters:
    - closest_absolute : GeoDataFrame of absolute optimization vehicles
    - closest_relative : GeoDataFrame of relative optimization vehicles
    - closest_closest  : GeoDataFrame of closest-match optimization vehicles

    Returns:
    - df_optimizations : DataFrame with all vehicle IDs per optimization
    - df_vehicle_ids   : DataFrame with top 10 vehicles per optimization for comparison
    """
    df_optimizations = pd.DataFrame({
        'optimization': ['fair_absolute', 'fair_relative', 'fair_closest'],
        'vehicles': [
            closest_absolute['uni_id'].tolist(),
            closest_relative['uni_id'].tolist(),
            closest_closest['uni_id'].tolist()
        ]
    })

    df_vehicle_ids = pd.DataFrame({
        'closest_absolute': closest_absolute['uni_id'].tolist()[:n],
        'closest_relative': closest_relative['uni_id'].tolist()[:n],
        'closest_simple':  closest_closest['uni_id'].tolist()[:n]
    })

    return df_optimizations, df_vehicle_ids

# FINAL FUNCTION

# PIPELINE

def run_fairness_pipeline(gdf, ams_gdf, n=10):
    """
    Executes the full fairness workflow in the correct order, using the same top-N for all three methods.

    Parameters:
    - gdf     : GeoDataFrame with vehicle % columns
    - ams_gdf : GeoDataFrame with Amsterdam stats
    - n       : number of top vehicles to select for simple, relative, and absolute optimization

    Returns:
    - df_area_statistics : DataFrame comparing area-level stats
    - df_optimizations   : Full list of IDs per optimization type
    - df_vehicle_ids     : Top-N vehicle IDs side-by-side
    """
    # 1) compute percentages
    gdf_p = calculate_percentages(gdf)

    # 2) ensure AMS stats floats
    ams_gdf = ams_gdf.astype({
        'P_nederlan': float, 'P_west_mig': float, 'P_n_west_m': float,
        'P_0_15': float, 'P_15_25': float, 'P_25_45': float,
        'P_45_65': float, 'P_65+': float, 'G_woz_woni': float
    })

    # 3) compute distances
    gdf_closest = calculate_closest_vehicle(gdf_p, ams_gdf)

    # 4) simple closest
    closest_simple, _ = select_top_n_vehicles(gdf_closest, n=n)

    # 5) relative iterative
    closest_relative, _ = iterative_closest_vehicles(gdf_closest, ams_gdf, target_n=n)

    # 6) absolute iterative
    closest_absolute, _ = iterative_closest_vehicles_absolute(gdf_p, ams_gdf, target_n=n)

    # 7) area-level comparison
    df_area_statistics = create_area_comparison_statistics(
        ams_gdf, closest_simple, closest_relative, closest_absolute
    )

    # 8) compile vehicle ID tables
    df_optimizations, df_vehicle_ids = generate_optimization_vehicle_table(
        closest_absolute, closest_relative, closest_simple, n
    )

    return closest_simple, closest_relative, closest_absolute, df_area_statistics, df_optimizations, df_vehicle_ids

# new core
from __future__ import annotations

"""Core execution logic for the sensor‑placement optimisation service.

This module turns the original one‑off script into a **pure function** so that
FastAPI (or any other caller) can execute it with different parameters without
side‑effects.  It writes all artefacts for a single run into a dedicated
sub‑directory and returns a JSON‑serialisable dictionary that the HTTP layer
can embed straight in its `/job/{id}` response.
"""
import matplotlib
matplotlib.use("Agg")

from pathlib import Path
from typing import Literal, TypedDict, Any
import uuid
import warnings

import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import plotly.subplots as sp
import folium
from shapely.geometry import mapping


import ast
import re

import json
import numpy as np
from shapely.geometry import LineString

from typing import Callable, Optional

# Third‑party pipeline helpers — these live in your fair_sensing package
from app.Optimization import (
    prepare_vehicles_with_stats,
    spatial_optimization_pipeline,
    temporal_optimization_pipeline,
    run_fairness_pipeline,
    run_max_coverage_pipeline,
    combine_optimized_dfs,
    compute_combined_optimization_scores,
    select_random_vehicles,
    vehicle_optimization_stats_pipeline,
    prepare_selected_vehicles_from_combined,
    master_function_analysis,
    visualization_master_function,
)

City = Literal["Amsterdam", "Rotterdam", "The Hague", "Utrecht"]  # extend when other datasets are ready
timeframeT = Literal["1day", "3day", "7day"]
optMethod = Literal["max_spatial", "max_temporal", "closest_relative"]
# New helpers
def get_joined_cbs_gdf(vehicles_gdf, cbs_gdf, vehicle_crs_col='crs28922_list', cbs_crs_col='crs28992'):
    """
    Loads data and returns joined CBS GeoDataFrame based on CRS cell matches.

    Parameters:
    - vehicles_csv_path : path to vehicles summary CSV
    - cbs_shp_path      : path to CBS shapefile
    - vehicle_crs_col   : column name in vehicles_gdf holding CRS-list
    - cbs_crs_col       : column name in cbs_gdf holding the CRS code

    Returns:
    - GeoDataFrame of filtered CBS cells
    """
    def extract_unique_crs_codes(df, column):
        unique_cells = set()
        for val in df[column]:
            if isinstance(val, list):
                unique_cells.update(val)
            else:
                inner = val.strip("[]")
                parts = inner.split("', '")
                parts = [p.strip(" '\"") for p in parts if p.strip(" '\"")]
                unique_cells.update(parts)
        return list(unique_cells)

  
    # Extract codes and filter
    codes = extract_unique_crs_codes(vehicles_gdf, column=vehicle_crs_col)
    return cbs_gdf[cbs_gdf[cbs_crs_col].isin(codes)].copy()

def calculate_and_compare_sums(cbs_gdf, sensed_gdf, city):
    """
    Calculates and compares the sums of columns for two GeoDataFrames, and computes the percentage
    of the sensed values relative to the city's total values.

    Parameters:
        cbs_gdf (gpd.GeoDataFrame): GeoDataFrame containing the full city's data.
        sensed_gdf (gpd.GeoDataFrame): GeoDataFrame containing the sensed data.

    Returns:
        pd.DataFrame: A DataFrame containing the sums for both datasets and the percentage of sensed values.
    """
    # Calculate the sum of all columns except 'G_woz_woni', 'geometry', 'index_right', and 'Lijn_Numbe' for the city-wide data
    if city == 'Amsterdam':
        cbs_sums = cbs_gdf.drop(columns=['crs28992', 'G_woz_woni', 'geometry', 'age_sum', 'migration_' ]).sum()
    else:
       cbs_sums = cbs_gdf.drop(columns=['crs28992', 'G_woz_woni', 'geometry', 'age_sum', 'migration_sum' ]).sum() 
    
    # Extract values from the Series
    values_t = cbs_sums.values

    # Calculate the sum of all columns except 'G_woz_woni', 'geometry', 'index_right', and 'Lijn_Numbe' for the sensed data
    if city == 'Amsterdam':
        sensed_sums = sensed_gdf.drop(columns=['crs28992', 'G_woz_woni', 'geometry', 'age_sum', 'migration_']).sum()
    else:
        sensed_sums = sensed_gdf.drop(columns=['crs28992', 'G_woz_woni', 'geometry', 'age_sum', 'migration_sum']).sum() 			
    
    # Extract values from the Series
    values_s = sensed_sums.values

    # Extract keys (index) from the Series
    keys = sensed_sums.index

    # Create a new DataFrame
    data = {
        'Sociodemo': keys,
        'Sums_sensed': values_s,
        'Sums_total': values_t
    }

    sums = pd.DataFrame(data)
    
    # Calculate the percentage of sensed values relative to the city's total values
    sums['Sensed_%'] = ((sums['Sums_sensed'] / sums['Sums_total']) * 100).round(2)
    # Calculate exclusion
    sums['Excluded!'] = (sums['Sums_total'] - sums['Sums_sensed']).round(0)


    # Convert sums to integer values for cleaner display
    sums['Sums_sensed'] = sums['Sums_sensed'].astype(int)
    sums['Sums_total'] = sums['Sums_total'].astype(int)
    sums['Sensed_%'] = sums['Sensed_%'].astype(float)
    sums['Excluded!'] = sums['Excluded!'].astype(int)
    
    return sums

def normalize_statistics(merged_df):
    """
    Normalizes the columns in the merged_statistics DataFrame based on 'A_inhab'.
    Drops 'A_woning' and 'A_inhab', and rounds the results to two decimal places. Get percentages!
    
    Parameters:
        merged_statistics (pd.DataFrame): The DataFrame containing the statistics to normalize.

    Returns:
        pd.DataFrame: A DataFrame with normalized statistics.
    """
    # Create a copy of the DataFrame to avoid modifying the original
    average_stats = merged_df.copy()


    # Identify columns to normalize
    columns_to_normalize = average_stats.columns.difference(['Area', 'G_woz_woni', 'A_inhab'])

    # Normalize the identified columns
    average_stats[columns_to_normalize] = average_stats[columns_to_normalize].div(average_stats['A_inhab'], axis=0)

    # Drop the 'A_inhab' column
    average_stats = average_stats.drop(columns=['A_inhab'])

    # Round to two decimal places
    average_stats = average_stats.round(4)

    return average_stats

def generate_summary_statistics(cbs_gdf, area_name=City):
    """
    Calculates summary statistics from the CBS GeoDataFrame, excluding specified columns,
    and formats the results into a summary DataFrame.

    Parameters:
        cbs_gdf (gpd.GeoDataFrame): GeoDataFrame containing CBS data.

    Returns:
        pd.DataFrame: A DataFrame containing the summary statistics.
    """
    # Calculate the sum for numeric columns excluding 'G_woz_woni' and 'geometry'
    sum_values = cbs_gdf.drop(columns=['geometry', 'G_woz_woni']).sum()
    
    # Calculate the average for 'G_woz_woni'
    average_woz_woni = cbs_gdf['G_woz_woni'].mean()
    
    # Round the sums and average to zero decimal places
    rounded_sum_values = sum_values.round(0)
    rounded_average_woz_woni = round(average_woz_woni, 0)
    
    # Create the summary row DataFrame
    summary_row_ams = pd.DataFrame({
        'Area': [area_name],
        'A_inhab': [rounded_sum_values.get('A_inhab', 0)],
        'A_0_15': [rounded_sum_values.get('A_0_15', 0)],
        'A_15_25': [rounded_sum_values.get('A_15_25', 0)],
        'A_25_45': [rounded_sum_values.get('A_25_45', 0)],
        'A_45_65': [rounded_sum_values.get('A_45_65', 0)],
        'A_65+': [rounded_sum_values.get('A_65+', 0)],
        'G_woz_woni': [rounded_average_woz_woni],
        'A_nederlan': [rounded_sum_values.get('A_nederlan', 0)],
        'A_west_mig': [rounded_sum_values.get('A_west_mig', 0)],
        'A_n_west_m': [rounded_sum_values.get('A_n_west_m', 0)],
    })

    summary_row_ams['A_inhab'] = summary_row_ams['A_inhab'].astype(int)
    summary_row_ams['G_woz_woni'] = summary_row_ams['G_woz_woni'].astype(int)

    return summary_row_ams

def gdf_to_featurecollection(gdf: gpd.GeoDataFrame, epsg: int = 4326) -> dict:
    if gdf.crs is None or gdf.crs.to_epsg() != epsg:
        gdf = gdf.to_crs(epsg)
    return json.loads(gdf.to_json())

def _cumulative_timestamps_for_linestring(ls: LineString, meters_per_sec: float) -> list[float]:
    coords = list(ls.coords)
    if len(coords) <= 1:
        return [0.0]
    dists = [0.0]
    for i in range(1, len(coords)):
        (x0, y0), (x1, y1) = coords[i-1], coords[i]
        dists.append(dists[-1] + float(np.hypot(x1 - x0, y1 - y0)))
    return [d / meters_per_sec for d in dists]

def lines_to_trips_featurecollection(
    lines_gdf: gpd.GeoDataFrame,
    id_col: str = "route_id",
    seconds_per_km: float = 120.0,
    epsg_metric: int = 28992,
    epsg_geo: int = 4326,
) -> dict:
    if lines_gdf.crs is None or lines_gdf.crs.to_epsg() != epsg_metric:
        work = lines_gdf.to_crs(epsg_metric)
    else:
        work = lines_gdf.copy()

    mps = 1000.0 / max(1e-6, seconds_per_km)  # m/s
    features = []
    for _, row in work.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        parts = list(geom.geoms) if geom.geom_type == "MultiLineString" else [geom]
        for part in parts:
            ts = _cumulative_timestamps_for_linestring(part, mps)
            part_geo = gpd.GeoSeries([part], crs=epsg_metric).to_crs(epsg_geo).iloc[0]
            coords = list(part_geo.coords)
            features.append({
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": coords},
                "properties": {
                    id_col: str(row.get(id_col, "")),
                    "timestamps": ts
                }
            })
    return {"type": "FeatureCollection", "features": features}

# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------




class RunResult(TypedDict):
    job_id: str
    artefacts: dict[str, dict[str, str]]
    top_vehicles: dict[str, list[str]]
    stats_table: dict[str, Any]
    geojson: dict[str, Any]            # <- was dict[str, str]
    graph_data: dict[str, dict[str, Any]]


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def lines_fc_with_props(
    lines_gdf: gpd.GeoDataFrame,
    props=("route_id","route_type","shape_id"),
    epsg_metric=28992, epsg_geo=4326,
    max_vertices=10_000_000
) -> dict:
    if lines_gdf is None or lines_gdf.empty:
        return {"type": "FeatureCollection", "features": []}

    if lines_gdf.crs is None or lines_gdf.crs.to_epsg() != epsg_metric:
        work = lines_gdf.to_crs(epsg_metric)
    else:
        work = lines_gdf.copy()
    wgs = work.to_crs(epsg_geo)

    feats, vtx = [], 0
    for _, row in wgs.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        parts = list(geom.geoms) if geom.geom_type == "MultiLineString" else [geom]
        base_props = {p: (None if p not in row else (int(row[p]) if pd.notna(row[p]) and str(row[p]).isdigit() else row[p]))
                      for p in props}
        for part in parts:
            coords = list(part.coords)
            vtx += len(coords)
            if vtx > max_vertices: break
            feats.append({
                "type": "Feature",
                "geometry": {"type": "LineString",
                             "coordinates": [(float(x), float(y)) for x, y in coords]},
                "properties": base_props,
            })
        if vtx > max_vertices: break
    return {"type": "FeatureCollection", "features": feats}



def _build_paths(city: City, timeframe: str, data_root: Path) -> dict[str, Path]:
    """Return all dataset paths for the chosen city/timeframe."""
    suffix = {"1day": "_1day", "3day": "_3days", "7day": "_7days"}.get(timeframe, "")

    if city == "Amsterdam":
        return {
            "city_stats": data_root / "city_stats_amsterdam.csv",
            # "points_gdf": data_root / "temp" / f"grouped_by_points_GVB{suffix}.gpkg",
            "vehicle_stats": data_root / "temp" / f"vehicles_stats_GVB{suffix}.gpkg",
            "city_geo": data_root / "Gemeente2.geojson",
            "cbs_full": data_root / "temp" / "full_cbs.gpkg",
            "transport_lines": data_root / "temp" / f"public_transport_GVB_28992.gpkg",

        }
    elif city == "Rotterdam":
        return {
            "city_stats": data_root / "rotterdam2_citystats.csv",
            #"points_gdf": data_root / "temp" / f"grouped_by_points_RET{suffix}.gpkg",
            "vehicle_stats": data_root / "temp" / f"vehicles_stats_RET{suffix}.gpkg",
            "city_geo": data_root / "gemeente_Rotterdam2.geojson",
            "cbs_full": data_root / "temp" / "full_cbs_rotterdam2.gpkg",
            # "transport_lines": data_root / "temp" / f"transport_lines{suffix}.gpkg",
        }
    elif city == "The Hague":
        return {
            "city_stats": data_root / "thehague_citystats.csv",
            #"points_gdf": data_root / "temp" / f"grouped_by_points_HTM{suffix}.gpkg",
            "vehicle_stats": data_root / "temp" / f"vehicles_stats_HTM{suffix}.gpkg",
            "city_geo": data_root / "Gemeente_DenHaag.geojson",
            "cbs_full": data_root / "temp" / "full_cbs_thehague.gpkg",
            #"transport_lines": data_root / "temp" / f"transport_lines{suffix}.gpkg",
        }
    elif city == "Utrecht":
        return {
            "city_stats": data_root / "utrecht_citystats.csv",
            #"points_gdf": data_root / "temp" / f"grouped_by_points_UOV{suffix}.gpkg",
            "vehicle_stats": data_root / "temp" / f"vehicles_stats_UOV{suffix}.gpkg",
            "city_geo": data_root / "Gemeente_Utrecht.geojson",
            "cbs_full": data_root / "temp" / "full_cbs_utrecht.gpkg",
            #"transport_lines": data_root / "temp" / f"transport_lines{suffix}.gpkg",
        }

def _to_id_list(obj):
    """Accept list/Series/DataFrame and return list[str] of uni_id."""
    if obj is None:
        return []
    if isinstance(obj, list):
        return [str(x) for x in obj]
    if hasattr(obj, "uni_id"):           # GeoDataFrame / DataFrame
        return obj.uni_id.astype(str).tolist()
    return [str(obj)]

def _guess_and_to_wgs84(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """If CRS missing, try RD heuristic; then project to EPSG:4326."""
    gdf = gdf.copy()
    if gdf.crs is None:
        xs = pd.Series([getattr(g, 'x', None) for g in gdf.geometry], dtype='float64').dropna()
        ys = pd.Series([getattr(g, 'y', None) for g in gdf.geometry], dtype='float64').dropna()
        rd_like = (xs.between(0, 300000).mean() > 0.8) and (ys.between(300000, 650000).mean() > 0.8)
        gdf = gdf.set_crs(28992 if rd_like else 4326, allow_override=True)
        #print("[DBG] selection.crs was None → assumed", gdf.crs)
    return gdf.to_crs(4326)

def _geom_to_point(g):
    """Return a Shapely Point for any geometry (Point, MultiPoint, Line*, Polygon*)."""
    if g is None or g.is_empty:
        return None
    gt = getattr(g, "geom_type", "")
    if gt == "Point":
        return g
    if gt == "MultiPoint":
        try:
            # first point if exists; fallback to centroid
            return next(iter(g.geoms), None) or g.centroid
        except Exception:
            return g.centroid
    try:
        return g.representative_point()
    except Exception:
        return g.centroid


# ---------------------------------------------------------------------------
# Main callable
# ---------------------------------------------------------------------------
ReportFn = Optional[Callable[[int, Optional[str], Optional[str]], None]]

def run_job(
    *,
    city: City,
    # opti_method: optMethod,
    timeframe: timeframeT,
    n_sensors: int,
    data_root: Path,
    output_dir: Path,
    report: ReportFn = None
) -> RunResult:
    """Execute one optimisation run.

    Parameters
    ----------
    city, opti_method, timeframe, n_sensors
        User‑controlled knobs from the React form.
    data_root
        Folder that contains **fair_sensing/data/** — mounted read‑only in prod.
    output_dir
        Pre‑created directory dedicated to *this* job where we can write PNGs
        and JSON blobs.  Caller chooses the parent path.
    """


    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        
    def _r(p, msg=None, step=None):
        if report:
            report(p, msg, step)

    VEHICLES_ROUTE_ID_COL = "route_id"   # in vehicles_stats / selections
    LINES_ROUTE_ID_COL    = "route_id" 


    # 1.  Resolve dataset paths ------------------------------------------------
    _r(3, "Loading datasets…", "load")
    paths = _build_paths(city, timeframe, data_root)
    
    # 2.  Read datasets --------------------------------------------------------
    warnings.simplefilter(action="ignore", category=pd.errors.SettingWithCopyWarning)

    _r(15, "Preparing datasets...", "prepare")
    cbs_full = gpd.read_file(paths["cbs_full"])
    city_stats = pd.read_csv(paths["city_stats"])
    # points_gdf = gpd.read_file(paths["points_gdf"])
    city_geo = gpd.read_file(paths["city_geo"])
    vehicles_stats = gpd.read_file(paths["vehicle_stats"])
    # transport_lines = gpd.read_file(paths["transport_lines"])



    def normalize_route_ids_preserve_crs(
        gdf: gpd.GeoDataFrame, col="route_id"
    ) -> gpd.GeoDataFrame:
        crs = gdf.crs
        def to_tokens(v):
            if v is None or (isinstance(v, float) and pd.isna(v)): return []
            if isinstance(v, (list, tuple, np.ndarray, pd.Series)):
                return [str(x).strip() for x in v if str(x).strip()]
            s = str(v).strip()
            if not s: return []
            # parse stringified list/tuple safely
            if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
                try:
                    parsed = ast.literal_eval(s)
                    if isinstance(parsed, (list, tuple)):
                        return [str(x).strip() for x in parsed if str(x).strip()]
                except Exception:
                    pass
            # fallback: extract all integers in the string
            return re.findall(r"\d+", s)

        out = gdf.copy()
        out[col] = out[col].apply(to_tokens)
        out = gpd.GeoDataFrame(out, geometry="geometry", crs=crs)
        return out


    # right after reading vehicles_stats:
    vehicles_stats = normalize_route_ids_preserve_crs(vehicles_stats, "route_id")
    print("vehicles_stats route_id sample:", vehicles_stats["route_id"].head(5).tolist())


    #print("[DBG] vehicles_stats.crs:", vehicles_stats.crs)
    #print("[DBG] vehicle geometry types:", vehicles_stats.geometry.geom_type.value_counts(dropna=False).to_dict())
    #print("[DBG] sample uni_id + geom type:", vehicles_stats[['uni_id','geometry']].head(5).apply(lambda r: (r['uni_id'], getattr(r['geometry'],'geom_type',None)), axis=1).tolist())
    
    # NEW

    # Preproject once and prebuild features once
    vehicles_wgs84 = vehicles_stats[['uni_id', 'geometry']].copy()
    vehicles_wgs84['uni_id'] = vehicles_wgs84['uni_id'].astype(str)

    # Reproject whole GeoSeries once
    vehicles_wgs84 = vehicles_wgs84.set_geometry('geometry').to_crs(4326)

    # def _points_fc_fast(sel_gdf: gpd.GeoDataFrame, source_crs) -> dict:
    #     if sel_gdf is None or sel_gdf.empty:
    #         return {"type": "FeatureCollection", "features": []}

    #     gdf = sel_gdf[['uni_id', 'geometry']].copy()
    #     gdf['uni_id'] = gdf['uni_id'].astype(str).str.strip()
    #     if gdf.crs is None and source_crs is not None:
    #         gdf = gdf.set_crs(source_crs, allow_override=True)
    #     if (gdf.crs.to_epsg() or gdf.crs.to_string()) not in (4326, "EPSG:4326"):
    #         gdf = gdf.to_crs(4326)

    #     feats = []
    #     append = feats.append
    #     for _, r in gdf.iterrows():
    #         uid = r['uni_id']
    #         g = r.geometry
    #         if g is None or g.is_empty:
    #             continue
    #         gt = getattr(g, "geom_type", "")
    #         if gt == "Point":
    #             xs = [(g.x, g.y)]
    #         elif gt == "MultiPoint":
    #             # emit every point
    #             xs = [(p.x, p.y) for p in g.geoms if p and not p.is_empty]
    #         else:
    #             # for Line/Polygon, pick a single display point
    #             rp = (g.representative_point() if hasattr(g, "representative_point") else g.centroid)
    #             xs = [(rp.x, rp.y)]

    #         for x, y in xs:
    #             if not np.isfinite(x) or not np.isfinite(y):
    #                 continue
    #             append({
    #                 "type": "Feature",
    #                 "geometry": {"type": "Point", "coordinates": [round(float(x), 6), round(float(y), 6)]},
    #                 "properties": {"uni_id": uid},
    #             })

    #     return {"type": "FeatureCollection", "features": feats}
    # Tunables (adjust to taste)
    MAX_POINTS_PER_VEHICLE = 500       # hard cap per selected vehicle
    MAX_POINTS_PER_METHOD  = 8000     # hard cap across all vehicles (this method)
    GRID_DECIMALS          = 5        # 5 ≈ ~1m at mid-latitudes; 4 ≈ ~10m

    def _downsample_coords(coords, k):
        """Deterministic downsample: keep at most k points by even stride."""
        n = len(coords)
        if n <= k:
            return coords
        step = max(1, n // k)
        # take every 'step'th point; ensure we still cap to k
        return coords[::step][:k]

    def _points_fc_fast(sel_gdf: gpd.GeoDataFrame, source_crs) -> dict:
        if sel_gdf is None or sel_gdf.empty:
            return {"type": "FeatureCollection", "features": []}

        gdf = sel_gdf[['uni_id', 'geometry']].copy()
        gdf['uni_id'] = gdf['uni_id'].astype(str).str.strip()
        if gdf.crs is None and source_crs is not None:
            gdf = gdf.set_crs(source_crs, allow_override=True)
        if (gdf.crs.to_epsg() or gdf.crs.to_string()) not in (4326, "EPSG:4326"):
            gdf = gdf.to_crs(4326)

        feats = []
        total_points_emitted = 0
        append = feats.append

        for _, r in gdf.iterrows():
            if total_points_emitted >= MAX_POINTS_PER_METHOD:
                break

            uid = r['uni_id']
            g   = r.geometry
            if g is None or g.is_empty:
                continue

            gt = getattr(g, "geom_type", "")
            # 1) collect raw coords for this vehicle
            if gt == "Point":
                coords = [(g.x, g.y)]
            elif gt == "MultiPoint":
                coords = [(p.x, p.y) for p in g.geoms if p and not p.is_empty]
            else:
                # single “anchor” for non-point geometries
                rp = (g.representative_point() if hasattr(g, "representative_point") else g.centroid)
                coords = [(rp.x, rp.y)]

            # 2) de-duplicate close points by snapping to a small grid
            if GRID_DECIMALS is not None:
                seen = set()
                uniq = []
                for x, y in coords:
                    key = (round(float(x), GRID_DECIMALS), round(float(y), GRID_DECIMALS))
                    if key not in seen and np.isfinite(key[0]) and np.isfinite(key[1]):
                        seen.add(key)
                        uniq.append(key)
                coords = uniq

            # 3) per-vehicle cap (deterministic stride)
            coords = _downsample_coords(coords, MAX_POINTS_PER_VEHICLE)

            # 4) enforce per-method cap
            budget = MAX_POINTS_PER_METHOD - total_points_emitted
            if budget <= 0:
                break
            if len(coords) > budget:
                coords = coords[:budget]

            # 5) emit
            for x, y in coords:
                append({
                    "type": "Feature",
                    "geometry": {"type": "Point",
                                "coordinates": [round(float(x), 6), round(float(y), 6)]},
                    "properties": {"uni_id": uid},
                })
            total_points_emitted += len(coords)
        # total_points_crs = total_points_emitted['geometry'].crs

        # optional: log what happened
        # print(f"[DBG] selected points emitted: {total_points_emitted} "
        #     f"(per-veh≤{MAX_POINTS_PER_VEHICLE}, per-method≤{MAX_POINTS_PER_METHOD}, grid={GRID_DECIMALS}dp)")
        # print(f"[DBG] selected point crs: {total_points_crs}")
        # return {"type": "FeatureCollection", "features": feats}







    # END NEW

    # 3.  Run the full optimisation pipeline ----------------------------------


    # Spatial
    _r(35, "Running spatial optimization pipeline…", "optimizing")
    def _get_selected(city: City, timeframe):
        suffix = {"1day": "_1day", "3day": "_3days", "7day": "_7days"}.get(timeframe, "")
        city_label = {"Amsterdam":"amsterdam", "Utrecht":"utrecht", "Rotterdam":"rotterdam", "The Hague":"thehague"}.get(city, "")

        selected = pd.read_csv(data_root / "temp" / f"selected_{city_label}{suffix}.csv")

        return selected
    selected = _get_selected(city, timeframe)

    # _, _, max_space_vehicles = spatial_optimization_pipeline(
    #     points_gdf,
    #     cbs_full,
    #     vehicles_stats,
    #     coverage_threshold=3,
    #     top_n=n_sensors,
    # )
    optimized_ids, filtered_vehicles, max_space_vehicles = spatial_optimization_pipeline(vehicles_stats, selected, top_n=n_sensors)

    # Temporal
    _r(55, "Running temporal optimization pipeline…", "optimizing")
    _, _, max_temp_vehicles = temporal_optimization_pipeline(
        vehicles_stats,
        top_n=n_sensors,
    )

    # Fairness
    _r(65, "Running fairness optimization pipeline…", "optimizing")
    *_unused, fair_vehicles = run_fairness_pipeline(
        vehicles_stats,
        city_stats,
        n=n_sensors,
    )

    # Max coverage
    _r(75, "Running max coverage optimization pipeline…", "optimizing")
    tops, *_unused2, max_pop_vehicles = run_max_coverage_pipeline(
        vehicles_stats,
        cbs_full,
        n=n_sensors,
    )
    # print("[DBG] max_space_vehicles:", type(max_space_vehicles), "len:", getattr(max_space_vehicles, "__len__", lambda: None)())
    # print("[DBG] max_temp_vehicles:",  type(max_temp_vehicles),  "len:", getattr(max_temp_vehicles, "__len__", lambda: None)())
    # print("[DBG] fair_vehicles:",      type(fair_vehicles),      "len:", getattr(fair_vehicles, "__len__", lambda: None)())
    # print("[DBG] max_pop_vehicles:",   type(max_pop_vehicles),   "len:", getattr(max_pop_vehicles, "__len__", lambda: None)())


    # Combine
    _r(80, "Combining optimization results…", "combine")
    combined_df = combine_optimized_dfs(
        max_space_vehicles,
        max_temp_vehicles,
        max_pop_vehicles,
        fair_vehicles,
    )
    # print("[DBG] combined_df columns:", list(combined_df.columns))
    scores_combined_df, top_combined_final = compute_combined_optimization_scores(
        combined_df
    )

    # Random selection for baseline
    random_vehicles = select_random_vehicles(vehicles_stats, n=n_sensors)

    all_vehicles = pd.DataFrame({"all_vehicles": vehicles_stats["uni_id"].astype(str)})
    all_vehicles = all_vehicles.reset_index(drop=True)

    final_df_cells = vehicle_optimization_stats_pipeline(
        vehicles_stats,
        cbs_full,
        city_stats,
        max_space_vehicles,
        max_temp_vehicles,
        max_pop_vehicles,
        fair_vehicles,
        top_combined_final,
        random_vehicles,
        all_vehicles,
    )
    final_df_cells.to_csv('final_df_six.csv')
    # print("[DBG]: final_df_cells", list(final_df_cells))

    # Fix minor display quirks (original script comments)
    final_df_cells.at["cells_unique", 'Amsterdam'] = len(cbs_full)
    final_df_cells.at["avg_points_per_cell", 'Amsterdam'] = "-"


    _r(85, "Preparing selections & sensed cells…", "visualize")

    # 0) Column names used by get_joined_cbs_gdf
    VEH_CODES_COL = "crs28922_list"   # <-- change to your actual vehicles column
    CBS_CODE_COL  = "crs28992"        # <-- change to your CBS column if different

    # 1) Per-method selected vehicles
    sel_max_spatial  = prepare_selected_vehicles_from_combined(vehicles_stats, combined_df, "max_spatial")
    sel_max_temporal = prepare_selected_vehicles_from_combined(vehicles_stats, combined_df, "max_temporal")
    sel_fair         = prepare_selected_vehicles_from_combined(vehicles_stats, combined_df, "closest_relative")


    # 2) Ensure selections carry the CBS code list column expected by get_joined_cbs_gdf
    def _with_codes(sel_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        if VEH_CODES_COL in sel_gdf.columns:
            return sel_gdf
        # Bring in codes by uni_id from vehicles_stats
        if "uni_id" not in sel_gdf.columns:
            raise ValueError("Selected vehicles GeoDataFrame is missing 'uni_id'.")
        need_cols = ["uni_id", VEH_CODES_COL]
        missing = [c for c in need_cols if c not in vehicles_stats.columns]
        if missing:
            raise ValueError(f"vehicles_stats missing required columns: {missing}")
        return sel_gdf.merge(vehicles_stats[need_cols], on="uni_id", how="left")

    sel_max_spatial  = _with_codes(sel_max_spatial)
    sel_max_temporal = _with_codes(sel_max_temporal)
    sel_fair         = _with_codes(sel_fair)


    def _lines_fc_fast(
        lines_gdf: gpd.GeoDataFrame,
        epsg_metric: int = 28992,
        epsg_geo: int = 4326,
        simplify_m: float = 0.0,          # keep for interface compatibility, but unused
        max_vertices: int = 10_000_000    # hard cap just for safety
    ) -> dict:
        """
        Convert a GeoDataFrame of lines into a GeoJSON FeatureCollection.
        No simplification, no guessing, preserves full geometry.
        """

        if lines_gdf is None or lines_gdf.empty:
            return {"type": "FeatureCollection", "features": []}

        # Always work in the metric CRS
        if lines_gdf.crs is None or lines_gdf.crs.to_epsg() != epsg_metric:
            work = lines_gdf.to_crs(epsg_metric)
        else:
            work = lines_gdf.copy()

        # Project to WGS84 for the map
        wgs = work.to_crs(epsg_geo)

        feats = []
        vtx_count = 0
        for _, g in wgs.iterrows():
            geom = g.geometry
            if geom is None or geom.is_empty:
                continue
            parts = list(geom.geoms) if geom.geom_type == "MultiLineString" else [geom]
            for part in parts:
                coords = list(part.coords)
                vtx_count += len(coords)
                if vtx_count > max_vertices:
                    break
                feats.append({
                    "type": "Feature",
                    "geometry": {
                        "type": "LineString",
                        "coordinates": [(float(x), float(y)) for x, y in coords],
                    },
                    "properties": {},  # currently empty
                })
            if vtx_count > max_vertices:
                print(f"[WARN] vertex cap reached at {vtx_count:,} points")
                break

        print(f"[DBG] _lines_fc_fast (no simplification): features={len(feats)}, vertices≈{vtx_count}")
        return {"type": "FeatureCollection", "features": feats}


    def _to_route_tokens(cell) -> list[str]:
        # Always return clean digit-only strings
        if cell is None or (isinstance(cell, float) and pd.isna(cell)): 
            return []
        if isinstance(cell, (list, tuple, np.ndarray, pd.Series)):
            return [str(x).strip() for x in cell if str(x).strip()]
        s = str(cell).strip()
        if not s:
            return []
        # parse real Python list if present
        if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
            try:
                import ast
                parsed = ast.literal_eval(s)
                if isinstance(parsed, (list, tuple)):
                    return [str(x).strip() for x in parsed if str(x).strip()]
            except Exception:
                pass
        # otherwise split out any digit groups
        import re
        toks = re.findall(r"\d+", s)
        return [t for t in toks if t]

    _r(85, "Preparing selections & sensed cells…", "visualize")

    # --- Per-method joined cells (FAST; code-based) ---
    joined_spatial_gdf  = get_joined_cbs_gdf(sel_max_spatial,  cbs_full)
    joined_temporal_gdf = get_joined_cbs_gdf(sel_max_temporal, cbs_full)
    joined_fair_gdf     = get_joined_cbs_gdf(sel_fair,         cbs_full)

    cbs_codes_series = cbs_full[CBS_CODE_COL].astype(str)

    def _unsensed_cells_gdf(sensed_subset: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Return CBS cells that are not covered by the sensed subset."""
        if sensed_subset is None or sensed_subset.empty:
            return cbs_full[["geometry"]]
        if CBS_CODE_COL not in sensed_subset.columns:
            remaining_idx = cbs_full.index.difference(sensed_subset.index)
            return cbs_full.loc[remaining_idx, ["geometry"]]
        sensed_codes = sensed_subset[CBS_CODE_COL].astype(str)
        mask = ~cbs_codes_series.isin(sensed_codes)
        return cbs_full.loc[mask, ["geometry"]]

    unsensed_spatial_gdf  = _unsensed_cells_gdf(joined_spatial_gdf)
    unsensed_temporal_gdf = _unsensed_cells_gdf(joined_temporal_gdf)
    unsensed_fair_gdf     = _unsensed_cells_gdf(joined_fair_gdf)

    _r(87, "Sending geometry...", "visualize")
    # --- Base layers to GeoJSON (geometry-only keeps payload small) ---
    boundary_fc  = gdf_to_featurecollection(city_geo, 4326)
    all_cells_fc = gdf_to_featurecollection(cbs_full[['geometry']], 4326)

    _r(90, "Sending the map payloads", "visualize")
    joined_by_method = {
        "max_spatial":      gdf_to_featurecollection(joined_spatial_gdf[['geometry']], 4326),
        "max_temporal":     gdf_to_featurecollection(joined_temporal_gdf[['geometry']], 4326),
        "closest_relative": gdf_to_featurecollection(joined_fair_gdf[['geometry']], 4326),
    }

    _r(92, "Building map", "visualize")
    def _points_fc(gdf):
        cols = ["uni_id","geometry"] if "uni_id" in gdf.columns else ["geometry"]
        return gdf_to_featurecollection(gdf[cols], 4326)
    

    def plot_transport_and_population_interactive(
        vehicles_gdf: gpd.GeoDataFrame,
        sensed_gdf: gpd.GeoDataFrame,
        unsensed_gdf: gpd.GeoDataFrame,
        boundary_gdf: gpd.GeoDataFrame,
        buffer_distance: float,
    ) -> go.Figure:
        """Plotly replica of the Matplotlib map (no basemap, fast rendering)."""
        target_crs = "EPSG:28992"
        boundary = boundary_gdf.to_crs(target_crs)
        veh = vehicles_gdf.to_crs(target_crs)
        sensed = sensed_gdf.to_crs(target_crs)
        unsensed = unsensed_gdf.to_crs(target_crs)

        def polys_to_trace(gdf, name, fill_color):
            if gdf is None or gdf.empty:
                return None
            xs, ys = [], []
            for geom in gdf.geometry:
                if geom is None or geom.is_empty:
                    continue
                polys = geom.geoms if geom.geom_type == "MultiPolygon" else [geom]
                for poly in polys:
                    x, y = poly.exterior.xy
                    xs.extend(list(x) + [None])
                    ys.extend(list(y) + [None])
            if not xs:
                return None
            return go.Scatter(
                x=xs, y=ys, name=name,
                mode="lines", fill="toself",
                line=dict(color="white", width=0.4),
                fillcolor=fill_color,
                hoverinfo="skip",
            )

        def points_to_trace(gdf, name):
            if gdf is None or gdf.empty:
                return None
            xs, ys = [], []
            for geom in gdf.geometry:
                if geom is None or geom.is_empty:
                    continue
                if geom.geom_type == "Point":
                    xs.append(float(geom.x))
                    ys.append(float(geom.y))
                elif geom.geom_type == "MultiPoint":
                    for pt in geom.geoms:
                        xs.append(float(pt.x))
                        ys.append(float(pt.y))
                else:
                    rp = geom.representative_point()
                    xs.append(float(rp.x))
                    ys.append(float(rp.y))
            if not xs:
                return None
            return go.Scattergl(
                x=[float(val) for val in xs],
                y=[float(val) for val in ys],
                name=name,
                mode="markers",
                marker=dict(color="black", size=1.5, opacity=0.9),
                hoverinfo="skip",
            )

        traces = [
            polys_to_trace(unsensed, "CBS Cells", "#ff78a3"),
            polys_to_trace(sensed, "Sensed CBS Cells", "#92c421"),
            points_to_trace(veh, "Vehicles"),
        ]
        fig = go.Figure([t for t in traces if t is not None])

        for geom in boundary.geometry:
            if geom is None or geom.is_empty:
                continue
            boundary_geom = geom.boundary
            segments = boundary_geom.geoms if hasattr(boundary_geom, "geoms") else [boundary_geom]
            for seg in segments:
                x_arr, y_arr = seg.xy
                x = [float(val) for val in x_arr]
                y = [float(val) for val in y_arr]
                fig.add_trace(go.Scatter(
                    x=x, y=y,
                    mode="lines",
                    line=dict(color="black", width=0.6),
                    hoverinfo="skip",
                    showlegend=False,
                ))

        fig.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="white",
            legend=dict(x=0.02, y=0.98),
            margin=dict(l=10, r=10, t=20, b=10),
            xaxis=dict(visible=False, showgrid=False, zeroline=False),
            yaxis=dict(visible=False, showgrid=False, zeroline=False),
            dragmode="pan",
        )
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
        return fig

    method_to_sel = {
        "max_spatial": sel_max_spatial,
        "max_temporal": sel_max_temporal,
        "closest_relative": sel_fair,
    }
    method_to_joined = {
        "max_spatial": joined_spatial_gdf,
        "max_temporal": joined_temporal_gdf,
        "closest_relative": joined_fair_gdf,
    }
    method_to_unsensed = {
        "max_spatial": unsensed_spatial_gdf,
        "max_temporal": unsensed_temporal_gdf,
        "closest_relative": unsensed_fair_gdf,
    }

    artefacts: dict[str, dict[str, str]] = {}

    for method_key, sel_gdf in method_to_sel.items():
        _r(93, f"Building {method_key} map", "visualize")
        sensed_gdf = method_to_joined[method_key]
        unsensed_gdf = method_to_unsensed[method_key]

        map_fig = plot_transport_and_population_interactive(
            vehicles_gdf=sel_gdf,
            sensed_gdf=sensed_gdf,
            unsensed_gdf=unsensed_gdf,
            boundary_gdf=city_geo,
            buffer_distance=50,
        )

        artefacts.setdefault(method_key, {})
        outfile = output_dir / f"map_{method_key}.html"
        map_fig.write_html(
            outfile,
            include_plotlyjs="cdn",
            config={
                "displayModeBar": False,
                "scrollZoom": True,
                "doubleClick": "reset",
                "staticPlot": False,
            },
        )
        artefacts[method_key]["map_html"] = outfile.name





    # selected_by_method = {
    #     "max_spatial":      _points_fc(sel_max_spatial),
    #     "max_temporal":     _points_fc(sel_max_temporal),
    #     "closest_relative": _points_fc(sel_fair),
    # }
 
    # selected_by_method = {
    #     "max_spatial":      _points_fc_fast(sel_max_spatial,  vehicles_stats.crs),
    #     "max_temporal":     _points_fc_fast(sel_max_temporal, vehicles_stats.crs),
    #     "closest_relative": _points_fc_fast(sel_fair,         vehicles_stats.crs),
    # }
    # print("[DBG] selected_by_method counts:",
    #   {k: len(v["features"]) for k, v in selected_by_method.items()})
    # for name, sel in [("max_spatial", sel_max_spatial), ("max_temporal", sel_max_temporal), ("closest_relative", sel_fair)]:
    #     print(f"[DBG] {name} rows:", len(sel), "geom types:", sel.geometry.geom_type.value_counts(dropna=False).to_dict())

    _r(97, "Sending stats", "visualize")
    # --- Per-method stats for the charts (cheap; Pandas only) ---
    # graph_data = {}
    # for key, jgdf in [
    #     ("max_spatial", joined_spatial_gdf),
    #     ("max_temporal", joined_temporal_gdf),
    #     ("closest_relative", joined_fair_gdf),
    # ]:
    #     stats_ams    = generate_summary_statistics(cbs_gdf=cbs_full,   area_name=city)
    #     stats_sensed = generate_summary_statistics(cbs_gdf=jgdf,       area_name="Sensed Area")
    #     merged       = pd.concat([stats_ams, stats_sensed], ignore_index=True)

    #     avg_stats = normalize_statistics(merged)  # percentages
    #     sums_df   = calculate_and_compare_sums(cbs_full, jgdf, city)

    #     graph_data[key] = {
    #         "sums":           sums_df.to_dict(orient="list"),
    #         "average_stats":  avg_stats.to_dict(orient="list"),
    #     }
    # --- do this ONCE before the for key, jgdf in [...] loop ---
    if city == 'Amsterdam':
        _EXCL = ['crs28992', 'G_woz_woni', 'geometry', 'age_sum', 'migration_']
    else:
        _EXCL = ['crs28992', 'G_woz_woni', 'geometry', 'age_sum', 'migration_sum']

    # these are just for speed; final order will still come from sensed_gdf (see below)
    _cbs_numeric_cols = [c for c in cbs_full.columns
                        if c not in _EXCL and pd.api.types.is_numeric_dtype(cbs_full[c])]

    _cbs_total_sums = cbs_full[_cbs_numeric_cols].sum()

    # keep the original behavior for average_stats (city row once)
    stats_ams = generate_summary_statistics(cbs_gdf=cbs_full, area_name=city)

    def calculate_and_compare_sums_fast(cbs_total_sums: pd.Series, cbs_exclusions: list[str], sensed_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
        """
        Same output as your original calculate_and_compare_sums(), but:
        - 'Totals' are precomputed and reused
        - Column order matches the original (taken from sensed_gdf after drop)
        """
        # Get the exact column order you had before (drop on sensed_gdf)
        sensed_cols = [c for c in sensed_gdf.columns if c not in cbs_exclusions]

        # Align city totals to that order
        totals = cbs_total_sums.reindex(sensed_cols)

        # Compute sensed sums only on those columns (vectorized)
        sensed = sensed_gdf[sensed_cols].sum()

        # Build the same frame, same column names & order
        sums = pd.DataFrame({
            'Sociodemo':  sensed_cols,
            'Sums_sensed': sensed.values,
            'Sums_total':  totals.values,
        })

        # Same math + rounding/casting as your original
        with np.errstate(divide='ignore', invalid='ignore'):
            pct = (sums['Sums_sensed'] / sums['Sums_total']) * 100
        sums['Sensed_%']  = np.round(pct, 2)
        sums['Excluded!'] = np.round(sums['Sums_total'] - sums['Sums_sensed'], 0)

        # Match dtypes exactly
        sums['Sums_sensed'] = sums['Sums_sensed'].round(0).astype(int)
        sums['Sums_total']  = sums['Sums_total'].round(0).astype(int)
        sums['Sensed_%']    = sums['Sensed_%'].astype(float)
        sums['Excluded!']   = sums['Excluded!'].astype(int)

        return sums

    graph_data = {}
    for key, jgdf in [
        ("max_spatial", joined_spatial_gdf),
        ("max_temporal", joined_temporal_gdf),
        ("closest_relative", joined_fair_gdf),
    ]:
        # city row once (stats_ams) + sensed row per method (unchanged function)
        stats_sensed = generate_summary_statistics(cbs_gdf=jgdf, area_name="Sensed Area")

        merged       = pd.concat([stats_ams, stats_sensed], ignore_index=True)

        avg_stats    = normalize_statistics(merged)  # keep as-is for identical output
        print(avg_stats)
        sums_df      = calculate_and_compare_sums_fast(_cbs_total_sums, _EXCL, jgdf)
        print(sums_df)
        graph_data[key] = {
            "sums":           sums_df.to_dict(orient="list"),
            "average_stats":  avg_stats.to_dict(orient="list"),
        }




    return RunResult(
        job_id=output_dir.name,
        artefacts=artefacts,
        top_vehicles={
            "spatial":   _to_id_list(max_space_vehicles),
            "temporal":  _to_id_list(max_temp_vehicles),
            "fair":      _to_id_list(fair_vehicles),
            "population":_to_id_list(max_pop_vehicles),
        },
        stats_table=final_df_cells.reset_index().to_dict(orient="list"),
        geojson={
            "boundary": boundary_fc,
            "all_cells": all_cells_fc,
            "joined_by_method": joined_by_method,       
            #"selected_by_method": selected_by_method, 
            # "transport_by_method": transport_by_method,
        },
        graph_data=graph_data,
    )

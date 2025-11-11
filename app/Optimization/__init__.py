from .analysis_vehicles_stats import prepare_vehicles_with_stats # first make stats for vehicles
from .optimization_vehicles_spatial import spatial_optimization_pipeline # then optimize the vehicles for spatial coverage
from .optimization_vehicles_temporal import temporal_optimization_pipeline # then optimize the vehicles for temporal coverage
from .optimization_vehicles_fairness import run_fairness_pipeline # then optimize the vehicles for fairness
from .optimization_vehicles_maximum import run_max_coverage_pipeline # then optimize the vehicles for maximum inhabitants, points, etc. 
from .create_combined_df import combine_optimized_dfs # combine the results into one dataframe
from .optimization_vehicles_combined import compute_combined_optimization_scores # compute the final scores combined vehicles optimization 
from .optimization_big_merge_stats_VIZ_points import select_random_vehicles # select random vehicles for optmization 
from .optimization_big_merge_stats_VIZ_points import vehicle_optimization_stats_pipeline # run the optimization pipeline for stats
from .optimization_big_merge_stats_VIZ_points import plot_vehicles_by_group # plot the vehicles by optmization - quick visualization 
from .optimization_big_merge_stats_VIZ_points import extract_string_lists # extract the string lists from the dataframe for analysis
from .create_optimized_vehicles_gdf import prepare_selected_vehicles_from_combined # prepare the selected vehicles from the combined dataframe for analysis and visualization
from .vehicle_VIZ_stats_exports import master_function_analysis # run the analysis for the selected vehicles
from .vehicle_VIZ_stats_exports import visualization_master_function # run the visualization for the selected vehicles
from .calculate_VIZ_frequencies import pipeline_plot_frequency # run the pipeline for plotting the frequencies of the selected vehicles




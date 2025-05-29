# analyze_artifacts.py (Version 2.5.1 - Clustered High-Res Plots & Report)
import json
import pandas as pd
from pathlib import Path
import traceback
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from tqdm import tqdm

try:
    from . import config # If part of a package
except ImportError:
    import config # Fallback for running as script

# --- Configuration ---
DEFAULT_ARTIFACTS_INPUT_DIR = config.ARTIFACTS_DIR
USER_DESKTOP = Path.home() / "Desktop"
CHUNKS_ROOT_DIR = USER_DESKTOP / "Experiment Chunks" # Adjust if your chunks are elsewhere
ANALYSIS_OUTPUT_ROOT_DIR = config.ANALYSIS_OUTPUT_ROOT_DIR
BASIS_METADATA_DIR = config.DEFAULT_BASIS_SEARCH_DIR / "basesdata"

SCRIPT_VERSION_SUFFIX = "v2.5.1_clusteredHR" # Updated version suffix

CLUSTER_ASSIGNMENTS_CSV_PATH = ANALYSIS_OUTPUT_ROOT_DIR / "basis_similarity_analysis" / "basis_geometric_clusters_thresh15deg.csv"
# --- End Configuration ---

def parse_basis_filename(filename_str: str | None):
    if not filename_str:
        return {
            "basis_npz_filename": None, "filename_detected_method_prefix_code": None,
            "basis_cat_from_filename": "Unknown_NoFilename", "basis_style_hint_from_filename": "Unknown",
            "basis_layer_from_filename": "Unknown", "basis_version_from_filename": "Unknown",
            "basis_type_from_filename": "UnknownType_NoFilename"
        }
    name_part = Path(filename_str).stem; parts = name_part.split('_'); b_type_for_plot = name_part
    temp_layer = "L?"; temp_version = "v?"
    if len(parts) >= 3 and parts[-2].startswith("L") and len(parts[-2]) > 1 and parts[-2][1:].isdigit() and \
       parts[-1].startswith("v") and len(parts[-1]) > 1 and parts[-1][1:].isdigit():
        b_type_for_plot = "_".join(parts[:-2]); temp_layer = parts[-2]; temp_version = parts[-1]
    elif len(parts) >= 2:
        if parts[-1].startswith("L") and len(parts[-1]) > 1 and parts[-1][1:].isdigit():
            b_type_for_plot = "_".join(parts[:-1]); temp_layer = parts[-1]
            if len(parts) > 1 and parts[-2].startswith("v") and len(parts[-2]) > 1 and parts[-2][1:].isdigit(): temp_version = parts[-2]
        elif parts[-1].startswith("v") and len(parts[-1]) > 1 and parts[-1][1:].isdigit():
            b_type_for_plot = "_".join(parts[:-1]); temp_version = parts[-1]
            if len(parts) > 1 and parts[-2].startswith("L") and len(parts[-2]) > 1 and parts[-2][1:].isdigit(): temp_layer = parts[-2]
    if not b_type_for_plot: b_type_for_plot = name_part
    detected_method_code = None; known_prefixes_list = ["DRND", "DMEAN", "PCA"]
    if parts and parts[0].upper() in known_prefixes_list:
        detected_method_code = parts[0]; b_cat_heuristic = parts[1] if len(parts) > 1 else detected_method_code
        b_style_heuristic = parts[2] if len(parts) > 2 else "Style?"
    else:
        b_cat_heuristic = parts[0] if parts else "UnknownCat"
        b_style_heuristic = parts[1] if len(parts) > 1 else "Style?"
    return {
        "basis_npz_filename": filename_str, "filename_detected_method_prefix_code": detected_method_code,
        "basis_cat_from_filename": b_cat_heuristic, "basis_style_hint_from_filename": b_style_heuristic,
        "basis_layer_from_filename": temp_layer, "basis_version_from_filename": temp_version,
        "basis_type_from_filename": b_type_for_plot
    }

def process_single_artifact(artifact_filepath: Path):
    try:
        with open(artifact_filepath, 'r', encoding='utf-8') as f: artifact_json_content = json.load(f)
        raw = artifact_json_content.get("raw_data", {}); proj_baseline = raw.get("prompt_vector_projection_baseline", {})
        align_baseline = raw.get("final_state_alignment_baseline", {})
        basis_npz_filename_from_raw = raw.get("basis_file_used")
        actual_basis_npz_filename = Path(basis_npz_filename_from_raw).name if basis_npz_filename_from_raw else None
        
        parsed_info_from_npz_fn = parse_basis_filename(actual_basis_npz_filename)
        basis_companion_meta = {}
        if actual_basis_npz_filename:
            meta_json_path = BASIS_METADATA_DIR / (Path(actual_basis_npz_filename).stem + ".json")
            if meta_json_path.is_file():
                try:
                    with open(meta_json_path, 'r', encoding='utf-8') as bf: loaded_meta = json.load(bf)
                    basis_companion_meta["bmeta_source_prompt_A_orig"] = loaded_meta.get("source_prompt_A")
                    basis_companion_meta["bmeta_source_prompt_B_orig"] = loaded_meta.get("source_prompt_B")
                    basis_companion_meta["bmeta_concept_orig"] = loaded_meta.get("basis_concept")
                    basis_companion_meta["bmeta_style_orig"] = loaded_meta.get("basis_style")
                    basis_companion_meta["bmeta_version_orig"] = loaded_meta.get("basis_version")
                    basis_companion_meta["bmeta_gen_method_orig"] = loaded_meta.get("generation_method")
                    basis_companion_meta["bmeta_neuron_A_idx"] = loaded_meta.get("neuron_A_idx")
                    basis_companion_meta["bmeta_neuron_B_idx"] = loaded_meta.get("neuron_B_idx")
                    basis_companion_meta["bmeta_gen_method_selected"] = loaded_meta.get("generation_method_selected")
                    basis_companion_meta["bmeta_u1_derivation"] = loaded_meta.get("u1_derivation_details")
                    basis_companion_meta["bmeta_u2_derivation"] = loaded_meta.get("u2_derivation_details")
                    prompts_A_list = loaded_meta.get("source_prompts_A", [])
                    basis_companion_meta["bmeta_prompts_A_count"] = len(prompts_A_list) if isinstance(prompts_A_list, list) else 0
                    basis_companion_meta["bmeta_prompt_A_first"] = prompts_A_list[0] if prompts_A_list else None
                    prompts_B_list = loaded_meta.get("source_prompts_B", [])
                    basis_companion_meta["bmeta_prompts_B_count"] = len(prompts_B_list) if isinstance(prompts_B_list, list) else 0
                    basis_companion_meta["bmeta_prompt_B_first"] = prompts_B_list[0] if prompts_B_list else None
                    basis_companion_meta["bmeta_target_layer"] = loaded_meta.get("target_layer")
                    basis_companion_meta["bmeta_model_name"] = loaded_meta.get("model_name")
                    basis_companion_meta["bmeta_concept"] = loaded_meta.get("basis_concept") # Repeated for easier access
                    basis_companion_meta["bmeta_style"] = loaded_meta.get("basis_style")
                    basis_companion_meta["bmeta_version"] = loaded_meta.get("basis_version")
                    basis_companion_meta["bmeta_pca_samples_collected"] = loaded_meta.get("pca_num_samples_collected")
                    # ... Add any other bmeta fields you need ...
                except Exception as e_meta: print(f"Warn: Error processing basis meta {meta_json_path}: {e_meta}")
        record = {
            "artifact_id": artifact_json_content.get("artifact_id"), "artifact_timestamp": artifact_json_content.get("timestamp"),
            "input_prompt_text": raw.get("input_prompt"), "input_prompt_class": raw.get("prompt_class_name_batch"),
            "basis_npz_filename": actual_basis_npz_filename, 
            **{k:v for k,v in parsed_info_from_npz_fn.items() if k != "basis_npz_filename"},
            **basis_companion_meta,
            "theta_baseline": proj_baseline.get("angle_deg"), "r_baseline": proj_baseline.get("r"),
            "x_proj_baseline": proj_baseline.get("x"), "y_proj_baseline": proj_baseline.get("y"),
            "sim_b1_baseline": align_baseline.get("sim_basis1"), "sim_b2_baseline": align_baseline.get("sim_basis2"),
            "generated_text_snippet": raw.get("generated_text", "")[:150],
            "collapse_detected": raw.get("collapse_detected"),
            "interventions_enabled_in_run": raw.get("interventions_enabled_during_run"),
        }
        return record
    except Exception as e: print(f"CRITICAL Error processing artifact {artifact_filepath.name}: {e}"); traceback.print_exc(); return None

def select_artifact_source_interactively(default_artifacts_dir: Path, chunks_root_dir: Path) -> list[Path]:
    # This function remains IDENTICAL to your v2.4.4 / previous versions
    print("\n--- Select Artifact Source ---")
    print(f"1: Default artifact directory ({default_artifacts_dir.resolve()})")
    print(f"2: Analyze specific chunk(s) from '{chunks_root_dir.resolve()}'")
    source_paths_to_analyze = []
    while True:
        choice = input("Enter your choice (1 or 2, or 'q' to quit script): ").strip().lower()
        if choice == 'q': print("Exiting script by user choice."); return []
        if choice == '1':
            if default_artifacts_dir.exists() and any(default_artifacts_dir.glob("ART-*.json")):
                source_paths_to_analyze.append(default_artifacts_dir)
                print(f"Selected default artifact directory: {default_artifacts_dir.name}"); return source_paths_to_analyze
            else: print(f"Warning: Default directory {default_artifacts_dir} missing or empty of ART-*.json files."); continue
        elif choice == '2':
            if not chunks_root_dir.is_dir(): print(f"ERROR: Chunks root '{chunks_root_dir}' not found."); return []
            available_chunks_roots = sorted([d for d in chunks_root_dir.iterdir() if d.is_dir() and not d.name.startswith(".")])
            if not available_chunks_roots: print(f"No subdirectories (chunks) found in '{chunks_root_dir}'."); return []
            print("\nAvailable chunks (base folders):"); [print(f"  {i+1}: {d.name}") for i, d in enumerate(available_chunks_roots)]
            while True:
                chunk_choices_str = input(f"Enter chunk number(s) (e.g., '1', '1,3'), 'all', 'back', or 'q': ").strip().lower()
                if chunk_choices_str == 'q': print("Exiting script."); return []
                if chunk_choices_str == 'back': break
                current_selection = []
                if chunk_choices_str == 'all': selected_indices = list(range(len(available_chunks_roots)))
                else:
                    try: selected_indices = [int(x.strip()) - 1 for x in chunk_choices_str.split(',')]
                    except ValueError: print("Invalid format."); continue
                    if not all(0 <= idx < len(available_chunks_roots) for idx in selected_indices): print("Invalid number(s)."); continue
                for idx in selected_indices:
                    artifacts_subfolder = available_chunks_roots[idx] / "artifacts"
                    if artifacts_subfolder.is_dir() and any(artifacts_subfolder.glob("ART-*.json")): current_selection.append(artifacts_subfolder)
                    else: print(f"Warning: Chunk '{available_chunks_roots[idx].name}' has no valid 'artifacts' subfolder with ART-*.json files. Skipped.")
                if current_selection:
                    source_paths_to_analyze = current_selection
                    print(f"Selected artifact sets: {[p.parent.name for p in source_paths_to_analyze]}"); return source_paths_to_analyze
                else: print("No valid chunks selected or found with artifacts.")
            if chunk_choices_str == 'back': continue # Go back to source selection (1 or 2)
        else: print("Invalid choice. Please enter 1, 2, or 'q'.")
    return [] # Should be unreachable if logic is correct

def main():
    ANALYSIS_OUTPUT_ROOT_DIR.mkdir(parents=True, exist_ok=True)

    df_clusters = None
    if not CLUSTER_ASSIGNMENTS_CSV_PATH.is_file():
        print(f"WARNING: Cluster assignments file not found at {CLUSTER_ASSIGNMENTS_CSV_PATH}")
        print("Proceeding without cluster IDs. Run calculate_basis_similarity.py (with clustering) to generate it.")
    else:
        try:
            df_clusters = pd.read_csv(CLUSTER_ASSIGNMENTS_CSV_PATH)
            print(f"Successfully loaded cluster assignments from: {CLUSTER_ASSIGNMENTS_CSV_PATH}")
            if 'filename' not in df_clusters.columns or 'cluster_id' not in df_clusters.columns:
                print("ERROR: Cluster CSV must contain 'filename' (for basis_npz_filename) and 'cluster_id' columns.")
                df_clusters = None
        except Exception as e:
            print(f"ERROR loading cluster assignments CSV: {e}"); df_clusters = None

    artifact_sources_to_process = select_artifact_source_interactively(DEFAULT_ARTIFACTS_INPUT_DIR, CHUNKS_ROOT_DIR)
    if not artifact_sources_to_process: print("No artifact sources selected. Exiting."); return

    for current_artifacts_input_dir in artifact_sources_to_process:
        source_name = current_artifacts_input_dir.parent.name if current_artifacts_input_dir.name == "artifacts" and current_artifacts_input_dir.parent.parent == CHUNKS_ROOT_DIR else current_artifacts_input_dir.name
        if current_artifacts_input_dir == DEFAULT_ARTIFACTS_INPUT_DIR: source_name = "default_full_set"
        
        print(f"\n\n{'='*10} Starting Analysis for: {source_name} ({SCRIPT_VERSION_SUFFIX}) {'='*10}")
        current_chunk_output_base_dir = ANALYSIS_OUTPUT_ROOT_DIR / f"analysis_results_{source_name}"
        current_chunk_output_base_dir.mkdir(parents=True, exist_ok=True)
        master_data_output_csv = current_chunk_output_base_dir / f"master_artifact_data_{source_name}_{SCRIPT_VERSION_SUFFIX}.csv"
        plots_output_dir = current_chunk_output_base_dir / f"plots_{source_name}_{SCRIPT_VERSION_SUFFIX}"
        plots_output_dir.mkdir(parents=True, exist_ok=True)
        
        all_artifact_files = sorted(list(current_artifacts_input_dir.glob("ART-*.json")))
        if not all_artifact_files: print(f"No artifact files in {current_artifacts_input_dir}. Skipping."); continue
        print(f"Found {len(all_artifact_files)} artifact files for '{source_name}'.")
        
        all_records = [rec for filepath in tqdm(all_artifact_files, desc=f"Processing Artifacts ({source_name})", unit="file") if (rec := process_single_artifact(filepath)) is not None]
        if not all_records: print(f"No records extracted for '{source_name}'. Skipping."); continue

        df = pd.DataFrame(all_records)
        for col in ['r_baseline', 'theta_baseline']:
            if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
        print(f"\nProcessed {len(df)} records into DataFrame for '{source_name}'.")

        if df_clusters is not None and not df.empty:
            if 'basis_npz_filename' not in df.columns:
                print(f"ERROR: 'basis_npz_filename' column missing in main DataFrame for '{source_name}'. Cannot merge cluster IDs.")
            else:
                df = pd.merge(df, df_clusters[['filename', 'cluster_id']], left_on='basis_npz_filename', right_on='filename', how='left')
                if 'filename_x' in df.columns and 'filename_y' in df.columns : # pandas created these due to 'filename' in both
                     df = df.drop(columns=['filename_y']).rename(columns={'filename_x': 'filename_from_df_clusters_if_collided'})
                elif 'filename' in df.columns and 'basis_npz_filename' in df.columns and 'filename' in df_clusters.columns:
                     # If 'filename' was the merge key from right and df already had a 'filename' not used as left_on key.
                     # This case should be rare if basis_npz_filename is consistently used.
                     # If df_clusters.filename (right_on='filename') was merged and named 'filename' in the output df,
                     # and df already had a different 'filename' column, it's complex.
                     # Safest is to ensure 'basis_npz_filename' is the unique key from left.
                     # The provided merge `right_on='filename'` means the column from `df_clusters` that was used for merging is 'filename'.
                     # If 'filename' appears in the merged df and it's not from `basis_npz_filename`, it might be the one from `df_clusters`.
                     # If `df` had a pre-existing `filename` column (other than `basis_npz_filename`), it would be `filename_x`.
                     # The current logic seems to try and handle it, but careful check of output columns is good.
                     # The key is that df['cluster_id'] gets populated correctly.
                     pass


                if 'cluster_id' in df.columns:
                    num_na_clusters = df['cluster_id'].isna().sum()
                    if num_na_clusters > 0: print(f"Warning: {num_na_clusters} records in '{source_name}' did not get a cluster_id (check 'basis_npz_filename' matching 'filename' in cluster CSV).")
                else: print(f"Warning: 'cluster_id' column not added after merge for '{source_name}'.")
                print(f"Merged cluster_id into DataFrame for '{source_name}'.")
        elif df_clusters is None: print(f"Skipping cluster_id merge for '{source_name}'.")
        
        print(f"\n--- Data Verification ({source_name}, {SCRIPT_VERSION_SUFFIX}) ---")
        key_cols_to_verify = ["filename_detected_method_prefix_code", "basis_cat_from_filename", "basis_style_hint_from_filename", "basis_layer_from_filename", "basis_version_from_filename", "basis_type_from_filename", "input_prompt_class", "bmeta_gen_method_selected", "bmeta_concept", "bmeta_style", "bmeta_version", "cluster_id"]
        for col in key_cols_to_verify:
            if col in df.columns:
                unique_vals = df[col].dropna().unique()
                display_vals_str = str(sorted(list(str(x) for x in unique_vals))[:15]) + (f"... ({len(unique_vals) - 15} more)" if len(unique_vals) > 15 else "")
                print(f"Unique values in '{col}' ({len(unique_vals)}): {display_vals_str}")
            else: print(f"Warn: Column '{col}' not found for verification in '{source_name}'.")
        print(f"DataFrame columns for '{source_name}' ({len(df.columns)} total): {df.columns.tolist()}")

        try:
            df.to_csv(master_data_output_csv, index=False, float_format='%.6f')
            print(f"\nMaster data for '{source_name}' saved to: {master_data_output_csv}")
        except Exception as e_save: print(f"ERROR saving master CSV for '{source_name}': {e_save}")

        print(f"\n--- Generating Analysis Plots ({source_name}, {SCRIPT_VERSION_SUFFIX}) ---")
        if df.empty: print(f"DataFrame empty for '{source_name}', skipping plots."); continue

        # --- Plot 1: Overall r_baseline distribution ---
        if "r_baseline" in df.columns and not df["r_baseline"].isnull().all():
            plt.figure(figsize=(12, 7)); sns.histplot(data=df, x="r_baseline", kde=True, bins=50)
            plt.title(f"Overall Distribution of r_baseline ({source_name}, {SCRIPT_VERSION_SUFFIX})")
            plt.savefig(plots_output_dir / f"r_baseline_overall_dist_{source_name}.png"); plt.close()
            print(f"Plot saved: r_baseline_overall_dist_{source_name}.png")

        # --- Plot 2: r_baseline by input_prompt_class ---
        if "r_baseline" in df.columns and "input_prompt_class" in df.columns and not df["input_prompt_class"].isnull().all():
            plt.figure(figsize=(max(15, len(df["input_prompt_class"].dropna().unique()) * 0.6 ), 10)) # Slightly more width per class
            order_pc = sorted(df["input_prompt_class"].dropna().unique())
            sns.boxplot(data=df, y="r_baseline", x="input_prompt_class", showfliers=False, order=order_pc)
            plt.title(f"r_baseline by Input Prompt Class ({source_name}, {SCRIPT_VERSION_SUFFIX})")
            plt.xticks(rotation=75, ha='right', fontsize=9); plt.tight_layout(); # Increased fontsize
            plt.savefig(plots_output_dir / f"r_baseline_by_prompt_class_{source_name}.png"); plt.close()
            print(f"Plot saved: r_baseline_by_prompt_class_{source_name}.png")

        # --- Plots 3 & 4: r_baseline by SEMANTIC basis_type_from_filename (Alpha & Median Sorted) ---
        if "r_baseline" in df.columns and "basis_type_from_filename" in df.columns and not df["basis_type_from_filename"].isnull().all():
            unique_bt_semantic = sorted(df["basis_type_from_filename"].dropna().unique())
            plt.figure(figsize=(max(16, len(unique_bt_semantic) * 0.4), 10)) # More width
            sns.boxplot(data=df, y="r_baseline", x="basis_type_from_filename", order=unique_bt_semantic, showfliers=False)
            plt.title(f"r_baseline by Semantic Basis Type (Alphabetical) ({source_name}, {SCRIPT_VERSION_SUFFIX})")
            plt.xticks(rotation=85, ha='right', fontsize=7); plt.tight_layout(); # Smaller fontsize for many types
            plt.savefig(plots_output_dir / f"r_baseline_by_SEMANTIC_basis_type_alpha_{source_name}.png"); plt.close()
            print(f"Plot saved: r_baseline_by_SEMANTIC_basis_type_alpha_{source_name}.png")
            
            df_num_r_sem = df.dropna(subset=['r_baseline'])
            if not df_num_r_sem.empty:
                median_r_bt_sem = df_num_r_sem.groupby("basis_type_from_filename")["r_baseline"].median().sort_values(ascending=False)
                sorted_bt_median_sem = median_r_bt_sem.index.tolist()
                plt.figure(figsize=(max(16, len(sorted_bt_median_sem) * 0.4), 10)) # More width
                sns.boxplot(data=df, y="r_baseline", x="basis_type_from_filename", order=sorted_bt_median_sem, showfliers=False)
                plt.title(f"r_baseline by Semantic Basis Type (Sorted Median) ({source_name}, {SCRIPT_VERSION_SUFFIX})")
                plt.xticks(rotation=85, ha='right', fontsize=7); plt.tight_layout(); # Smaller fontsize
                plt.savefig(plots_output_dir / f"r_baseline_by_SEMANTIC_basis_type_median_{source_name}.png"); plt.close()
                print(f"Plot saved: r_baseline_by_SEMANTIC_basis_type_median_{source_name}.png")
        
        # --- New Plot: r_baseline by Geometric Cluster (High-Res) ---
        if "cluster_id" in df.columns and not df["cluster_id"].isnull().all() and \
           "r_baseline" in df.columns and not df["r_baseline"].isnull().all() and \
           "basis_type_from_filename" in df.columns :
            
            df_plot_cluster = df.dropna(subset=['cluster_id', 'r_baseline', 'basis_type_from_filename']).copy()
            df_plot_cluster.loc[:, 'cluster_id_str'] = df_plot_cluster['cluster_id'].astype(int).astype(str)
            median_r_by_cluster_overall = df_plot_cluster.groupby('cluster_id_str')['r_baseline'].median().sort_values(ascending=False)
            sorted_cluster_ids_by_median_r = median_r_by_cluster_overall.index.tolist()

            def get_cluster_label_fn(series_basis_types):
                modes = series_basis_types.mode(); label = "UnknownType"
                if not modes.empty:
                    label = modes[0]
                    num_unique_types = len(series_basis_types.unique())
                    num_bases_in_cluster = len(series_basis_types) # Count of rows for this type in this cluster
                    if num_unique_types > 1 : label += f" (Mix+{num_unique_types-1})" # If multiple distinct basis_type_from_filename in cluster
                    # This count below (num_bases_in_cluster) is now for unique basis_npz_filename, not rows in df_plot_cluster
                    # This was corrected in the following block for the report
                return label
            
            unique_bases_in_clusters = df_plot_cluster[['cluster_id_str', 'basis_npz_filename', 'basis_type_from_filename']].drop_duplicates()
            # Recalculate semantic labels map based on unique bases per cluster
            def get_cluster_label_from_unique(group): # group is a df for one cluster_id_str
                modes = group['basis_type_from_filename'].mode()
                label = "UnknownType"
                if not modes.empty:
                    label = modes[0]
                    num_unique_types_in_cluster = group['basis_type_from_filename'].nunique()
                    num_actual_bases_in_cluster = group['basis_npz_filename'].nunique()
                    if num_unique_types_in_cluster > 1: label += f" (Mix+{num_unique_types_in_cluster-1})"
                    elif num_actual_bases_in_cluster > 1 : label += f" (x{num_actual_bases_in_cluster})" # e.g. (x2) if 2 bases of same type
                return label
            cluster_semantic_labels_map = unique_bases_in_clusters.groupby('cluster_id_str').apply(get_cluster_label_from_unique).to_dict()
            
            df_plot_cluster.loc[:, 'cluster_display_label'] = df_plot_cluster['cluster_id_str'].apply(
                lambda cid_str: f"Cl{cid_str}:{cluster_semantic_labels_map.get(cid_str, 'Err')[:35]}"
            )
            sorted_cluster_display_labels = [f"Cl{cid_str}:{cluster_semantic_labels_map.get(cid_str, 'Err')[:35]}" for cid_str in sorted_cluster_ids_by_median_r]
            
            num_clusters_to_plot = len(sorted_cluster_display_labels)
            if num_clusters_to_plot > 0:
                ideal_width_per_label_inch = 0.25; calculated_width = num_clusters_to_plot * ideal_width_per_label_inch
                fig_width = min(120, max(24, calculated_width)); fig_height = 15
                labels_per_inch = num_clusters_to_plot / fig_width if fig_width > 0 else float('inf')
                
                if labels_per_inch <= 5: tf = 7
                elif labels_per_inch <= 10: tf = 6
                elif labels_per_inch <= 15: tf = 5
                elif labels_per_inch <= 20: tf = 4
                else: tf = 3

                plt.figure(figsize=(fig_width, fig_height))
                sns.boxplot(data=df_plot_cluster, y="r_baseline", x="cluster_display_label", order=sorted_cluster_display_labels, showfliers=False, palette="Spectral_r")
                plt.title(f"r_baseline by Geometric Cluster (Sorted Overall Median r_baseline) - Thresh 15deg\n({source_name}, {SCRIPT_VERSION_SUFFIX})", fontsize=16)
                plt.xticks(rotation=90, ha='right', fontsize=tf); plt.yticks(fontsize=12)
                plt.ylabel("r_baseline (all test prompts)", fontsize=14); plt.xlabel(f"Geometric Cluster ID (Dominant Semantic Type) - Unique Clusters: {num_clusters_to_plot}", fontsize=14)
                plt.tight_layout(pad=1.5)
                plot_fn_cl = plots_output_dir / f"r_baseline_by_geom_cluster_median_HIGHRES_{source_name}.png"
                try: plt.savefig(plot_fn_cl, dpi=300); plt.close()
                except Exception as e_save_plot: print(f"Error saving geom cluster plot: {e_save_plot}"); plt.close()
                print(f"High-resolution geometric cluster plot saved: {plot_fn_cl}")

                N_TOP_BOTTOM = 25; report_lines = [f"\n--- Top/Bottom {N_TOP_BOTTOM} Geom Clusters by Overall Median r_baseline ({source_name}, Thresh 15deg) ---", f"Total unique geom clusters: {num_clusters_to_plot}\n", "--- Top N ---"]
                for i, cid_s in enumerate(sorted_cluster_ids_by_median_r[:N_TOP_BOTTOM]):
                    med_r = median_r_by_cluster_overall.get(cid_s, np.nan); sem_l = cluster_semantic_labels_map.get(cid_s, "Unk")
                    bases_ic_df = unique_bases_in_clusters[unique_bases_in_clusters['cluster_id_str'] == cid_s]
                    bases_ic = bases_ic_df['basis_npz_filename'].unique() # Get unique filenames
                    bases_s = ", ".join(bases_ic[:3]) + (f", ...({len(bases_ic)-3} more)" if len(bases_ic)>3 else "")
                    report_lines.append(f"{i+1}. ClID:{cid_s} | Med.r_base:{med_r:.4f} | Dom.Type:{sem_l} | Bases({len(bases_ic)}):{bases_s}")
                if num_clusters_to_plot > N_TOP_BOTTOM * 2 : # Ensure there's enough separation for bottom N
                    report_lines.append("\n--- Bottom N ---")
                    for i, cid_s in enumerate(sorted_cluster_ids_by_median_r[-N_TOP_BOTTOM:]):
                        med_r = median_r_by_cluster_overall.get(cid_s,np.nan); sem_l = cluster_semantic_labels_map.get(cid_s,"Unk")
                        bases_ic_df = unique_bases_in_clusters[unique_bases_in_clusters['cluster_id_str'] == cid_s]
                        bases_ic = bases_ic_df['basis_npz_filename'].unique()
                        bases_s = ", ".join(bases_ic[:3]) + (f", ...({len(bases_ic)-3} more)" if len(bases_ic)>3 else "")
                        report_lines.append(f"{num_clusters_to_plot-N_TOP_BOTTOM+i+1}. ClID:{cid_s} | Med.r_base:{med_r:.4f} | Dom.Type:{sem_l} | Bases({len(bases_ic)}):{bases_s}")
                [print(line) for line in report_lines]
                report_fn = plots_output_dir / f"top_bottom_geom_clusters_report_{source_name}.txt"
                with open(report_fn, 'w', encoding='utf-8') as f_rep: [f_rep.write(line + "\n") for line in report_lines]
                print(f"Top/Bottom cluster report saved to: {report_fn}")
            else: print(f"No clusters with data to plot for '{source_name}'.")
        else: print(f"Skipping r_baseline by geom cluster plot for '{source_name}': missing data.")
        
        # --- Existing Oblique/Polar Plots (Theta plots) ---
        if "theta_baseline" in df.columns and not df["theta_baseline"].isnull().all() and \
           "input_prompt_class" in df.columns and "basis_type_from_filename" in df.columns:
            oblique_class_names = ["Class 3A Traditional Obliques", "Class 3D Metaphoric Drift", 
                                   "Class 3G Failure Simulation", "Class 3H Recursive Embeddings", 
                                   "Class 3I Structural Contradictions"]
            if "TargetSemantic_Safe" in df["basis_type_from_filename"].unique(): # Check if this semantic type exists
                df_oblique_target = df[
                    df["input_prompt_class"].isin(oblique_class_names) & 
                    (df["basis_type_from_filename"] == "TargetSemantic_Safe") # Ensure this type is exactly as in your df
                ].copy()
                if not df_oblique_target.empty and not df_oblique_target["theta_baseline"].isnull().all():
                    if 'theta_norm' not in df_oblique_target.columns: 
                         df_oblique_target.loc[:, "theta_norm"] = df_oblique_target["theta_baseline"].fillna(0) % 360

                    plt.figure(figsize=(12,7)); sns.histplot(data=df_oblique_target, x="theta_norm", bins=36, kde=True)
                    plt.title(f"Theta for Oblique Prompts on 'TargetSemantic_Safe' Bases ({source_name}, {SCRIPT_VERSION_SUFFIX})")
                    plt.xlabel("theta_baseline (deg, 0-360)"); plt.xticks(np.arange(0,361,45)); plt.xlim(0,360)
                    plt.savefig(plots_output_dir / f"theta_oblique_on_target_safe_{source_name}.png"); plt.close()
                    print(f"Plot saved: theta_oblique_on_target_safe_{source_name}.png")

                    if "r_baseline" in df_oblique_target.columns and not df_oblique_target['r_baseline'].isnull().all():
                        df_polar = df_oblique_target[df_oblique_target["r_baseline"].fillna(0) > 0.05].copy()
                        if not df_polar.empty:
                            if 'theta_rad' not in df_polar.columns:
                                 df_polar.loc[:, "theta_rad"] = np.deg2rad(df_polar["theta_norm"])
                            plt.figure(figsize=(8,8)); ax=plt.subplot(111,projection='polar'); 
                            ax.scatter(df_polar["theta_rad"], df_polar["r_baseline"], alpha=0.3,s=15)
                            ax.set_theta_zero_location("N"); ax.set_theta_direction(-1); 
                            rmax_val = df_polar["r_baseline"].max(); 
                            ax.set_rmax(max(0.5, rmax_val * 1.1 if pd.notnull(rmax_val) else 0.5))
                            plt.title(f"Polar Oblique on Target 'Safe' (r>0.05, {source_name}, {SCRIPT_VERSION_SUFFIX})")
                            plt.savefig(plots_output_dir / f"polar_oblique_on_target_safe_{source_name}.png"); plt.close()
                            print(f"Plot saved: polar_oblique_on_target_safe_{source_name}.png")
                        else: print(f"No data for polar plot (Oblique on Target Safe, r > 0.05) for '{source_name}'.")
                else: print(f"No valid data for 'Oblique Prompts on TargetSemantic_Safe Bases' plot for '{source_name}'.")
            else: print(f"'TargetSemantic_Safe' basis type not found in data for '{source_name}', skipping related oblique plots.")
        else: print(f"Skipping theta_baseline plots for '{source_name}' due to missing columns or all NaN data.")
        
        print(f"\n--- Analysis Script Finished for source: {source_name} ({SCRIPT_VERSION_SUFFIX}) ---")
    print(f"\n{'='*10} All Selected Artifact Sources Processed ({SCRIPT_VERSION_SUFFIX}) {'='*10}")

if __name__ == "__main__":
    main()
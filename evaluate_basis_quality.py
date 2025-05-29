# evaluate_basis_quality.py (Version 2.1.3 - P-value Corrections Integrated)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import circstd
import traceback
from tqdm import tqdm
from collections import Counter
from statsmodels.stats.multitest import multipletests # For p-value correction

try:
    from . import config 
except ImportError:
    import config 

# --- Configuration ---
R_THRESHOLD_FOR_THETA_ANALYSIS = 0.1
RELEVANT_POS_CLASS = "Class 1 Polar (Safe-aligned)"
RELEVANT_NEG_CLASS = "Class 1 Polar (Unsafe-aligned)"
CONTROL_CLASS = "Control Mixed-Form Semantic Nulls"
TARGET_THETA_POS = 0.0
TARGET_THETA_NEG = 180.0
THETA_SEPARATION_IDEAL = 180.0
NUM_PERMUTATIONS_DEFAULT = 1000 
# --- End Configuration ---

def calculate_angular_difference(angle1_deg, angle2_deg):
    if pd.isna(angle1_deg) or pd.isna(angle2_deg): return np.nan
    diff_rad = np.arctan2(np.sin(np.deg2rad(angle1_deg - angle2_deg)), np.cos(np.deg2rad(angle1_deg - angle2_deg)))
    return abs(np.rad2deg(diff_rad))

def calculate_individual_basis_quality_metrics(df_group):
    metrics = {}
    required_cols = ['input_prompt_class', 'r_baseline', 'theta_baseline_norm']
    if not all(col in df_group.columns for col in required_cols):
        expected_metrics = ['r_pos_mean', 'r_neg_mean', 'r_ctrl_mean', 'n_pos', 'n_neg', 'n_ctrl', 
                            'relevance_r_ratio', 'n_pos_theta_strong', 'n_neg_theta_strong', 
                            'median_theta_pos', 'circ_std_theta_pos', 'pos_theta_deviation_from_target',
                            'median_theta_neg', 'circ_std_theta_neg', 'neg_theta_deviation_from_target',
                            'theta_separation_abs_diff', 'theta_polarity_score']
        return pd.Series({k: np.nan for k in expected_metrics})
    r_pos_data = df_group[df_group['input_prompt_class'] == RELEVANT_POS_CLASS]['r_baseline'].dropna()
    r_neg_data = df_group[df_group['input_prompt_class'] == RELEVANT_NEG_CLASS]['r_baseline'].dropna()
    r_ctrl_data = df_group[df_group['input_prompt_class'] == CONTROL_CLASS]['r_baseline'].dropna()
    metrics['r_pos_mean'] = r_pos_data.mean() if not r_pos_data.empty else np.nan
    metrics['r_neg_mean'] = r_neg_data.mean() if not r_neg_data.empty else np.nan
    metrics['r_ctrl_mean'] = r_ctrl_data.mean() if not r_ctrl_data.empty else np.nan
    metrics['n_pos'] = len(r_pos_data); metrics['n_neg'] = len(r_neg_data); metrics['n_ctrl'] = len(r_ctrl_data)
    if pd.notna(metrics['r_pos_mean']) and pd.notna(metrics['r_neg_mean']) and \
       pd.notna(metrics['r_ctrl_mean']) and abs(metrics['r_ctrl_mean']) > config.EPSILON:
        metrics['relevance_r_ratio'] = (metrics['r_pos_mean'] + metrics['r_neg_mean']) / (2 * metrics['r_ctrl_mean'])
    else: metrics['relevance_r_ratio'] = np.nan
    theta_pos_data = df_group[(df_group['input_prompt_class'] == RELEVANT_POS_CLASS) & (df_group['r_baseline'] > R_THRESHOLD_FOR_THETA_ANALYSIS)]['theta_baseline_norm'].dropna()
    theta_neg_data = df_group[(df_group['input_prompt_class'] == RELEVANT_NEG_CLASS) & (df_group['r_baseline'] > R_THRESHOLD_FOR_THETA_ANALYSIS)]['theta_baseline_norm'].dropna()
    metrics['n_pos_theta_strong'] = len(theta_pos_data); metrics['n_neg_theta_strong'] = len(theta_neg_data)
    if not theta_pos_data.empty:
        metrics['median_theta_pos'] = np.median(theta_pos_data)
        try: metrics['circ_std_theta_pos'] = np.rad2deg(circstd(np.deg2rad(theta_pos_data))) if len(theta_pos_data) >= 2 else 0.0
        except FloatingPointError: metrics['circ_std_theta_pos'] = 0.0 if len(np.unique(theta_pos_data)) == 1 else np.nan
        metrics['pos_theta_deviation_from_target'] = calculate_angular_difference(metrics['median_theta_pos'], TARGET_THETA_POS)
    else: metrics['median_theta_pos']=np.nan; metrics['circ_std_theta_pos']=np.nan; metrics['pos_theta_deviation_from_target']=np.nan
    if not theta_neg_data.empty:
        metrics['median_theta_neg'] = np.median(theta_neg_data)
        try: metrics['circ_std_theta_neg'] = np.rad2deg(circstd(np.deg2rad(theta_neg_data))) if len(theta_neg_data) >= 2 else 0.0
        except FloatingPointError: metrics['circ_std_theta_neg'] = 0.0 if len(np.unique(theta_neg_data)) == 1 else np.nan
        metrics['neg_theta_deviation_from_target'] = calculate_angular_difference(metrics['median_theta_neg'], TARGET_THETA_NEG)
    else: metrics['median_theta_neg']=np.nan; metrics['circ_std_theta_neg']=np.nan; metrics['neg_theta_deviation_from_target']=np.nan
    if pd.notna(metrics['median_theta_pos']) and pd.notna(metrics['median_theta_neg']):
        sep_diff = calculate_angular_difference(metrics['median_theta_pos'], metrics['median_theta_neg'])
        metrics['theta_separation_abs_diff'] = sep_diff
        metrics['theta_polarity_score'] = max(0.0, sep_diff / THETA_SEPARATION_IDEAL) if THETA_SEPARATION_IDEAL > 0 else np.nan
    else: metrics['theta_separation_abs_diff'] = np.nan; metrics['theta_polarity_score'] = np.nan
    return pd.Series(metrics)

def calculate_theta_polarity_for_permutation(
    projections_df_for_exemplar: pd.DataFrame,
    shuffled_labels_map: dict,
) -> float:
    if not all(col in projections_df_for_exemplar.columns for col in ['input_prompt_text', 'theta_baseline_norm']):
        return np.nan 
    df_to_process = projections_df_for_exemplar.copy()
    df_to_process['current_shuffled_label'] = df_to_process['input_prompt_text'].map(shuffled_labels_map)
    if df_to_process['current_shuffled_label'].isnull().any(): return np.nan 
    theta_pos_data_shuffled = df_to_process[df_to_process['current_shuffled_label'] == 0]['theta_baseline_norm'].dropna()
    theta_neg_data_shuffled = df_to_process[df_to_process['current_shuffled_label'] == 1]['theta_baseline_norm'].dropna()
    median_theta_pos_shuffled = np.median(theta_pos_data_shuffled) if not theta_pos_data_shuffled.empty else np.nan
    median_theta_neg_shuffled = np.median(theta_neg_data_shuffled) if not theta_neg_data_shuffled.empty else np.nan
    if pd.notna(median_theta_pos_shuffled) and pd.notna(median_theta_neg_shuffled):
        sep_diff = calculate_angular_difference(median_theta_pos_shuffled, median_theta_neg_shuffled)
        return max(0.0, sep_diff / THETA_SEPARATION_IDEAL) if THETA_SEPARATION_IDEAL > 0 else np.nan
    return 0.0

def run_permutation_for_cluster(
    exemplar_basis_npz_filename: str,
    observed_cluster_polarity_score: float,
    df_master_projections: pd.DataFrame,
    unique_polar_prompt_texts: list[str],
    true_labels_for_polar_prompts: list[int],
    num_permutations: int = NUM_PERMUTATIONS_DEFAULT
):
    exemplar_projections_all_cols = df_master_projections[
        (df_master_projections['basis_npz_filename'] == exemplar_basis_npz_filename) &
        (df_master_projections['input_prompt_text'].isin(unique_polar_prompt_texts))
    ].copy()
    exemplar_projections_unique_prompts = exemplar_projections_all_cols.drop_duplicates(subset=['input_prompt_text'], keep='first')
    try:
        exemplar_projections_reindexed = exemplar_projections_unique_prompts.set_index('input_prompt_text').reindex(unique_polar_prompt_texts).reset_index()
    except Exception as e_reindex: print(f"Reindex failed for {exemplar_basis_npz_filename}: {e_reindex}"); return np.nan, np.nan, np.nan
    exemplar_projections_for_perm = exemplar_projections_reindexed[
        pd.notna(exemplar_projections_reindexed['r_baseline']) &
        (exemplar_projections_reindexed['r_baseline'] > R_THRESHOLD_FOR_THETA_ANALYSIS)
    ].copy()
    if exemplar_projections_for_perm.empty: return np.nan, np.nan, np.nan
    current_prompts_for_perm = exemplar_projections_for_perm['input_prompt_text'].tolist()
    if not current_prompts_for_perm: return np.nan, np.nan, np.nan
    true_label_map = {text: label for text, label in zip(unique_polar_prompt_texts, true_labels_for_polar_prompts)}
    current_true_labels_for_perm = [true_label_map[text] for text in current_prompts_for_perm if text in true_label_map]
    if len(current_true_labels_for_perm) != len(current_prompts_for_perm): return np.nan, np.nan, np.nan
    if not current_true_labels_for_perm: return np.nan, np.nan, np.nan
    null_polarity_scores = []
    shuffled_labels_array_for_perm = np.array(current_true_labels_for_perm)
    for _ in range(num_permutations):
        np.random.shuffle(shuffled_labels_array_for_perm)
        shuffled_labels_map_for_iter = {text: label for text, label in zip(current_prompts_for_perm, shuffled_labels_array_for_perm)}
        perm_score = calculate_theta_polarity_for_permutation(exemplar_projections_for_perm, shuffled_labels_map_for_iter)
        if pd.notna(perm_score): null_polarity_scores.append(perm_score)
    if not null_polarity_scores: return np.nan, np.nan, np.nan
    p_value = (np.sum(np.array(null_polarity_scores) >= observed_cluster_polarity_score) + 1.0) / (len(null_polarity_scores) + 1.0)
    return p_value, np.mean(null_polarity_scores), np.std(null_polarity_scores)

def main():
    if not hasattr(config, 'SCRIPT_VERSION_SUFFIX_FOR_ANALYSIS_FILES') or not config.SCRIPT_VERSION_SUFFIX_FOR_ANALYSIS_FILES:
        print("CRITICAL ERROR: 'SCRIPT_VERSION_SUFFIX_FOR_ANALYSIS_FILES' not defined or empty in config.py."); return

    source_name_for_paths = "default_full_set" 
    master_csv_basename = f"master_artifact_data_{source_name_for_paths}_{config.SCRIPT_VERSION_SUFFIX_FOR_ANALYSIS_FILES}.csv"
    master_csv_path = config.ANALYSIS_OUTPUT_ROOT_DIR / f"analysis_results_{source_name_for_paths}" / master_csv_basename
    output_csv_basename = f"cluster_quality_evaluation_perm_{source_name_for_paths}_{config.SCRIPT_VERSION_SUFFIX_FOR_ANALYSIS_FILES}.csv"
    output_csv_path = config.ANALYSIS_OUTPUT_ROOT_DIR / f"analysis_results_{source_name_for_paths}" / output_csv_basename
    plots_dir_basename = f"plots_{source_name_for_paths}_{config.SCRIPT_VERSION_SUFFIX_FOR_ANALYSIS_FILES}"
    quality_plots_dir = config.ANALYSIS_OUTPUT_ROOT_DIR / f"analysis_results_{source_name_for_paths}" / plots_dir_basename / "cluster_quality_analysis_perm"
    quality_plots_dir.mkdir(parents=True, exist_ok=True)

    if not master_csv_path.is_file(): print(f"ERROR: Master CSV file not found at {master_csv_path}."); return
    print(f"Loading master data from: {master_csv_path}")
    try:
        df_master = pd.read_csv(master_csv_path, low_memory=False)
        essential_cols = ['basis_npz_filename', 'input_prompt_class', 'r_baseline', 'theta_baseline', 'cluster_id', 'basis_type_from_filename', 'bmeta_gen_method_selected', 'bmeta_concept', 'input_prompt_text']
        missing_essential = [col for col in essential_cols if col not in df_master.columns]
        if missing_essential: print(f"ERROR: Master CSV missing: {missing_essential}. Aborting."); return
        for col in ['r_baseline', 'theta_baseline']: df_master[col] = pd.to_numeric(df_master[col], errors='coerce')
        df_master['cluster_id'] = pd.to_numeric(df_master['cluster_id'], errors='coerce').astype('Int64')
        if 'theta_baseline_norm' not in df_master.columns: df_master.loc[:, 'theta_baseline_norm'] = df_master['theta_baseline'] % 360
        df_master = df_master.dropna(subset=['basis_npz_filename', 'input_prompt_class', 'r_baseline', 'theta_baseline_norm', 'input_prompt_text'])
        print(f"Loaded and preprocessed {len(df_master)} records.")
    except Exception as e: print(f"ERROR loading master CSV: {e}"); traceback.print_exc(); return
    if df_master.empty: print("DataFrame empty. Exiting."); return

    print("Calculating quality metrics per individual basis file...")
    individual_basis_quality_df = df_master.groupby('basis_npz_filename', group_keys=False).apply(calculate_individual_basis_quality_metrics).reset_index()
    metadata_cols = ['basis_npz_filename', 'cluster_id', 'basis_type_from_filename', 'bmeta_gen_method_selected', 'bmeta_concept', 'bmeta_style', 'bmeta_version']
    actual_meta_cols = [col for col in metadata_cols if col in df_master.columns]
    if 'basis_npz_filename' in actual_meta_cols:
        unique_basis_meta_df = df_master[actual_meta_cols].drop_duplicates(subset=['basis_npz_filename']).reset_index(drop=True)
        individual_basis_quality_df = pd.merge(individual_basis_quality_df, unique_basis_meta_df, on='basis_npz_filename', how='left')
    print("Aggregating quality metrics to cluster level...")
    if 'cluster_id' not in individual_basis_quality_df.columns or individual_basis_quality_df['cluster_id'].isnull().all():
        print("ERROR: 'cluster_id' missing/all NaN. Cannot aggregate or run permutations."); return
    agg_func = {
        'theta_polarity_score':'median', 'relevance_r_ratio':'median', 'r_pos_mean':'median', 'r_neg_mean':'median', 'r_ctrl_mean':'median',
        'pos_theta_deviation_from_target': 'median', 'neg_theta_deviation_from_target': 'median',
        'circ_std_theta_pos': 'median', 'circ_std_theta_neg': 'median', 'theta_separation_abs_diff': 'median',
        'basis_npz_filename':'count', 'basis_type_from_filename':lambda x:x.mode()[0] if not x.mode().empty else "Unk",
        'bmeta_gen_method_selected':lambda x:x.mode()[0] if not x.mode().empty else "Unk",
        'bmeta_concept':lambda x:x.mode()[0] if not x.mode().empty else "Unk",
        'bmeta_style':lambda x:x.mode()[0] if not x.mode().empty else "Unk",
    }
    valid_agg_func = {k:v for k,v in agg_func.items() if k in individual_basis_quality_df.columns or k=='basis_npz_filename'}
    cluster_quality_df = individual_basis_quality_df.groupby('cluster_id').agg(valid_agg_func).reset_index().rename(columns={'basis_npz_filename':'num_bases_in_cluster'})

    print("\nRunning permutation tests for theta_polarity_score per cluster...")
    polar_safe_texts = df_master[df_master['input_prompt_class'] == RELEVANT_POS_CLASS]['input_prompt_text'].unique().tolist()
    polar_unsafe_texts = df_master[df_master['input_prompt_class'] == RELEVANT_NEG_CLASS]['input_prompt_text'].unique().tolist()
    unique_polar_texts_ordered = []
    temp_seen_texts = set()
    for text in polar_safe_texts:
        if text not in temp_seen_texts: unique_polar_texts_ordered.append(text); temp_seen_texts.add(text)
    for text in polar_unsafe_texts:
        if text not in temp_seen_texts: unique_polar_texts_ordered.append(text); temp_seen_texts.add(text)
    true_polar_labels_map = {text: 0 for text in polar_safe_texts}
    for text in polar_unsafe_texts: true_polar_labels_map[text] = 1
    final_unique_polar_texts = unique_polar_texts_ordered
    final_true_polar_labels = [true_polar_labels_map.get(text) for text in final_unique_polar_texts]
    valid_indices_for_labels = [i for i, label in enumerate(final_true_polar_labels) if label is not None]
    final_unique_polar_texts = [final_unique_polar_texts[i] for i in valid_indices_for_labels]
    final_true_polar_labels = [final_true_polar_labels[i] for i in valid_indices_for_labels]

    if not final_unique_polar_texts: print("ERROR: No unique Class 1 Polar prompt texts found for permutation tests.")
    else:
        print(f"Using {len(final_unique_polar_texts)} unique polar prompts ({sum(1 for x in final_true_polar_labels if x==0)} safe, {sum(1 for x in final_true_polar_labels if x==1)} unsafe) for permutation testing.")
        perm_results_all_strats = []
        for _, cluster_row in tqdm(cluster_quality_df.iterrows(), total=len(cluster_quality_df), desc="Cluster Permutations"):
            cid = cluster_row['cluster_id']; obs_pol = cluster_row['theta_polarity_score']
            if pd.isna(obs_pol): 
                for strat_name in ["A_first", "B_metric"]: perm_results_all_strats.append({'cluster_id':cid, 'exemplar_strat':strat_name, 'theta_polarity_p_value':np.nan, 'perm_null_mean':np.nan, 'perm_null_std':np.nan, 'exemplar_basis_name':"N/A_ObsNaN"})
                continue
            bases_in_cluster_slice = individual_basis_quality_df[individual_basis_quality_df['cluster_id'] == cid]
            if bases_in_cluster_slice.empty: continue
            bases_in_cluster = bases_in_cluster_slice.copy()

            exemplar_A = bases_in_cluster['basis_npz_filename'].iloc[0]
            pA, nMA, nSA = run_permutation_for_cluster(exemplar_A, obs_pol, df_master, final_unique_polar_texts, final_true_polar_labels)
            perm_results_all_strats.append({'cluster_id':cid, 'exemplar_strat':'A_first', 'theta_polarity_p_value':pA, 'perm_null_mean':nMA, 'perm_null_std':nSA, 'exemplar_basis_name':exemplar_A})
            
            exemplar_B = exemplar_A 
            if 'r_pos_mean' in bases_in_cluster.columns and 'r_neg_mean' in bases_in_cluster.columns:
                bases_in_cluster.loc[:,'_r_sum_temp'] = bases_in_cluster['r_pos_mean'].fillna(0) + bases_in_cluster['r_neg_mean'].fillna(0)
                if not bases_in_cluster['_r_sum_temp'].empty and not bases_in_cluster['_r_sum_temp'].isnull().all():
                     exemplar_B = bases_in_cluster.sort_values('_r_sum_temp', ascending=False)['basis_npz_filename'].iloc[0]
                if '_r_sum_temp' in bases_in_cluster.columns: bases_in_cluster = bases_in_cluster.drop(columns=['_r_sum_temp']) 
            
            pB, nMB, nSB = run_permutation_for_cluster(exemplar_B, obs_pol, df_master, final_unique_polar_texts, final_true_polar_labels)
            perm_results_all_strats.append({'cluster_id':cid, 'exemplar_strat':'B_metric', 'theta_polarity_p_value':pB, 'perm_null_mean':nMB, 'perm_null_std':nSB, 'exemplar_basis_name':exemplar_B})

        if perm_results_all_strats:
            perm_df_long = pd.DataFrame(perm_results_all_strats)
            perm_df_pivot = perm_df_long.pivot_table(index='cluster_id', columns='exemplar_strat', values=['theta_polarity_p_value', 'perm_null_mean', 'perm_null_std', 'exemplar_basis_name'], aggfunc='first')
            perm_df_pivot.columns = [f'{val}_{strat}' for val, strat in perm_df_pivot.columns]
            cluster_quality_df = pd.merge(cluster_quality_df, perm_df_pivot.reset_index(), on='cluster_id', how='left')
            print("\nPermutation test results added.")

            alpha = 0.05
            for strat_suffix in ["A_first", "B_metric"]:
                pval_col_name = f'theta_polarity_p_value_{strat_suffix}'
                if pval_col_name in cluster_quality_df.columns:
                    raw_pvals = cluster_quality_df[pval_col_name].dropna()
                    if not raw_pvals.empty:
                        reject_bonf, pvals_bonf, _, _ = multipletests(raw_pvals, alpha=alpha, method='bonferroni')
                        cluster_quality_df.loc[raw_pvals.index, f'pval_{strat_suffix}_bonferroni'] = pvals_bonf
                        cluster_quality_df.loc[raw_pvals.index, f'reject_{strat_suffix}_bonferroni'] = reject_bonf
                        reject_fdr, pvals_fdr, _, _ = multipletests(raw_pvals, alpha=alpha, method='fdr_bh')
                        cluster_quality_df.loc[raw_pvals.index, f'pval_{strat_suffix}_fdr_bh'] = pvals_fdr
                        cluster_quality_df.loc[raw_pvals.index, f'reject_{strat_suffix}_fdr_bh'] = reject_fdr
                        print(f"P-value corrections applied for strategy '{strat_suffix}'. Sig Bonf: {np.sum(reject_bonf)}, Sig FDR: {np.sum(reject_fdr)}")
                    else: print(f"No non-NaN p-values for '{strat_suffix}'.")

    cluster_quality_df['composite_score_cluster'] = cluster_quality_df['theta_polarity_score'].fillna(0) * cluster_quality_df['relevance_r_ratio'].fillna(0) * ((cluster_quality_df['r_pos_mean'].fillna(0) + cluster_quality_df['r_neg_mean'].fillna(0)) / 2)
    sort_by_cols = ['composite_score_cluster']
    ascending_orders = [False]
    pval_sort_col = 'pval_A_first_fdr_bh' # Example, choose your preferred p-value for sorting
    if pval_sort_col in cluster_quality_df.columns:
        sort_by_cols.insert(0, pval_sort_col) # Sort by p-value first
        ascending_orders.insert(0, True)
    if 'theta_polarity_score' in cluster_quality_df.columns:
         sort_by_cols.append('theta_polarity_score') # Then by polarity score
         ascending_orders.append(False)
    cluster_quality_df = cluster_quality_df.sort_values(by=sort_by_cols, ascending=ascending_orders).reset_index(drop=True)
    
    print(f"\nTop performing GEOMETRIC CLUSTERS (sorted by p-value then composite score):")
    cols_to_show_base = ['cluster_id', 'num_bases_in_cluster', 'bmeta_concept', 'theta_polarity_score']
    cols_perm_A = [f'theta_polarity_p_value_A_first', f'pval_A_first_fdr_bh', f'reject_A_first_fdr_bh', f'exemplar_basis_name_A_first']
    cols_perm_B = [f'theta_polarity_p_value_B_metric', f'pval_B_metric_fdr_bh', f'reject_B_metric_fdr_bh', f'exemplar_basis_name_B_metric']
    cols_other_metrics = ['relevance_r_ratio', 'r_pos_mean', 'r_neg_mean', 'composite_score_cluster']
    cols_to_show_final = cols_to_show_base + cols_perm_A + cols_perm_B + cols_other_metrics
    cols_to_show_final_unique = []
    for col in cols_to_show_final:
        if col not in cols_to_show_final_unique and col in cluster_quality_df.columns: cols_to_show_final_unique.append(col)
    print(cluster_quality_df[cols_to_show_final_unique].head(30).to_string(max_colwidth=30, float_format="%.4f"))

    try:
        cluster_quality_df.to_csv(output_csv_path, index=False, float_format='%.4f')
        print(f"\nCluster quality evaluation (with p-values) saved to: {output_csv_path}")
    except Exception as e: print(f"ERROR saving final cluster quality CSV: {e}")

    plot_cols = ['theta_polarity_score', 'relevance_r_ratio', 'r_pos_mean', 'bmeta_gen_method_selected', 'bmeta_concept', 'num_bases_in_cluster']
    if all(col in cluster_quality_df.columns for col in plot_cols):
        plot_df = cluster_quality_df.dropna(subset=[c for c in plot_cols if c != 'num_bases_in_cluster'])
        if not plot_df.empty:
            plt.figure(figsize=(14, 10)); avg_r = (plot_df['r_pos_mean'].fillna(0) + plot_df['r_neg_mean'].fillna(0)) / 2
            size_m = pd.to_numeric(plot_df['num_bases_in_cluster'], errors='coerce').fillna(1).clip(1)
            
            # --- MODIFIED: Add p-value to style/hue if desired ---
            hue_col = 'bmeta_gen_method_selected'
            style_col = 'bmeta_concept'
            # Example: if you want to highlight significant clusters in the plot
            # This assumes 'reject_A_first_fdr_bh' exists and is boolean
            # if 'reject_A_first_fdr_bh' in plot_df.columns:
            #     plot_df.loc[:, 'Significant (FDR A_first < 0.05)'] = plot_df['reject_A_first_fdr_bh'].fillna(False)
            #     style_col = 'Significant (FDR A_first < 0.05)' # Change style to show significance

            sns.scatterplot(data=plot_df, x='relevance_r_ratio', y='theta_polarity_score', 
                            hue=hue_col, style=style_col, 
                            size=size_m, sizes=(40,500), # Increased min size slightly
                            alpha=0.8, legend='auto')
            plt.title(f'Cluster Quality: Polarity vs. Relevance ({source_name_for_paths})', fontsize=16)
            plt.xlabel('Median Relevance Ratio (per cluster)', fontsize=12); plt.ylabel('Median Theta Polarity Score (per cluster)', fontsize=12)
            plt.axhline(0.8,c='grey',ls=':',alpha=0.7, label='Polarity > 0.8'); plt.axvline(1.5,c='grey',ls=':',alpha=0.7, label='Relevance Ratio > 1.5')
            plt.legend(title='Dom. Method/Concept (Size ~ #Bases)', bbox_to_anchor=(1.02,1), loc='upper left', borderaxespad=0.)
            plt.grid(True,alpha=0.3); plt.tight_layout(rect=[0,0,0.80,0.96])
            plot_sp = quality_plots_dir / f"cluster_quality_scatter_{source_name_for_paths}.png"
            plt.savefig(plot_sp, dpi=150); print(f"Cluster quality scatter plot saved: {plot_sp}"); plt.close()
        else: print("No data for cluster scatter plot.")
    else: print(f"Skipping cluster scatter: missing one of {plot_cols}")

if __name__ == "__main__":
    main()
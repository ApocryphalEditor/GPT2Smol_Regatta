# artifact_formatter.py
# --- VERSION 5.2: Displaying both PeakNs, NumPy type handling, Corrected Basis Check ---

import random
import datetime
import uuid
from pathlib import Path
import numpy as np # Ensure NumPy is imported
from config import ANALYSIS_LAYER

# Removed direct import of basis_u1

def generate_artifact_id() -> str:
    ts = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    short_uuid = str(uuid.uuid4())[:4]
    return f"ART-{ts}-{short_uuid}"

# --- Helper Function for Formatting Projection ---
def format_projection(proj_data: dict | None, label: str, display_lines: list, tags: list, basis_loaded: bool):
    """Formats projection data (baseline or intervened)."""
    safe_label_tag = label.upper().replace(" ", "_").replace("(", "").replace(")", "").replace(":", "")
    
    if not basis_loaded: 
        display_lines.append(f"{label} (L{ANALYSIS_LAYER}): N/A (Basis Not Loaded)")
        tags.append(f"{safe_label_tag}_PROJ_NO_BASIS"); return
        
    if proj_data is not None:
        angle = proj_data.get('angle_deg', 'N/A')
        r = proj_data.get('r', 'N/A')
        stability = "Nominal"; tag_suffix = "_PROJ_NOMINAL"
        if isinstance(r, (float, int, np.floating, np.integer)): 
            r_float = float(r) 
            if r_float > 0.7: stability = "Strong Alignment"; tag_suffix = "_PROJ_STRONG"
            elif r_float < 0.15: stability = "Weak Alignment"; tag_suffix = "_PROJ_WEAK"
            r_str = f"{r_float:.4f}"
        else: r_str = "N/A" 
            
        if isinstance(angle, (float, int, np.floating, np.integer)): 
            angle_float = float(angle)
            angle_str = f"{angle_float:.1f}\u00b0"
        else: angle_str = "N/A" 
        
        display_lines.append(f"{label} (L{ANALYSIS_LAYER}): \u03b8={angle_str} | r={r_str} | Status: {stability}")
        tags.append(safe_label_tag + tag_suffix)
    else:
        display_lines.append(f"{label} (L{ANALYSIS_LAYER}): Calculation Failed or N/A")
        tags.append(safe_label_tag + "_PROJ_FAIL")

# --- Helper Function for Formatting Alignment ---
def format_alignment(align_data: dict | None, label: str, display_lines: list, tags: list, basis_loaded: bool):
    """Formats final state alignment data (baseline or intervened)."""
    safe_label_tag = label.upper().replace(" ", "_").replace("(", "").replace(")", "").replace(":", "")

    if not basis_loaded:
         display_lines.append(f"{label} (L{ANALYSIS_LAYER}): N/A (Basis Not Loaded)")
         tags.append(f"{safe_label_tag}_ALIGN_NO_BASIS"); return

    if align_data is not None:
        sim1 = align_data.get('sim_basis1', 'N/A')
        sim2 = align_data.get('sim_basis2', 'N/A')

        sim1_is_num = isinstance(sim1, (float,int, np.floating, np.integer))
        sim2_is_num = isinstance(sim2, (float,int, np.floating, np.integer))
        
        sim1_float = float(sim1) if sim1_is_num else None
        sim2_float = float(sim2) if sim2_is_num else None

        sim1_str = f"{sim1_float:.4f}" if sim1_is_num else 'N/A'
        sim2_str = f"{sim2_float:.4f}" if sim2_is_num else 'N/A'
            
        display_lines.append(f"{label} (L{ANALYSIS_LAYER}): Sim(B1)={sim1_str} | Sim(B2)={sim2_str}")
        
        tag_suffix = "_ALIGN_UNKNOWN"
        if sim1_is_num and sim2_is_num: 
            abs_sim1 = abs(sim1_float); abs_sim2 = abs(sim2_float)
            if abs_sim1 > abs_sim2 * 1.5 and abs_sim1 > 0.3: tag_suffix = "_ALIGN_B1_DOM"
            elif abs_sim2 > abs_sim1 * 1.5 and abs_sim2 > 0.3: tag_suffix = "_ALIGN_B2_DOM"
            elif abs_sim1 < 0.1 and abs_sim2 < 0.1: tag_suffix = "_ALIGN_LOW"
            else: tag_suffix = "_ALIGN_MIXED"
        tags.append(safe_label_tag + tag_suffix)
    else:
        display_lines.append(f"{label} (L{ANALYSIS_LAYER}): Calculation Failed or N/A")
        tags.append(safe_label_tag + "_ALIGN_FAIL")

# --- Helper Function for Formatting Neuro Trace ---
def format_neuro_trace(stats_dict: dict | None, label: str, display_lines: list, tags: list):
    """Formats activation stats summary (baseline or intervened)."""
    safe_label_tag = label.upper().replace(" ", "_").replace("(", "").replace(")", "").replace(":", "")
    tag_prefix = f"{safe_label_tag}" 

    if not stats_dict:
        display_lines.append(f"{label}: N/A")
        tags.append(f"{tag_prefix}_STATS_MISSING") 
        return

    summary_lines = []
    layer_keys = list(stats_dict.keys())
    
    sorted_layer_indices_int = [] 
    non_int_keys = []
    
    for k in layer_keys:
        try:
            sorted_layer_indices_int.append(int(k)) 
        except (ValueError, TypeError):
            non_int_keys.append(k) 
            
    sorted_layer_indices_int.sort() 
        
    found_valid_stats = False 
    
    for layer_idx_int in sorted_layer_indices_int:
        stats = stats_dict.get(layer_idx_int) 
        
        if not stats or not isinstance(stats, dict): 
            continue 

        if "error" in stats:
            summary_lines.append(f"  L{layer_idx_int}: Error ({stats['error']})") 
            continue 

        found_valid_stats = True 

        mean_all = stats.get('mean_activation_all_tokens', 'N/A')
        max_neu_overall = stats.get('max_activating_neuron_idx', 'N/A') 
        max_neu_last_tkn = stats.get('peak_neuron_last_token_idx', 'N/A') 
        last_stats = stats.get("last_token_stats", {})
        last_mean = last_stats.get('mean', 'N/A')
        last_median = last_stats.get('median', 'N/A')

        mean_all_str = f"{mean_all:.3f}" if isinstance(mean_all, (float,int,np.floating, np.integer)) else 'N/A'
        last_mean_str = f"{last_mean:.3f}" if isinstance(last_mean, (float,int,np.floating, np.integer)) else 'N/A'
        last_median_str = f"{last_median:.3f}" if isinstance(last_median, (float,int,np.floating, np.integer)) else 'N/A'
        
        peak_overall_str = f"PeakN(All)={max_neu_overall}" if max_neu_overall != 'N/A' and max_neu_overall != -1 else "PeakN(All)=N/A"
        peak_last_tkn_str = f"PeakN(Last)={max_neu_last_tkn}" if max_neu_last_tkn != 'N/A' and max_neu_last_tkn != -1 else "PeakN(Last)=N/A"
        
        summary_lines.append(f"  L{layer_idx_int}: Mean={mean_all_str}, LastMean={last_mean_str}, LastMed={last_median_str}, {peak_overall_str}, {peak_last_tkn_str}")

        if isinstance(last_mean, (float, int, np.floating, np.integer)):
            last_mean_f = float(last_mean) 
            if last_mean_f < -0.1: tags.append(f"{tag_prefix}_L{layer_idx_int}_SUPPRESSED")
            if last_mean_f > 0.5: tags.append(f"{tag_prefix}_L{layer_idx_int}_ACTIVE")
        if isinstance(mean_all, (float, int, np.floating, np.integer)):
             mean_all_f = float(mean_all)
             if mean_all_f > 0.8: tags.append(f"{tag_prefix}_L{layer_idx_int}_OVERALL_HIGH")

    for layer_idx_other in non_int_keys:
         stats = stats_dict.get(layer_idx_other)
         if isinstance(stats, str): 
             summary_lines.append(f"  {layer_idx_other}: {stats}")

    if summary_lines:
        display_lines.append(f"{label}:")
        display_lines.extend(summary_lines)
        if found_valid_stats:
             tags.append(f"{tag_prefix}_STATS_CAPTURED")
        else:
             tags.append(f"{tag_prefix}_CAPTURE_NO_VALID_STATS") 
    else: 
        display_lines.append(f"{label}: No processable layer stats found.") 
        tags.append(f"{tag_prefix}_CAPTURE_EMPTY_LAYERS")

# --- Main Formatting Function ---
def format_artifact(raw_results: dict) -> dict:
    """Translates raw experiment results into a displayable Artifact object."""

    display_lines = []
    tags = []
    artifact_id = raw_results.get("artifact_id", generate_artifact_id())

    if raw_results.get("error"):
        display_lines.append(f"[{artifact_id}] Processing Error: {raw_results['error']}")
        tags.append("ERROR")
        return {
            "artifact_id": artifact_id, "display_text": display_lines, "tags": tags,
            "timestamp": raw_results.get("run_timestamp", datetime.datetime.now().isoformat()), 
            "source_prompt": raw_results.get("input_prompt", "N/A"),
            "raw_data": raw_results
        }

    basis_file = raw_results.get("basis_file_used")
    basis_error = raw_results.get("basis_load_error")
    basis_loaded_successfully = bool(basis_file) and (basis_error is None or basis_error == "")

    if basis_file:
        try: display_lines.append(f"Basis: {Path(basis_file).name}")
        except Exception: display_lines.append(f"Basis: {basis_file} (Path Error)")
        if basis_error: 
             display_lines.append(f"Basis Status: Error during load ({basis_error})")
             tags.append("BASIS_ERROR")
    elif basis_error: 
        display_lines.append(f"Basis Error: {basis_error}")
        tags.append("BASIS_ERROR")
    else: 
        display_lines.append("Basis: Not Loaded")
        tags.append("BASIS_MISSING")

    display_lines.append("--- Baseline State ---")
    format_projection(raw_results.get("prompt_vector_projection_baseline"), "Prompt Projection (Baseline)", display_lines, tags, basis_loaded_successfully)
    format_alignment(raw_results.get("final_state_alignment_baseline"), "Final State Align (Baseline)", display_lines, tags, basis_loaded_successfully)
    format_neuro_trace(raw_results.get("activation_stats_baseline"), "Neuro-Trace Summary (Baseline)", display_lines, tags)

    interventions_active = raw_results.get("interventions_enabled_during_run", False)
    intervened_stats = raw_results.get("activation_stats_intervened") 

    if interventions_active:
        tags.append("INTERVENTION_ACTIVE")
        display_lines.append("--- Intervened State ---")
        if intervened_stats is not None: 
            format_projection(raw_results.get("prompt_vector_projection_intervened"), "Prompt Projection (Intervened)", display_lines, tags, basis_loaded_successfully)
            format_alignment(raw_results.get("final_state_alignment_intervened"), "Final State Align (Intervened)", display_lines, tags, basis_loaded_successfully)
            format_neuro_trace(intervened_stats, "Neuro-Trace Summary (Intervened)", display_lines, tags)
        else: 
            tags.append("INTERVENED_CAPTURE_FAILED")
            format_projection(None, "Prompt Projection (Intervened)", display_lines, tags, basis_loaded_successfully)
            format_alignment(None, "Final State Align (Intervened)", display_lines, tags, basis_loaded_successfully)
            format_neuro_trace(None, "Neuro-Trace Summary (Intervened)", display_lines, tags)

    collapse = raw_results.get("collapse_detected")
    if collapse is not None:
        status = "DETECTED - Signal Compromised" if collapse else "Nominal"
        display_lines.append(f"Integrity Check: Signal Collapse {status}")
        if collapse: tags.append("COLLAPSE_DETECTED")
        else: tags.append("INTEGRITY_NOMINAL")
    else:
        display_lines.append("Integrity Check: Unknown")
        tags.append("COLLAPSE_UNKNOWN")

    generated_text = raw_results.get("generated_text", "")
    if generated_text and not generated_text.startswith("[Generation Error"):
        display_lines.append(f"Generated Output: '{generated_text}'")
        tags.append("GENERATION_SUCCESS")
    elif generated_text: 
        display_lines.append(generated_text)
        tags.append("GENERATION_ERROR")
    else: 
        display_lines.append("Generated Output: [Empty]")
        tags.append("GENERATION_EMPTY")

    artifact_data = {
        "artifact_id": artifact_id,
        "display_text": display_lines, 
        "tags": sorted(list(set(tags))), 
        "timestamp": raw_results.get("run_timestamp", datetime.datetime.now().isoformat()),
        "source_prompt": raw_results.get("input_prompt", "N/A"),
        "raw_data": raw_results 
    }
    return artifact_data
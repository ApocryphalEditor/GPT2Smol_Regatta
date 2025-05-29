# create_pca_bases_from_file.py (Version with CSV Logging)
import time
from pathlib import Path
from tqdm import tqdm
import argparse
import re 
import csv # For logging
import uuid # For unique run ID
import datetime # For log timestamp
import json # For potentially logging list of prompts

# Assuming config.py and game_controller.py are in the same directory or accessible
try:
    from . import config
    from .game_controller import GameController
    from .config import BasisGenerationMethods 
except ImportError: # Fallback for running as script
    import config
    from game_controller import GameController
    from config import BasisGenerationMethods

# --- Configuration for this script ---
PCA_PROMPT_SETS_DIR = Path("./promptsets") 
BASIS_SAVE_DIRECTORY = config.DEFAULT_BASIS_SEARCH_DIR
LOG_DIR = Path("./generation_logs") # Directory for saving generation logs
# --- End Configuration ---

# --- Parser Keywords ---
TARGET_NPZ_FILENAME_KEY = "TARGET_NPZ_FILENAME" 
BASIS_CONCEPT_KEY = "BASIS_CONCEPT"
BASIS_STYLE_KEY = "BASIS_STYLE"
BASIS_VERSION_KEY = "BASIS_VERSION"
TARGET_LAYER_KEY = "TARGET_LAYER"
POLE_A_NAME_KEY = "POLE_A_NAME"
POLE_B_NAME_KEY = "POLE_B_NAME"
PROMPT_A_START_KEY = "PROMPT_A_START"
PROMPT_A_END_KEY = "PROMPT_A_END"
PROMPT_B_START_KEY = "PROMPT_B_START"
PROMPT_B_END_KEY = "PROMPT_B_END"
END_BASIS_DEF_KEY = "---END_BASIS_DEF---"
# --- End Parser Keywords ---

# --- LOG CSV Header ---
LOG_CSV_HEADER_COLUMNS = [
    "log_timestamp", "generation_run_id", "definition_file_used",
    "target_npz_filename_from_source", "final_npz_filename_generated",
    "generation_method_selected", "layer_idx", 
    "num_prompts_A_in_def", "num_prompts_B_in_def", 
    "example_prompt_A", "example_prompt_B", 
    "status", "failure_point", "message_from_controller",
    "final_npz_filepath_generated", "metadata_json_path_generated",
    # Debug metrics from controller
    "dbg_vec_A_norm_initial", "dbg_vec_B_norm_initial", 
    "dbg_vec_A_B_cosine_sim_initial",
    "dbg_diff_vec_norm_unnormalized", "dbg_mean_vec_norm_unnormalized",
    "dbg_u1_norm_final", "dbg_u2_norm_final", "dbg_u1_u2_dot_product"
]
# --- End LOG CSV Header ---

def parse_pca_definition_file(filepath: Path) -> list[dict]:
    # (This function remains the same as the last version you approved for it)
    all_basis_definitions = []
    current_definition = {}
    current_prompts_A = []
    current_prompts_B = []
    in_prompt_a_block = False
    in_prompt_b_block = False
    line_number = 0
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line_content in f:
                line_number += 1
                line = line_content.strip()
                if not line or line.startswith("#"): continue
                if line.upper() == END_BASIS_DEF_KEY:
                    if current_definition.get(TARGET_NPZ_FILENAME_KEY) and current_prompts_A and current_prompts_B:
                        current_definition["prompts_A_list"] = list(current_prompts_A)
                        current_definition["prompts_B_list"] = list(current_prompts_B)
                        all_basis_definitions.append(dict(current_definition))
                    elif current_definition:
                        print(f"Warning (Line ~{line_number}): Incomplete basis definition block ending with '{END_BASIS_DEF_KEY}'. Discarded: {current_definition.keys()}")
                    current_definition = {}; current_prompts_A = []; current_prompts_B = []
                    in_prompt_a_block = False; in_prompt_b_block = False
                    continue
                if in_prompt_a_block:
                    if line.upper() == PROMPT_A_END_KEY: in_prompt_a_block = False
                    else: current_prompts_A.append(line)
                    continue
                if in_prompt_b_block:
                    if line.upper() == PROMPT_B_END_KEY: in_prompt_b_block = False
                    else: current_prompts_B.append(line)
                    continue
                if line.upper() == PROMPT_A_START_KEY:
                    in_prompt_a_block = True; current_prompts_A = []
                    continue
                if line.upper() == PROMPT_B_START_KEY:
                    in_prompt_b_block = True; current_prompts_B = []
                    continue
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip().upper().replace(" ", "_") 
                    value = value.strip()
                    if key == TARGET_LAYER_KEY:
                        try: current_definition[key] = int(value)
                        except ValueError: print(f"Warning (L{line_number}): Invalid int for {TARGET_LAYER_KEY}: '{value}'. Using default."); current_definition[key] = config.ANALYSIS_LAYER
                    else: current_definition[key] = value
                elif line.strip(): print(f"Warning (L{line_number}): Unrecognized line: '{line[:100]}...'")
        if current_definition.get(TARGET_NPZ_FILENAME_KEY) and current_prompts_A and current_prompts_B:
            current_definition["prompts_A_list"] = list(current_prompts_A)
            current_definition["prompts_B_list"] = list(current_prompts_B)
            all_basis_definitions.append(dict(current_definition))
    except FileNotFoundError: print(f"ERROR: File not found {filepath}"); return []
    except Exception as e: print(f"ERROR parsing file {filepath}: {e}"); return []
    if all_basis_definitions: print(f"Parsed {len(all_basis_definitions)} PCA basis definitions from {filepath.name}")
    return all_basis_definitions

def select_pca_definition_file() -> Path | None:
    # (This function remains the same)
    PCA_PROMPT_SETS_DIR.mkdir(parents=True, exist_ok=True)
    text_files = sorted([f for f in PCA_PROMPT_SETS_DIR.glob("*.txt") if f.is_file()])
    if not text_files:
        print(f"No .txt files found in '{PCA_PROMPT_SETS_DIR.resolve()}'. Please create PCA definition file."); return None
    print(f"\nAvailable PCA definition files in '{PCA_PROMPT_SETS_DIR.name}':")
    for i, file_path in enumerate(text_files): print(f"  {i+1}: {file_path.name}")
    while True:
        try:
            choice_str = input(f"Enter file number (1-{len(text_files)}), or 'q' to quit: ").strip()
            if choice_str.lower() == 'q': print("Selection cancelled."); return None
            choice_idx = int(choice_str) - 1
            if 0 <= choice_idx < len(text_files): return text_files[choice_idx]
            else: print("Invalid choice.")
        except ValueError: print("Invalid input.")
        except KeyboardInterrupt: print("\nSelection cancelled."); return None

def main():
    print("--- PCA Basis Generation Script (v_filename_prefix, v_csv_log) ---")

    # --- Log File Setup ---
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    generation_run_id = str(uuid.uuid4())
    log_filename = LOG_DIR / f"pca_basis_generation_log_{generation_run_id[:8]}.csv"
    # --- End Log File Setup ---

    parser = argparse.ArgumentParser(description="Generates PCA-based basis vectors from a structured text definition file.")
    parser.add_argument("--inputfile", type=str, default=None, help="Path to PCA basis definition file. Interactive if not provided.")
    args = parser.parse_args()

    selected_filepath = None
    if args.inputfile:
        candidate_path = Path(args.inputfile)
        if candidate_path.is_file(): selected_filepath = candidate_path
        else:
            candidate_in_subdir = PCA_PROMPT_SETS_DIR / candidate_path.name
            if candidate_in_subdir.is_file(): selected_filepath = candidate_in_subdir
            else: print(f"ERROR: Specified input file '{args.inputfile}' not found directly or in '{PCA_PROMPT_SETS_DIR}'.")
    
    if not selected_filepath:
        print("No input file specified via --inputfile, or file not found.")
        selected_filepath = select_pca_definition_file()

    if not (selected_filepath and selected_filepath.is_file()):
        print("No valid PCA definition file selected or found. Exiting."); return

    print(f"\nProcessing PCA definition file: {selected_filepath.resolve()}")
    print(f"Generation log will be saved to: {log_filename.resolve()}")
    pca_basis_definitions = parse_pca_definition_file(selected_filepath)

    if not pca_basis_definitions:
        print("No valid PCA basis definitions parsed. Exiting."); return
    print(f"\nFound {len(pca_basis_definitions)} PCA basis definitions to generate.")

    try:
        print("Initializing GameController..."); controller = GameController(load_existing_artifacts=False) # Don't load artifacts for this script
        if config.DIMENSION <= 0: # Ensure dimension is set
             if hasattr(controller, 'experiment_runner') and hasattr(controller.experiment_runner, 'model') and controller.experiment_runner.model is not None:
                 config.DIMENSION = controller.experiment_runner.model.cfg.d_mlp
             elif experiment_runner.model is not None : # If controller didn't trigger it, but runner has it
                  config.DIMENSION = experiment_runner.model.cfg.d_mlp
             else: # Try to load if still not set
                 print("Model not loaded by controller, trying explicit load for DIMENSION..."); import experiment_runner 
                 experiment_runner.load_model_and_tokenizer() # this sets config.DIMENSION
             if config.DIMENSION <=0: print("CRITICAL: Model dimension not set after attempts."); return
        print(f"GameController ready. Model Dim: {config.DIMENSION}")
    except Exception as e: print(f"CRITICAL: GameController init failed: {e}"); return

    generated_count = 0; failed_count = 0
    method_prefix_for_pca = "PCA" 

    with open(log_filename, 'w', newline='', encoding='utf-8') as log_file:
        log_writer = csv.DictWriter(log_file, fieldnames=LOG_CSV_HEADER_COLUMNS)
        log_writer.writeheader()

        for definition_dict in tqdm(pca_basis_definitions, desc="Generating PCA Bases", unit="basis"):
            log_row_data = {k: None for k in LOG_CSV_HEADER_COLUMNS} # Initialize with None
            log_row_data["log_timestamp"] = datetime.datetime.now().isoformat()
            log_row_data["generation_run_id"] = generation_run_id
            log_row_data["definition_file_used"] = str(selected_filepath.name) # Just filename for brevity
            log_row_data["generation_method_selected"] = BasisGenerationMethods.PCA_A_B_CLUSTER

            original_target_npz = definition_dict.get(TARGET_NPZ_FILENAME_KEY)
            prompts_A = definition_dict.get("prompts_A_list")
            prompts_B = definition_dict.get("prompts_B_list")
            
            log_row_data["target_npz_filename_from_source"] = original_target_npz
            log_row_data["num_prompts_A_in_def"] = len(prompts_A) if prompts_A else 0
            log_row_data["num_prompts_B_in_def"] = len(prompts_B) if prompts_B else 0
            log_row_data["example_prompt_A"] = prompts_A[0][:100] + '...' if prompts_A else None # Log first as example
            log_row_data["example_prompt_B"] = prompts_B[0][:100] + '...' if prompts_B else None


            if not original_target_npz or not prompts_A or not prompts_B:
                tqdm.write(f"Warning: Skipping definition due to missing filename or prompts: {original_target_npz or 'N/A_Filename'}")
                log_row_data["status"] = "FAILURE"
                log_row_data["failure_point"] = "initial_parse_incomplete"
                log_row_data["message_from_controller"] = "Missing target_npz, prompts_A, or prompts_B in definition block."
                log_writer.writerow(log_row_data)
                failed_count +=1; continue

            path_obj = Path(original_target_npz)
            final_npz_filename_str = ""
            if path_obj.stem.upper().startswith(method_prefix_for_pca + "_"):
                final_npz_filename_str = original_target_npz
            else:
                final_npz_filename_str = f"{method_prefix_for_pca}_{path_obj.stem}{path_obj.suffix}"
            full_npz_save_path = BASIS_SAVE_DIRECTORY / final_npz_filename_str
            log_row_data["final_npz_filename_generated"] = final_npz_filename_str

            layer_idx = definition_dict.get(TARGET_LAYER_KEY, config.ANALYSIS_LAYER)
            concept = definition_dict.get(BASIS_CONCEPT_KEY, f"{Path(final_npz_filename_str).stem.split('_')[1] if '_' in Path(final_npz_filename_str).stem else 'PCAConcept'}")
            style = definition_dict.get(BASIS_STYLE_KEY, "MultiPromptPCA")
            version = definition_dict.get(BASIS_VERSION_KEY, "v1") 
            log_row_data["layer_idx"] = layer_idx
            
            tqdm.write(f"\nGenerating PCA Basis: {final_npz_filename_str}")
            tqdm.write(f"  (Original source filename: {original_target_npz})")

            generation_result = controller.generate_basis_vectors_and_save(
                method=BasisGenerationMethods.PCA_A_B_CLUSTER,
                prompts_A=prompts_A,
                prompts_B=prompts_B,
                layer_idx=layer_idx,
                save_filepath_str=str(full_npz_save_path),
                basis_concept=concept,
                basis_style=style,
                basis_version=version
            )
            
            log_row_data["status"] = "SUCCESS" if generation_result["success"] else "FAILURE"
            log_row_data["failure_point"] = generation_result.get("failure_point")
            log_row_data["message_from_controller"] = generation_result.get("message")
            log_row_data["final_npz_filepath_generated"] = generation_result.get("final_npz_filepath")
            log_row_data["metadata_json_path_generated"] = generation_result.get("metadata_json_filepath")

            debug_metrics = generation_result.get("debug_metrics", {})
            for key, value in debug_metrics.items():
                log_key = f"dbg_{key}"
                if log_key in LOG_CSV_HEADER_COLUMNS:
                    log_row_data[log_key] = value
            
            log_writer.writerow(log_row_data)

            if generation_result["success"]:
                tqdm.write(f"  SUCCESS: {generation_result['message']}")
                generated_count += 1
            else:
                tqdm.write(f"  FAILURE: {generation_result['message']}")
                failed_count += 1

    print(f"\n--- PCA Basis Generation Complete ---")
    print(f"Successfully generated {generated_count} PCA bases.")
    if failed_count > 0: print(f"Failed or skipped {failed_count} PCA bases.")
    print(f"Basis files saved in: {BASIS_SAVE_DIRECTORY.resolve()}")
    print(f"Companion metadata JSONs saved in: {(BASIS_SAVE_DIRECTORY / 'basesdata').resolve()}")
    print(f"Detailed generation log saved to: {log_filename.resolve()}")


if __name__ == "__main__":
    main()
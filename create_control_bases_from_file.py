# create_control_bases_from_file.py (Version with method-prefixed filenames, CSV Logging, and valid_methods fix)
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
PROMPT_SETS_DIR = Path("./promptsets") 
BASIS_SAVE_DIRECTORY = config.DEFAULT_BASIS_SEARCH_DIR
TARGET_LAYER_FOR_BASES = config.ANALYSIS_LAYER
LOG_DIR = Path("./generation_logs") 
# --- End Configuration ---

# --- LOG CSV Header ---
LOG_CSV_HEADER_COLUMNS = [
    "log_timestamp", "generation_run_id", "definition_file_used",
    "target_npz_filename_from_source", "final_npz_filename_generated",
    "generation_method_selected", "layer_idx", 
    "num_prompts_A_in_def", "num_prompts_B_in_def", 
    "example_prompt_A", "example_prompt_B", 
    "status", "failure_point", "message_from_controller",
    "final_npz_filepath_generated", "metadata_json_path_generated",
    "dbg_vec_A_norm_initial", "dbg_vec_B_norm_initial", 
    "dbg_vec_A_B_cosine_sim_initial",
    "dbg_diff_vec_norm_unnormalized", "dbg_mean_vec_norm_unnormalized",
    "dbg_u1_norm_final", "dbg_u2_norm_final", "dbg_u1_u2_dot_product"
]
# --- End LOG CSV Header ---


def get_method_short_code(method_name: str) -> str:
    if not method_name: return "UNKWN" # Handle empty method name string
    if method_name == BasisGenerationMethods.MEAN_A_VS_B_GRAM_SCHMIDT:
        return "MAB" 
    elif method_name == BasisGenerationMethods.DIFF_AB_RAND_ORTHO_U2:
        return "DRND"
    elif method_name == BasisGenerationMethods.DIFF_AB_MEAN_AB_ORTHO_U2:
        return "DMEAN"
    elif method_name == BasisGenerationMethods.PCA_A_B_CLUSTER: 
        return "PCA" 
    else:
        parts = method_name.split('_')
        if parts: return "".join(p[0] for p in parts if p).upper()[:5] # Ensure p is not empty
        return "CUSTOM"

def parse_target_npz_filename_for_metadata(npz_filename_str: str) -> dict:
    # (This function remains the same as the last version you approved for it)
    stem = Path(npz_filename_str).stem 
    parts = stem.split('_')
    meta = {"concept": "UnknownFileParse", "style": "UnknownFileParse", "version": "UnknownFileParse"}
    if len(parts) >= 3: 
        if parts[0] in ["Safe", "Unrelated", "CtrlPolar"]:
            if len(parts) >= 4: 
                meta["concept"] = parts[0]; meta["style"] = parts[1]; meta["version"] = parts[3] 
                if len(parts) >= 5 and parts[2].startswith("L") and parts[3].startswith("v") and not parts[1].startswith("L"):
                     meta["concept"] = parts[0]; meta["style"] = parts[1] + "_" + parts[2]; meta["version"] = parts[4]
                elif len(parts) >= 4 and parts[2].startswith("L") and not parts[1].startswith("L"):
                     meta["concept"] = parts[0]; meta["style"] = parts[1]; meta["version"] = parts[3]
        elif parts[0] == "Cont" and parts[1] == "N" and len(parts) == 6:
            meta["concept"] = parts[2]; meta["style"] = parts[3]; meta["version"] = parts[5] 
        elif parts[0] == "RandOneHot" and len(parts) >= 4:
            meta["concept"] = parts[0]; meta["style"] = parts[1]; meta["version"] = parts[3]
        else: 
            meta["concept"] = parts[0]
            if len(parts) > 1 and not parts[1].startswith("L"): meta["style"] = parts[1]
            else: meta["style"] = "default"
            version_part_found = False
            for p_test in reversed(parts):
                if p_test.startswith("v") and len(p_test) > 1 and p_test[1:].isdigit():
                    meta["version"] = p_test; version_part_found = True; break
            if not version_part_found: meta["version"] = "v_unknown"
    if meta["concept"] == "UnknownFileParse": print(f"Warning: Could not robustly parse metadata for {stem}.")
    return meta

def select_prompt_definition_file() -> Path | None:
    # (This function remains the same)
    PROMPT_SETS_DIR.mkdir(parents=True, exist_ok=True)
    text_files = sorted([f for f in PROMPT_SETS_DIR.glob("*.txt") if f.is_file() and "pca" not in f.name.lower()])
    if not text_files: print(f"No non-PCA .txt files found in '{PROMPT_SETS_DIR.resolve()}'."); return None
    print(f"\nAvailable basis definition files in '{PROMPT_SETS_DIR.name}':")
    for i, file_path in enumerate(text_files): print(f"  {i+1}: {file_path.name}")
    while True:
        try:
            choice = input(f"Enter file number (1-{len(text_files)}), or 'q' to quit: ").strip()
            if choice.lower() == 'q': print("Selection cancelled."); return None
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(text_files): return text_files[choice_idx]
            else: print("Invalid choice.")
        except ValueError: print("Invalid input.")
        except KeyboardInterrupt: print("\nSelection cancelled."); return None

def select_basis_generation_method_interactively(available_methods: list[str]) -> str | None:
    # (This function remains the same)
    print("\nAvailable basis generation methods:")
    default_method = BasisGenerationMethods.MEAN_A_VS_B_GRAM_SCHMIDT; default_idx = -1
    for i, method_name in enumerate(available_methods):
        print(f"  {i+1}: {method_name}", end="")
        if method_name == default_method: print(" (Default)", end=""); default_idx = i
        print()
    prompt_msg = f"Enter method number (1-{len(available_methods)})"
    if default_idx != -1: prompt_msg += f", Enter for default ('{default_method}')"
    prompt_msg += ", or 'q' to quit: "
    while True:
        try:
            choice_str = input(prompt_msg).strip()
            if choice_str.lower() == 'q': return None
            if not choice_str and default_idx != -1: return available_methods[default_idx]
            choice_num = int(choice_str)
            if 1 <= choice_num <= len(available_methods): return available_methods[choice_num - 1]
            else: print(f"Invalid selection.")
        except ValueError: print("Invalid input.")
        except KeyboardInterrupt: print("\nMethod selection cancelled."); return None

def process_basis_definition_file(
    filepath: Path, 
    controller: GameController, 
    selected_method: str,
    log_writer: csv.DictWriter, 
    generation_run_id: str,
    valid_methods_list: list[str] # New parameter to pass the list of valid method names
    ):
    print(f"\n--- Processing Basis Definition File: {filepath.name} using method: {selected_method} ---")
    
    bases_to_generate = [] 
    current_filename_from_file = None; prompt_a_text = None; prompt_b_text = None
    try: # (Parsing logic for bases_to_generate remains the same)
        with open(filepath, 'r', encoding='utf-8') as f:
            for line_num, line_content in enumerate(f):
                line = line_content.strip()
                if not line or line.startswith("#"): continue
                if line.lower().endswith(".npz"):
                    if current_filename_from_file and (prompt_a_text is None or prompt_b_text is None):
                        print(f"  Warn (L{line_num+1}): New file '{line}' for prev '{current_filename_from_file}'. Discarded.")
                    current_filename_from_file = line; prompt_a_text = None; prompt_b_text = None
                elif line.startswith("Prompt A:"):
                    if not current_filename_from_file: print(f"  Warn (L{line_num+1}): 'Prompt A:' no file. Skip."); continue
                    if prompt_a_text is not None: print(f"  Warn (L{line_num+1}): Dupe 'Prompt A:' for '{current_filename_from_file}'. Overwrite.")
                    prompt_a_text = line[len("Prompt A:"):].strip()
                elif line.startswith("Prompt B:"):
                    if not current_filename_from_file: print(f"  Warn (L{line_num+1}): 'Prompt B:' no file. Skip."); continue
                    if prompt_a_text is None: print(f"  Warn (L{line_num+1}): 'Prompt B:' no 'A:' for '{current_filename_from_file}'. Skip B."); continue
                    prompt_b_text = line[len("Prompt B:"):].strip()
                    if current_filename_from_file and prompt_a_text and prompt_b_text:
                        bases_to_generate.append({"original_npz_filename": current_filename_from_file, "prompt_A": prompt_a_text, "prompt_B": prompt_b_text})
                        current_filename_from_file = None; prompt_a_text = None; prompt_b_text = None
                else: print(f"  Warn (L{line_num+1}): Unrecognized line: '{line[:50]}...'")
        if current_filename_from_file and (prompt_a_text is None or prompt_b_text is None):
            print(f"  Warn (EOF): Incomplete entry for '{current_filename_from_file}'. Discarded.")
    except FileNotFoundError: print(f"ERROR: File not found {filepath}"); return
    except Exception as e: print(f"ERROR parsing file {filepath}: {e}"); return

    if not bases_to_generate: print("No valid basis definitions found."); return
    print(f"\nFound {len(bases_to_generate)} basis definitions to generate.")
    generated_count = 0; failed_count = 0
    method_prefix = get_method_short_code(selected_method)

    for basis_def in tqdm(bases_to_generate, desc="Generating Bases", unit="basis"):
        log_row_data = {k: None for k in LOG_CSV_HEADER_COLUMNS}
        log_row_data["log_timestamp"] = datetime.datetime.now().isoformat()
        log_row_data["generation_run_id"] = generation_run_id
        log_row_data["definition_file_used"] = str(filepath.name)
        log_row_data["generation_method_selected"] = selected_method
        log_row_data["layer_idx"] = TARGET_LAYER_FOR_BASES

        original_filename = basis_def["original_npz_filename"]
        pa = basis_def["prompt_A"]
        pb = basis_def["prompt_B"]
        
        log_row_data["target_npz_filename_from_source"] = original_filename
        log_row_data["num_prompts_A_in_def"] = 1 
        log_row_data["num_prompts_B_in_def"] = 1
        log_row_data["example_prompt_A"] = pa[:100] + '...' if len(pa) > 100 else pa
        log_row_data["example_prompt_B"] = pb[:100] + '...' if len(pb) > 100 else pb

        final_filename_stem = Path(original_filename).stem
        if selected_method != BasisGenerationMethods.MEAN_A_VS_B_GRAM_SCHMIDT and method_prefix:
            # Use valid_methods_list passed as argument
            known_prefixes = [get_method_short_code(m).upper() for m in valid_methods_list if get_method_short_code(m)]
            if not any(Path(original_filename).stem.upper().startswith(p + "_") for p in known_prefixes if p):
                 final_filename_stem = f"{method_prefix}_{Path(original_filename).stem}"
        
        final_npz_filename_str = final_filename_stem + Path(original_filename).suffix
        full_npz_save_path = BASIS_SAVE_DIRECTORY / final_npz_filename_str
        log_row_data["final_npz_filename_generated"] = final_npz_filename_str
        
        metadata_parts = parse_target_npz_filename_for_metadata(original_filename)
        concept_for_meta = metadata_parts["concept"]
        style_for_meta = metadata_parts["style"]
        version_for_meta = metadata_parts["version"]

        tqdm.write(f"\nGenerating: {final_npz_filename_str} (Method: {selected_method})")
        tqdm.write(f"  (Original filename in source: {original_filename})")

        generation_result = controller.generate_basis_vectors_and_save(
            method=selected_method, prompts_A=[pa], prompts_B=[pb], 
            layer_idx=TARGET_LAYER_FOR_BASES, save_filepath_str=str(full_npz_save_path),
            basis_concept=concept_for_meta, basis_style=style_for_meta, basis_version=version_for_meta
        )
        
        log_row_data["status"] = "SUCCESS" if generation_result["success"] else "FAILURE"
        log_row_data["failure_point"] = generation_result.get("failure_point")
        log_row_data["message_from_controller"] = generation_result.get("message")
        log_row_data["final_npz_filepath_generated"] = generation_result.get("final_npz_filepath") 
        log_row_data["metadata_json_path_generated"] = generation_result.get("metadata_json_filepath")

        debug_metrics = generation_result.get("debug_metrics", {})
        for key, value in debug_metrics.items():
            log_key = f"dbg_{key}" 
            if log_key in LOG_CSV_HEADER_COLUMNS: log_row_data[log_key] = value
        log_writer.writerow(log_row_data)
        
        if generation_result["success"]: tqdm.write(f"  SUCCESS: {generation_result['message']}"); generated_count += 1
        else: tqdm.write(f"  FAILURE: {generation_result['message']}"); failed_count += 1

    print(f"\n--- Basis Generation Complete ---"); print(f"Method Used: {selected_method}")
    print(f"Successfully generated {generated_count} bases.");
    if failed_count > 0: print(f"Failed {failed_count} bases.")
    print(f"Saved in: {BASIS_SAVE_DIRECTORY.resolve()}"); print(f"Metadata in: {(BASIS_SAVE_DIRECTORY / 'basesdata').resolve()}")

def main():
    print("--- Custom Control Basis Generation Script (v_filename_prefix, v_csv_log, valid_methods_fix) ---")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    generation_run_id = str(uuid.uuid4())
    log_filename = LOG_DIR / f"control_basis_generation_log_{generation_run_id[:8]}.csv"
    
    parser = argparse.ArgumentParser(description="Generates control basis vectors.")
    # Define valid_methods here to be available for argparse choices and for passing to process_basis_definition_file
    valid_methods_for_this_script = [
        getattr(BasisGenerationMethods, m) for m in dir(BasisGenerationMethods) 
        if not m.startswith('_') and isinstance(getattr(BasisGenerationMethods, m), str)
        and m != "PCA_A_B_CLUSTER" # Exclude PCA from this script's interactive/CLI options
    ]
    
    parser.add_argument(
        "--method", type=str, default=None, choices=valid_methods_for_this_script,
        help=(f"Basis generation method. Available: {', '.join(valid_methods_for_this_script)}. Interactive if not provided.")
    )
    args = parser.parse_args()

    selected_method_to_use = args.method
    if selected_method_to_use is None:
        print("\nNo method specified via --method.")
        selected_method_to_use = select_basis_generation_method_interactively(valid_methods_for_this_script)
        if selected_method_to_use is None: print("No method selected. Exiting."); return
    print(f"\nUsing basis generation method: '{selected_method_to_use}'")
    print(f"Generation log will be saved to: {log_filename.resolve()}")
    
    try:
        print("Initializing GameController..."); controller = GameController(load_existing_artifacts=False)
        if config.DIMENSION <= 0:
             if hasattr(controller, 'experiment_runner') and hasattr(controller.experiment_runner, 'model') and controller.experiment_runner.model is not None:
                 config.DIMENSION = controller.experiment_runner.model.cfg.d_mlp
             elif 'experiment_runner' in globals() and experiment_runner.model is not None : 
                  config.DIMENSION = experiment_runner.model.cfg.d_mlp
             else: 
                 print("Model or DIMENSION not available, attempting explicit load for DIMENSION..."); 
                 if 'experiment_runner' not in globals(): import experiment_runner
                 experiment_runner.load_model_and_tokenizer() 
             if config.DIMENSION <=0: print("CRITICAL: Model dimension (config.DIMENSION) not properly set."); return
        print(f"GameController ready. Model Dim: {config.DIMENSION}")
    except Exception as e: print(f"CRITICAL: GameController init failed: {e}"); return

    selected_file = select_prompt_definition_file()
    if selected_file and selected_file.is_file():
        with open(log_filename, 'w', newline='', encoding='utf-8') as log_file:
            log_writer = csv.DictWriter(log_file, fieldnames=LOG_CSV_HEADER_COLUMNS)
            log_writer.writeheader()
            # Pass the valid_methods_for_this_script list here
            process_basis_definition_file(selected_file, controller, selected_method_to_use, log_writer, generation_run_id, valid_methods_for_this_script)
    else:
        if selected_file: print(f"ERROR: Path '{selected_file}' not valid. Exiting.")
        else: print("No basis definition file selected. Exiting.")

if __name__ == "__main__":
    main()
# create_random_bases.py
import random
from pathlib import Path
import csv # For logging (NEW)
import uuid # For logging (NEW)
import datetime # For logging (NEW)

# Assuming local imports for a structured project
try:
    from .game_controller import GameController 
    from . import config 
    from .config import LOG_CSV_HEADER_COLUMNS # Import if defined in config, or define locally
except ImportError: # Fallback for running as script
    from game_controller import GameController 
    import config
    # If LOG_CSV_HEADER_COLUMNS is not in config, define it here for the log
    # This should ideally match the one in create_pca_bases_from_file.py etc.
    LOG_CSV_HEADER_COLUMNS = [
        "log_timestamp", "generation_run_id", "definition_file_used",
        "target_npz_filename_from_source", "final_npz_filename_generated",
        "generation_method_selected", "layer_idx", 
        "neuron_A_idx", "neuron_B_idx", # Specific to one-hot
        "status", "failure_point", "message_from_controller",
        "final_npz_filepath_generated", "metadata_json_path_generated",
        "dbg_vec_A_norm_initial", "dbg_vec_B_norm_initial", 
        "dbg_vec_A_B_cosine_sim_initial", # Will be 0 for one-hot
        "dbg_u1_norm_final", "dbg_u2_norm_final", "dbg_u1_u2_dot_product" # Will be 1,1,0
    ]


# --- Configuration for this script ---
NUM_RANDOM_BASES_TO_CREATE = 30 # Updated to 30
TARGET_LAYER_FOR_ONE_HOT = config.ANALYSIS_LAYER 
# MODEL_DIMENSION will be fetched from config after controller init
BASIS_SAVE_DIRECTORY = config.DEFAULT_BASIS_SEARCH_DIR
LOG_DIR = Path("./generation_logs") # NEW for logging
# --- End Configuration ---

def run_creation():
    print("--- Starting Random One-Hot Basis Generation ---")
    LOG_DIR.mkdir(parents=True, exist_ok=True) # NEW
    generation_run_id = str(uuid.uuid4())      # NEW
    log_filename = LOG_DIR / f"onehot_basis_generation_log_{generation_run_id[:8]}.csv" # NEW
    print(f"Generation log will be saved to: {log_filename.resolve()}")


    if not BASIS_SAVE_DIRECTORY.exists():
        BASIS_SAVE_DIRECTORY.mkdir(parents=True, exist_ok=True)

    MODEL_DIMENSION_FROM_CONFIG = 0 # Initialize

    try:
        print("Initializing GameController...")
        controller = GameController(load_existing_artifacts=False) # Don't load all artifacts
        MODEL_DIMENSION_FROM_CONFIG = config.DIMENSION # Fetch after controller might have updated it
        print(f"GameController initialized. Using model dimension: {MODEL_DIMENSION_FROM_CONFIG}")
    except Exception as e:
        print(f"CRITICAL ERROR: Could not initialize GameController: {e}")
        return 

    if MODEL_DIMENSION_FROM_CONFIG <= 0:
        print(f"Error: Model dimension is not valid ({MODEL_DIMENSION_FROM_CONFIG}). Cannot proceed.")
        return

    generated_count = 0
    failed_count = 0 # NEW

    # Open log file for writing
    with open(log_filename, 'w', newline='', encoding='utf-8') as log_file: # NEW
        # Adjust header for one-hot specifics if needed, or use a more generic one.
        # For simplicity, reusing a slightly adapted version of LOG_CSV_HEADER_COLUMNS.
        # Some fields like prompt_A/B will be None or N/A for one-hot.
        custom_log_header = [h for h in LOG_CSV_HEADER_COLUMNS if 'prompt' not in h and 'agg' not in h and 'diff' not in h and 'mean' not in h]
        custom_log_header.insert(7, "neuron_A_idx_param") # Add specific one-hot params if desired
        custom_log_header.insert(8, "neuron_B_idx_param")
        
        log_writer = csv.DictWriter(log_file, fieldnames=custom_log_header, extrasaction='ignore') # Ignore extra keys from controller dict
        log_writer.writeheader()


        for i in range(NUM_RANDOM_BASES_TO_CREATE):
            log_row_data = {k: None for k in custom_log_header} # NEW
            log_row_data["log_timestamp"] = datetime.datetime.now().isoformat() # NEW
            log_row_data["generation_run_id"] = generation_run_id # NEW
            log_row_data["definition_file_used"] = "create_random_bases.py_script" # NEW
            log_row_data["generation_method_selected"] = "one_hot_neurons" # NEW
            log_row_data["layer_idx"] = TARGET_LAYER_FOR_ONE_HOT # NEW (though not directly used by onehot vec gen)


            instance_num = i + 1
            neuron_A = random.randint(0, MODEL_DIMENSION_FROM_CONFIG - 1)
            neuron_B = random.randint(0, MODEL_DIMENSION_FROM_CONFIG - 1)
            while neuron_B == neuron_A: 
                neuron_B = random.randint(0, MODEL_DIMENSION_FROM_CONFIG - 1)

            log_row_data["neuron_A_idx_param"] = neuron_A # NEW
            log_row_data["neuron_B_idx_param"] = neuron_B # NEW

            filename_stem = f"RandOneHot_N{neuron_A}vN{neuron_B}_L{TARGET_LAYER_FOR_ONE_HOT}_v{instance_num}"
            npz_filepath = BASIS_SAVE_DIRECTORY / f"{filename_stem}.npz"
            
            log_row_data["target_npz_filename_from_source"] = filename_stem + ".npz" # NEW (source is the script itself)
            log_row_data["final_npz_filename_generated"] = filename_stem + ".npz" # NEW (no prefixing for one-hot)


            print(f"\nAttempting to generate basis {instance_num}/{NUM_RANDOM_BASES_TO_CREATE}: {npz_filepath.name}")
            print(f"  Neuron A: {neuron_A}, Neuron B: {neuron_B}, Layer: {TARGET_LAYER_FOR_ONE_HOT}")

            # --- MODIFIED CALL ---
            generation_result = controller.generate_onehot_basis_and_save(
                neuron_idx_A=neuron_A,
                neuron_idx_B=neuron_B,
                dimension=MODEL_DIMENSION_FROM_CONFIG, 
                save_filepath_str=str(npz_filepath)
            )
            # --- END MODIFIED CALL ---

            log_row_data["status"] = "SUCCESS" if generation_result["success"] else "FAILURE" # NEW
            log_row_data["failure_point"] = generation_result.get("failure_point") # NEW
            log_row_data["message_from_controller"] = generation_result.get("message") # NEW
            log_row_data["final_npz_filepath_generated"] = generation_result.get("final_npz_filepath") # NEW
            log_row_data["metadata_json_path_generated"] = generation_result.get("metadata_json_filepath") # NEW
            
            debug_metrics = generation_result.get("debug_metrics", {}) # NEW
            for key, value in debug_metrics.items():
                log_key = f"dbg_{key}"
                if log_key in custom_log_header:
                    log_row_data[log_key] = value
            log_writer.writerow(log_row_data) # NEW


            if generation_result["success"]:
                print(f"  SUCCESS: {generation_result['message']}")
                generated_count +=1
            else:
                print(f"  FAILURE: {generation_result['message']}")
                failed_count +=1 # NEW
                # Decide if you want to break on first failure for one-hots
                # print(f"  Stopping further generation due to failure.")
                # break 

    print(f"\n--- Finished Random One-Hot Basis Generation ---")
    print(f"Successfully generated {generated_count} out of {NUM_RANDOM_BASES_TO_CREATE} requested bases.")
    if failed_count > 0: print(f"Failed to generate {failed_count} one-hot bases.") # NEW
    print(f"Detailed generation log saved to: {log_filename.resolve()}") # NEW


if __name__ == "__main__":
    run_creation()
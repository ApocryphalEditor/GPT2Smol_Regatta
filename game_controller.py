# game_controller.py (Version with structured dict return for basis generation)

import datetime
import traceback
from pathlib import Path
import numpy as np
import time 

try:
    from . import experiment_runner
    from . import artifact_formatter
    from . import artifact_manager
    from . import utils_game
    from . import config
    from .config import BasisGenerationMethods, PCA_NUM_SAMPLES_PER_POLE, MAX_OUTPUT_LINES, EPSILON, DIMENSION
except ImportError: 
    import experiment_runner
    import artifact_formatter
    import artifact_manager
    import utils_game
    import config
    from config import BasisGenerationMethods, PCA_NUM_SAMPLES_PER_POLE, MAX_OUTPUT_LINES, EPSILON, DIMENSION


N_MIN_PCA_SAMPLES_PER_POLE = 2 
N_MIN_PCA_SAMPLES_TOTAL = 3  

class GameController:
    def _create_default_generation_result(self) -> dict:
        """Initializes the dictionary for basis generation results."""
        return {
            "success": False,
            "message": "", 
            "final_npz_filepath": None,
            "metadata_json_filepath": None, 
            "failure_point": None, 
            "debug_metrics": {
                "vec_A_norm_initial": None,       # Norm of the first (or averaged) vec_A before any basis vector derivation
                "vec_B_norm_initial": None,       # Norm of the first (or averaged) vec_B before any basis vector derivation
                "vec_A_B_cosine_sim_initial": None, # Cosine similarity between normalized initial vec_A and vec_B
                "diff_vec_norm_unnormalized": None, 
                "mean_vec_norm_unnormalized": None, 
                "u1_norm_final": None,
                "u2_norm_final": None,
                "u1_u2_dot_product": None 
            }
        }

    def __init__(self, load_existing_artifacts: bool = True): # Default changed as per prior discussion
        self.output_history = []
        if experiment_runner.model is None:
            print("[Controller Init] Experiment runner model not loaded. Attempting to load...")
            try:
                experiment_runner.load_model_and_tokenizer()
                config.DIMENSION = experiment_runner.model.cfg.d_mlp # Ensure DIMENSION is set from model
            except Exception as e:
                print(f"[Controller Init Error] Failed to ensure model load: {e}")
        elif config.DIMENSION != experiment_runner.model.cfg.d_mlp : # Ensure DIMENSION matches loaded model
             config.DIMENSION = experiment_runner.model.cfg.d_mlp
             print(f"[Controller Init WARN] config.DIMENSION updated to {config.DIMENSION} from loaded model.")


        self.all_artifacts = [] 
        if load_existing_artifacts:
            print("[Controller Init] Loading existing artifacts...")
            self.all_artifacts = artifact_manager.get_all_artifacts()
        else:
            print("[Controller Init] Skipping load of existing artifacts for this instance.")
        
        self.intervention_rules_string: str = ""
        self.parsed_intervention_rules: dict = {}
        self.intervention_parse_errors: list = []
        self.interventions_are_active: bool = False
        self.use_sampling: bool = config.DEFAULT_DO_SAMPLE
        self.gen_temp: float = config.DEFAULT_TEMPERATURE
        self.gen_top_k: int = config.DEFAULT_TOP_K

        self.add_output_line("System Initialized. Awaiting input.")
        self.add_output_line(f"Using model: {config.MODEL_NAME}, Layer: {config.ANALYSIS_LAYER}, Dim: {config.DIMENSION}")
        self.add_output_line(f"Default Gen Mode: {'Sampling' if self.use_sampling else 'Greedy'}", "INFO")
        print(f"[Controller] Initialized. Effective DIMENSION: {config.DIMENSION}")

    def add_output_line(self, line: str, source: str = "SYSTEM"):
        # ... (same as before) ...
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        full_line = f"[{timestamp} {source}] {line}"
        self.output_history.append(full_line)
        if len(self.output_history) > MAX_OUTPUT_LINES:
            self.output_history = self.output_history[-MAX_OUTPUT_LINES:]

    def process_input(self, user_input: str):
        # ... (same as before) ...
        user_input = user_input.strip()
        if not user_input: return
        self.add_output_line(f"> {user_input}", source="USER")
        current_basis_path, basis_error = experiment_runner.get_current_basis_info()
        if current_basis_path is None:
            self.add_output_line("FATAL ERROR: No vector basis loaded. Experiment aborted.", "ERROR")
            if basis_error: self.add_output_line(f"Basis Status: {basis_error}", "ERROR")
            return
        elif basis_error:
            self.add_output_line(f"CRITICAL WARNING: Current basis has error ({basis_error}). Results may be unreliable.", "WARN")
        self.add_output_line("Running analysis...", source="SYS")
        int_enabled_runner, int_rules_runner = experiment_runner.get_intervention_status()
        if int_enabled_runner: self.add_output_line(f"Notice: Runner Interventions ACTIVE ({len(int_rules_runner)} layers with rules).", "INFO")
        generation_params = {"do_sample": self.use_sampling}
        if self.use_sampling:
            generation_params["temperature"] = self.gen_temp
            if self.gen_top_k > 0: generation_params["top_k"] = self.gen_top_k
        try:
            raw_results = experiment_runner.run_experiment(user_input, generation_params)
        except Exception as e:
            self.add_output_line(f"Experiment Runner Error: {e}", source="ERROR"); traceback.print_exc(); return
        try:
            artifact = artifact_formatter.format_artifact(raw_results)
        except Exception as e:
            self.add_output_line(f"Artifact Formatting Error: {e}", source="ERROR"); traceback.print_exc(); return
        if artifact_manager.save_artifact(artifact): self.all_artifacts.insert(0, artifact)
        else: self.add_output_line(f"Failed to save artifact {artifact.get('artifact_id')}", source="ERROR")
        for line in artifact.get("display_text", ["Error: Artifact has no display text."]): self.add_output_line(line, source="ANLZ")
        self.check_for_unlocks(); self.add_output_line("Awaiting input.", source="SYSTEM")

    def check_for_unlocks(self): pass
    def get_output_history_display(self) -> str: return "\n".join(self.output_history)
    def get_artifact_summary_list(self) -> list[str]: return [f"{a.get('artifact_id', 'NO_ID')} ({a.get('timestamp', 'No Time')[:19].replace('T',' ')})" for a in self.all_artifacts]
    def get_artifact_by_summary_string(self, summary_string: str) -> dict | None:
        # ... (same as before) ...
        try:
            artifact_id = summary_string.split(" ")[0]
            for artifact in self.all_artifacts:
                if artifact.get("artifact_id") == artifact_id: return artifact
        except Exception as e: print(f"Error finding artifact from summary string '{summary_string}': {e}")
        return None
    def load_new_basis(self, file_path_str: str) -> bool:
        # ... (same as before) ...
        try:
            file_path = Path(file_path_str); self.add_output_line(f"Attempting to load basis: {file_path.name}", "SYS")
            success = experiment_runner.load_basis_vectors(file_path)
            if success: msg = f"Successfully loaded basis: {file_path.name}"; self.add_output_line(msg, "INFO")
            else:
                error_msg = experiment_runner.get_current_basis_info()[1]
                msg = f"Failed to load basis: {file_path.name}. Error: {error_msg if error_msg else 'Unknown'}"
                self.add_output_line(msg, "ERROR")
            return success
        except Exception as e: self.add_output_line(f"Error initiating basis load: {e}", "ERROR"); traceback.print_exc(); return False

    def generate_basis_vectors_and_save(
        self, method: str, prompts_A: list[str], prompts_B: list[str], layer_idx: int,
        save_filepath_str: str, basis_concept: str | None = None,
        basis_style: str | None = None, basis_version: str | None = None
    ) -> dict:
        """ Generates basis vectors (u1, u2) using a specified method and saves them. Returns a result dictionary. """
        
        result = self._create_default_generation_result()
        self.add_output_line(f"Basis Gen Start: Method='{method}', L{layer_idx}, #PromptsA={len(prompts_A)}, #PromptsB={len(prompts_B if prompts_B else [])}", "SYS")

        valid_method_values = [getattr(BasisGenerationMethods, attr_name) for attr_name in dir(BasisGenerationMethods) if not attr_name.startswith('_') and isinstance(getattr(BasisGenerationMethods, attr_name), str)]
        if method not in valid_method_values:
            result["message"] = f"Invalid basis generation method: '{method}'. Valid: {valid_method_values}"
            result["failure_point"] = "method_validation"
            return result

        if not prompts_A:
            result["message"] = "Error: prompts_A cannot be empty."
            result["failure_point"] = "prompts_A_empty"
            return result

        # Validate prompt list lengths based on method
        if method != BasisGenerationMethods.PCA_A_B_CLUSTER:
            if len(prompts_A) != 1:
                result["message"] = f"Method '{method}' expects exactly one prompt in prompts_A, got {len(prompts_A)}."
                result["failure_point"] = "prompts_A_length_mismatch_non_pca"; return result
            if method in [BasisGenerationMethods.MEAN_A_VS_B_GRAM_SCHMIDT, BasisGenerationMethods.DIFF_AB_RAND_ORTHO_U2, BasisGenerationMethods.DIFF_AB_MEAN_AB_ORTHO_U2]:
                if not prompts_B or len(prompts_B) != 1:
                    result["message"] = f"Method '{method}' expects exactly one prompt in prompts_B, got {len(prompts_B) if prompts_B else 0}."
                    result["failure_point"] = "prompts_B_length_mismatch_non_pca"; return result
        else: # PCA_A_B_CLUSTER
            if not prompts_B: 
                 result["message"] = f"PCA method '{method}' requires prompts_B for the second pole."
                 result["failure_point"] = "pca_prompts_B_missing"; return result
            if len(prompts_A) < N_MIN_PCA_SAMPLES_PER_POLE or len(prompts_B) < N_MIN_PCA_SAMPLES_PER_POLE :
                msg = (f"PCA method '{method}' needs at least {N_MIN_PCA_SAMPLES_PER_POLE} prompts per pole (target {PCA_NUM_SAMPLES_PER_POLE}). Got A:{len(prompts_A)}, B:{len(prompts_B)}.")
                self.add_output_line(f"Warning: {msg}", "WARN")
                if len(prompts_A) + len(prompts_B) < N_MIN_PCA_SAMPLES_TOTAL :
                     result["message"] = f"PCA needs at least {N_MIN_PCA_SAMPLES_TOTAL} total valid prompts for robust results."
                     result["failure_point"] = "pca_insufficient_total_samples"; return result
        
        try:
            save_filepath = Path(save_filepath_str)
            # Filename prefixing is now handled by calling scripts, controller uses exact path.
            if save_filepath.exists(): self.add_output_line(f"Warning: File '{save_filepath}' exists. It will be overwritten.", "WARN")
            save_filepath.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e_path:
            result["message"] = f"Error preparing save path '{save_filepath_str}': {e_path}"
            result["failure_point"] = "save_path_preparation"; return result

        u1: np.ndarray | None = None
        u2: np.ndarray | None = None
        current_time_seed = int(time.time())
        pca_random_state = 42 # Fixed for PCA reproducibility

        # Metadata dictionary setup
        basis_metadata_dict = {
            "generation_method_selected": method, "timestamp": datetime.datetime.now().isoformat(),
            "target_layer": layer_idx, "model_name": config.MODEL_NAME, "dimension": DIMENSION,
            "basis_npz_filename": save_filepath.name, # Final filename
            "basis_concept": basis_concept if basis_concept is not None else "unknown_concept",
            "basis_style": basis_style if basis_style is not None else "unknown_style",
            "basis_version": basis_version if basis_version is not None else "unknown_version",
            "source_prompts_A": prompts_A, "source_prompts_B": prompts_B if prompts_B else None,
        }

        try:
            vec_A_primary = None
            if method != BasisGenerationMethods.PCA_A_B_CLUSTER:
                vec_A_primary = experiment_runner.get_mean_activation_for_prompt(prompts_A[0], layer_idx)
                if vec_A_primary is None:
                    result["message"] = f"Failed to get activation for primary Prompt A: {prompts_A[0]}"
                    result["failure_point"] = "vec_A_activation_fetch"; return result
                result["debug_metrics"]["vec_A_norm_initial"] = float(np.linalg.norm(vec_A_primary))
            
            vec_B_primary = None
            if method != BasisGenerationMethods.PCA_A_B_CLUSTER and prompts_B:
                vec_B_primary = experiment_runner.get_mean_activation_for_prompt(prompts_B[0], layer_idx)
                if vec_B_primary is None and method in [BasisGenerationMethods.MEAN_A_VS_B_GRAM_SCHMIDT, BasisGenerationMethods.DIFF_AB_RAND_ORTHO_U2, BasisGenerationMethods.DIFF_AB_MEAN_AB_ORTHO_U2]:
                    result["message"] = f"Failed to get activation for primary Prompt B: {prompts_B[0]}"
                    result["failure_point"] = "vec_B_activation_fetch"; return result
                if vec_B_primary is not None:
                    result["debug_metrics"]["vec_B_norm_initial"] = float(np.linalg.norm(vec_B_primary))
                    if vec_A_primary is not None: # Calculate cosine sim if both exist
                        norm_a = utils_game.normalise(vec_A_primary.copy()) # Use copy to avoid altering original
                        norm_b = utils_game.normalise(vec_B_primary.copy())
                        result["debug_metrics"]["vec_A_B_cosine_sim_initial"] = float(np.dot(norm_a, norm_b))
            
            # --- Method-specific logic ---
            if method == BasisGenerationMethods.MEAN_A_VS_B_GRAM_SCHMIDT:
                u1, u2 = utils_game.create_orthonormal_basis(vec_A_primary, vec_B_primary)
                basis_metadata_dict["u1_derivation_details"] = "u1 = normalize(vec_A_primary)"
                basis_metadata_dict["u2_derivation_details"] = "u2 from vec_B_primary via Gram-Schmidt on u1"

            elif method == BasisGenerationMethods.DIFF_AB_RAND_ORTHO_U2:
                diff_vec = utils_game.get_difference_vector(vec_A_primary, vec_B_primary)
                result["debug_metrics"]["diff_vec_norm_unnormalized"] = float(np.linalg.norm(diff_vec))
                u1 = utils_game.normalise(diff_vec)
                u2 = utils_game.get_random_orthogonal_vector(u1, random_seed=current_time_seed)
                basis_metadata_dict["u1_derivation_details"] = "u1 = normalize(vec_A_primary - vec_B_primary)"
                basis_metadata_dict["u2_derivation_details"] = "u2 = random vector made orthogonal to u1"
                basis_metadata_dict["random_seed_for_u2"] = current_time_seed
                
            elif method == BasisGenerationMethods.DIFF_AB_MEAN_AB_ORTHO_U2:
                diff_vec = utils_game.get_difference_vector(vec_A_primary, vec_B_primary)
                result["debug_metrics"]["diff_vec_norm_unnormalized"] = float(np.linalg.norm(diff_vec))
                u1 = utils_game.normalise(diff_vec)
                mean_vec_candidate = utils_game.get_mean_vector(vec_A_primary, vec_B_primary)
                result["debug_metrics"]["mean_vec_norm_unnormalized"] = float(np.linalg.norm(mean_vec_candidate))
                u2 = utils_game.create_u2_orthogonal_to_u1_from_candidate(u1, mean_vec_candidate)
                basis_metadata_dict["u1_derivation_details"] = "u1 = normalize(vec_A_primary - vec_B_primary)"
                basis_metadata_dict["u2_derivation_details"] = "u2_cand = (A+B)/2, u2 from u2_cand orthog to u1"
                
            elif method == BasisGenerationMethods.PCA_A_B_CLUSTER:
                all_pca_vectors, failed_pca_prompts = [], []
                for i, p_a in enumerate(prompts_A):
                    vec = experiment_runner.get_mean_activation_for_prompt(p_a, layer_idx)
                    if vec is not None: all_pca_vectors.append(vec)
                    else: failed_pca_prompts.append(f"A{i}:{p_a[:30]}...")
                num_A_collected = sum(1 for p in prompts_A if p not in [fp.split(':',1)[1][:30]+'...' for fp in failed_pca_prompts if fp.startswith('A')]) # Approx
                
                for i, p_b in enumerate(prompts_B):
                    vec = experiment_runner.get_mean_activation_for_prompt(p_b, layer_idx)
                    if vec is not None: all_pca_vectors.append(vec)
                    else: failed_pca_prompts.append(f"B{i}:{p_b[:30]}...")
                
                if len(all_pca_vectors) < N_MIN_PCA_SAMPLES_TOTAL:
                    result["message"] = f"PCA needs >= {N_MIN_PCA_SAMPLES_TOTAL} valid vectors, got {len(all_pca_vectors)}. Failed: {len(failed_pca_prompts)}"
                    result["failure_point"] = "pca_insufficient_collected_samples"; return result
                
                u1_pca, u2_pca = utils_game.perform_pca_on_activations(all_pca_vectors, n_components=2, random_state_pca=pca_random_state)
                u1, u2 = u1_pca, u2_pca
                basis_metadata_dict.update({
                    "u1_derivation_details": "u1=PC1 from PCA(prompts_A+prompts_B)", 
                    "u2_derivation_details": "u2=PC2 from PCA(prompts_A+prompts_B)",
                    "pca_num_samples_collected": len(all_pca_vectors), 
                    "pca_num_samples_target_A": len(prompts_A), "pca_num_samples_target_B": len(prompts_B),
                    "pca_random_state_used": pca_random_state,
                    "pca_failed_prompts": failed_pca_prompts if failed_pca_prompts else None
                })
            else: # Should have been caught by initial validation
                result["message"] = f"Internal error: Unhandled method '{method}' in dispatch logic."
                result["failure_point"] = "unhandled_method_dispatch"; return result

        except (ValueError, RuntimeError) as e_util:
            result["message"] = f"Vector computation error for method '{method}': {str(e_util)}"
            result["failure_point"] = "vector_computation_util_error"; return result
        except Exception as e_gen:
            result["message"] = f"Unexpected error during vector gen for method '{method}': {str(e_gen)}"
            result["failure_point"] = "unexpected_vector_gen_error"; traceback.print_exc(); return result

        if u1 is None: # u2 can be None if PCA only yields 1 component, though we ask for 2
            result["message"] = f"Method {method} failed: u1 is None post-computation."
            result["failure_point"] = "u1_is_None_final"; return result
        result["debug_metrics"]["u1_norm_final"] = float(np.linalg.norm(u1))

        if method == BasisGenerationMethods.PCA_A_B_CLUSTER and u2 is None:
            self.add_output_line(f"Warning: Method {method} (PCA) resulted in u2 being None. Saving 1D basis not fully supported by loader.", "WARN")
            basis_metadata_dict["WARNING_u2_is_None_PCA"] = True
            # Forcing a failure if u2 is None for PCA for now, as loader expects 2D.
            result["message"] = f"PCA method {method} failed: u2 is None post-computation (loader expects 2D)."
            result["failure_point"] = "u2_is_None_pca_final"; return result
        elif u2 is None and method != BasisGenerationMethods.PCA_A_B_CLUSTER : # Other methods should always produce u2
            result["message"] = f"Method {method} failed: u2 is None post-computation."
            result["failure_point"] = "u2_is_None_final"; return result
        
        if u2 is not None:
            result["debug_metrics"]["u2_norm_final"] = float(np.linalg.norm(u2))
            dot_product = np.dot(u1, u2)
            result["debug_metrics"]["u1_u2_dot_product"] = float(dot_product)
            if not np.isclose(dot_product, 0.0, atol=EPSILON * 100): 
                result["message"] = f"Validation Fail: u1/u2 not orthogonal. Dot: {dot_product:.3e}"
                result["failure_point"] = "final_orthogonality_check"; return result
            basis_metadata_dict["u1_u2_dot_product_check"] = float(dot_product)
        
        try:
            save_data = {'basis_1': u1.astype(np.float32)}
            if u2 is not None: save_data['basis_2'] = u2.astype(np.float32)
            
            np.savez_compressed(save_filepath, **save_data)
            result["final_npz_filepath"] = str(save_filepath)
            
            metadata_dir = save_filepath.parent / "basesdata"
            metadata_dir.mkdir(parents=True, exist_ok=True)
            metadata_json_filename = save_filepath.stem + ".json" 
            metadata_filepath = metadata_dir / metadata_json_filename
            basis_metadata_dict["dimension"] = config.DIMENSION # Ensure current DIMENSION is logged

            if utils_game.safe_save_json(basis_metadata_dict, metadata_filepath):
                result["success"] = True
                result["message"] = f"Basis saved (Method: {method}): NPZ to {save_filepath.name}, Meta to {metadata_filepath.name}"
                result["metadata_json_filepath"] = str(metadata_filepath)
                self.add_output_line(result["message"], "INFO")
            else:
                result["message"] = f"Saved NPZ to {save_filepath.name}, but FAILED to save metadata JSON."
                result["failure_point"] = "metadata_json_save_fail"
                # NPZ was saved, so this isn't a total failure of generation, but metadata is crucial.
        except Exception as e_save:
            result["message"] = f"Error saving NPZ or JSON for method '{method}': {str(e_save)}"
            result["failure_point"] = "file_save_final"; traceback.print_exc()
        
        return result

    def generate_onehot_basis_and_save(self, neuron_idx_A: int, neuron_idx_B: int, dimension: int, save_filepath_str: str) -> dict:
        result = self._create_default_generation_result()
        self.add_output_line(f"OneHot Gen Start: N{neuron_idx_A} vs N{neuron_idx_B}, Dim={dimension}", "SYS")
        result["debug_metrics"]["vec_A_norm_initial"] = 1.0 # By definition for one-hot
        result["debug_metrics"]["vec_B_norm_initial"] = 1.0 # By definition for one-hot
        result["debug_metrics"]["vec_A_B_cosine_sim_initial"] = 0.0 # By definition if neurons differ

        if not (0 <= neuron_idx_A < dimension and 0 <= neuron_idx_B < dimension):
            result["message"] = f"Neuron indices ({neuron_idx_A}, {neuron_idx_B}) out of bounds for dim {dimension}."
            result["failure_point"] = "onehot_idx_bounds"; return result
        if neuron_idx_A == neuron_idx_B:
            result["message"] = "Neuron indices for one-hot basis must be different."
            result["failure_point"] = "onehot_idx_same"; return result
            
        try:
            save_filepath = Path(save_filepath_str)
            if save_filepath.exists(): self.add_output_line(f"Warning: File '{save_filepath}' exists. Overwriting.", "WARN")
            save_filepath.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e_path:
            result["message"] = f"Error preparing save path for one-hot: {e_path}"
            result["failure_point"] = "onehot_save_path_prep"; return result

        vec_A = np.zeros(dimension, dtype=np.float32); vec_A[neuron_idx_A] = 1.0
        vec_B = np.zeros(dimension, dtype=np.float32); vec_B[neuron_idx_B] = 1.0
        u1, u2 = vec_A, vec_B # For one-hot, they are already orthonormal if distinct

        result["debug_metrics"]["u1_norm_final"] = 1.0
        result["debug_metrics"]["u2_norm_final"] = 1.0
        dot_product = np.dot(u1,u2) # Should be 0
        result["debug_metrics"]["u1_u2_dot_product"] = float(dot_product)
        # Orthogonality check for one-hot is trivial if indices are different.

        basis_metadata_dict = {
            "generation_method_selected": "one_hot_neurons", "timestamp": datetime.datetime.now().isoformat(),
            "neuron_A_idx": neuron_idx_A, "neuron_B_idx": neuron_idx_B,
            "model_name": config.MODEL_NAME, "dimension": dimension, # Use passed dimension
            "basis_npz_filename": save_filepath.name,
            "basis_concept": "N/A_onehot", "basis_style": "N/A_onehot", "basis_version": "N/A_onehot",
            "u1_derivation_details": f"u1=one_hot(N{neuron_idx_A})", "u2_derivation_details": f"u2=one_hot(N{neuron_idx_B})",
            "u1_u2_dot_product_check": float(dot_product)
        }
        
        try:
            np.savez_compressed(save_filepath, basis_1=u1, basis_2=u2)
            result["final_npz_filepath"] = str(save_filepath)
            metadata_dir = save_filepath.parent / "basesdata"
            metadata_dir.mkdir(parents=True, exist_ok=True)
            metadata_json_filename = save_filepath.stem + ".json"
            metadata_filepath = metadata_dir / metadata_json_filename

            if utils_game.safe_save_json(basis_metadata_dict, metadata_filepath):
                result["success"] = True
                result["message"] = f"One-hot basis saved: NPZ to {save_filepath.name}, Meta to {metadata_filepath.name}"
                result["metadata_json_filepath"] = str(metadata_filepath)
                self.add_output_line(result["message"], "INFO")
            else:
                result["message"] = f"Saved OneHot NPZ, but FAILED to save metadata JSON."
                result["failure_point"] = "onehot_metadata_save_fail"
        except Exception as e_save:
            result["message"] = f"Error saving OneHot NPZ or JSON: {str(e_save)}"
            result["failure_point"] = "onehot_file_save_final"; traceback.print_exc()
        
        return result

    # --- Other existing methods (intervention, UI state, etc.) ---
    def get_current_basis_path_and_error(self) -> tuple[Path | None, str | None]: # ... (same as before) ...
        return experiment_runner.get_current_basis_info()
    def set_intervention_rules(self, rules_string: str): # ... (same as before, ensure DIMENSION is correct) ...
        self.intervention_rules_string = rules_string
        if config.DIMENSION <=0 : # Check if DIMENSION needs update from model
            if experiment_runner.model and hasattr(experiment_runner.model.cfg, 'd_mlp'):
                config.DIMENSION = experiment_runner.model.cfg.d_mlp
                print(f"[Controller Interv] DIMENSION updated to {config.DIMENSION} for parsing.")
            else:
                self.add_output_line("ERROR: DIMENSION not available for intervention parsing!", "ERROR")
                experiment_runner.set_active_interventions({}, False); return

        parsed_rules, errors = utils_game.parse_dynamic_interventions(rules_string)
        self.intervention_parse_errors = errors; self.parsed_intervention_rules = parsed_rules
        experiment_runner.set_active_interventions(self.parsed_intervention_rules, self.interventions_are_active)
        if errors: [self.add_output_line(f"Rule Parse Err: {err}", "WARN") for err in errors]
        elif rules_string.strip() and not parsed_rules and not errors: self.add_output_line("Warn: Rules string valid but no rules parsed.", "WARN")
        elif parsed_rules: self.add_output_line(f"Interv rules parsed: {len(parsed_rules)}L, {sum(len(r) for r in parsed_rules.values())}R.", "INFO")
    def set_interventions_active(self, active: bool): # ... (same as before) ...
        self.interventions_are_active = active
        experiment_runner.set_active_interventions(self.parsed_intervention_rules, self.interventions_are_active)
        runner_enabled, rules_in_runner = experiment_runner.get_intervention_status()
        if runner_enabled == self.interventions_are_active:
             status_msg = "ENABLED" if runner_enabled else "DISABLED"
             num_rules_active = sum(len(r) for r in rules_in_runner.values()) if runner_enabled else 0
             num_layers_active = len(rules_in_runner) if runner_enabled else 0
             if runner_enabled and not self.parsed_intervention_rules and not self.intervention_rules_string.strip(): self.add_output_line(f"Interv {status_msg} (No rules).", "WARN")
             elif runner_enabled and self.intervention_parse_errors: self.add_output_line(f"Interv {status_msg} ({num_layers_active}L, {num_rules_active}R) [!] ERRORS.", "WARN")
             elif runner_enabled: self.add_output_line(f"Interv {status_msg} ({num_layers_active}L, {num_rules_active}R).", "INFO")
             else: self.add_output_line(f"Interv {status_msg}.", "INFO")
        else: self.add_output_line(f"STATE MISMATCH! Ctrl: {self.interventions_are_active}, Run: {runner_enabled}", "ERROR")
    def get_intervention_status_for_ui(self) -> tuple[str, bool, bool]: # ... (same as before) ...
        is_runner_enabled, rules_in_runner = experiment_runner.get_intervention_status()
        is_logically_active = self.interventions_are_active; num_layers = len(self.parsed_intervention_rules)
        num_rules = sum(len(r) for r in self.parsed_intervention_rules.values()); has_errors = bool(self.intervention_parse_errors)
        status_str = "ACTIVE" if is_logically_active else "INACTIVE"
        if is_logically_active:
            if not self.parsed_intervention_rules and not self.intervention_rules_string.strip(): display_string = f"Interv: {status_str} (No rules)"
            elif has_errors: display_string = f"Interv: {status_str} ({num_layers}L, {num_rules}R) [!] {len(self.intervention_parse_errors)} errors"
            else: display_string = f"Interv: {status_str} ({num_layers}L, {num_rules}R)"
        else: display_string = f"Interv: {status_str}";
        if has_errors and not is_logically_active : display_string += f" ({len(self.intervention_parse_errors)} parse err)" # Show parse err count even if inactive
        return display_string, is_logically_active and is_runner_enabled, has_errors
    def get_intervention_rules_string(self) -> str: return self.intervention_rules_string # ... (same as before) ...
    def set_generation_mode(self, use_sampling_mode: bool): # ... (same as before) ...
        self.use_sampling = use_sampling_mode; self.add_output_line(f"Gen mode: {'Sampling' if self.use_sampling else 'Greedy'}", "INFO")
    def get_initial_sampling_state(self) -> bool: return self.use_sampling # ... (same as before) ...
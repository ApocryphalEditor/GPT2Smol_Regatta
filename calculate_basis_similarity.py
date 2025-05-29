# calculate_basis_similarity.py
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.linalg import subspace_angles
from tqdm import tqdm
import time

# ADD THIS IMPORT
from sklearn.cluster import AgglomerativeClustering

import config
import utils_game

# --- Configuration ---
BASIS_FILES_DIR = config.DEFAULT_BASIS_SEARCH_DIR
OUTPUT_DIR = config.ANALYSIS_OUTPUT_ROOT_DIR / "basis_similarity_analysis"
# --- End Configuration ---

def load_basis_vectors_from_npz(filepath: Path) -> tuple[np.ndarray | None, np.ndarray | None]:
    # ... (this function remains the same as before) ...
    try:
        with np.load(filepath) as data:
            u1 = data.get('basis_1')
            u2 = data.get('basis_2')
            if u1 is not None and u2 is not None:
                u1 = u1.astype(np.float32)
                u2 = u2.astype(np.float32)
                if u1.shape == (config.DIMENSION,) and u2.shape == (config.DIMENSION,):
                    return u1, u2
                else:
                    print(f"Warning: Dimension mismatch in {filepath.name}. Expected ({config.DIMENSION},), got u1:{u1.shape}, u2:{u2.shape}")
                    return None, None
            else:
                print(f"Warning: 'basis_1' or 'basis_2' not found in {filepath.name}")
                return None, None
    except Exception as e:
        print(f"Error loading basis vectors from {filepath.name}: {e}")
        return None, None

# NEW FUNCTION FOR CLUSTERING
def perform_clustering(
    distance_matrix_df: pd.DataFrame,
    angle_threshold_degrees: float = 15.0, # Threshold for merging clusters
    linkage_method: str = 'average'
) -> pd.DataFrame:
    """
    Performs agglomerative clustering on a precomputed distance matrix.

    Args:
        distance_matrix_df (pd.DataFrame): DataFrame where index and columns are basis filenames
                                           and values are pairwise distances (e.g., principal angles).
        angle_threshold_degrees (float): The distance threshold for merging clusters, in degrees.
        linkage_method (str): Linkage method for agglomerative clustering (e.g., 'average', 'complete').

    Returns:
        pd.DataFrame: DataFrame with 'filename' and 'cluster_id'.
    """
    print(f"\nPerforming clustering with threshold: {angle_threshold_degrees}° ({linkage_method} linkage)...")
    if distance_matrix_df.empty:
        print("Error: Distance matrix is empty, cannot perform clustering.")
        return pd.DataFrame(columns=['filename', 'cluster_id'])

    angle_threshold_radians = np.deg2rad(angle_threshold_degrees)

    # Ensure the distance matrix is symmetric and NaNs are handled (e.g., fill with max distance or drop rows/cols)
    # For now, assume it's mostly good. sklearn might handle NaNs depending on version/method.
    # If NaNs are an issue, you might need to impute them (e.g., with np.pi/2 for max distance)
    # or ensure no NaNs are passed to the clustering algorithm.
    # For AgglomerativeClustering, input should not have NaNs.
    
    # Simple NaN handling: replace with a large distance (pi/2, or slightly more to be safe)
    # This ensures problematic bases become their own clusters or don't wrongly merge.
    processed_distance_matrix = distance_matrix_df.fillna(np.pi).to_numpy()


    clustering_model = AgglomerativeClustering(
        n_clusters=None,  # Important when using distance_threshold
        metric='precomputed',  # We provide a distance matrix
        linkage=linkage_method,
        distance_threshold=angle_threshold_radians
    )

    try:
        cluster_labels = clustering_model.fit_predict(processed_distance_matrix)
    except ValueError as e:
        print(f"Error during clustering: {e}")
        print("This might happen if the distance matrix still contains NaNs or unexpected values after fillna.")
        print("Shape of matrix passed to clustering:", processed_distance_matrix.shape)
        # Fallback: create a dummy clustering where each item is its own cluster
        cluster_labels = np.arange(len(distance_matrix_df))


    cluster_assignments = pd.DataFrame({
        'filename': distance_matrix_df.index, # Assumes index has filenames
        'cluster_id': cluster_labels
    })

    num_unique_clusters = cluster_assignments['cluster_id'].nunique()
    print(f"Found {num_unique_clusters} unique clusters.")
    print("Cluster size distribution:")
    print(cluster_assignments['cluster_id'].value_counts().value_counts().sort_index().to_string()) # Counts of cluster sizes

    return cluster_assignments


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    basis_files_paths = sorted(list(BASIS_FILES_DIR.glob("*.npz")))
    if not basis_files_paths:
        print(f"No .npz basis files found in {BASIS_FILES_DIR}")
        return

    num_bases = len(basis_files_paths)
    print(f"Found {num_bases} basis files to compare.")

    # --- Define paths for distance matrices ---
    # These are the outputs from the *previous* run of this script, or will be generated now
    principal_angle_dist_path = OUTPUT_DIR / "principal_angle_distances.csv"
    u1_cosine_dist_path = OUTPUT_DIR / "u1_cosine_angle_distances.csv" # Also generated for completeness

    # --- Load or Calculate Principal Angle Distance Matrix ---
    if principal_angle_dist_path.exists():
        print(f"\nLoading pre-computed principal angle distance matrix from: {principal_angle_dist_path}")
        df_principal_angle_dist = pd.read_csv(principal_angle_dist_path, index_col=0)
        # Ensure basis_filenames are consistent with the loaded df
        basis_filenames_from_csv = df_principal_angle_dist.index.tolist()
        if len(basis_filenames_from_csv) != num_bases or \
           not all(Path(BASIS_FILES_DIR / fn).exists() for fn in basis_filenames_from_csv):
            print("Warning: Loaded distance matrix filenames don't match current basis files. Recalculating...")
            # Fall through to recalculate
        else:
            # Ensure the filenames from the CSV match the order of basis_files_paths if we were to use them directly
            # For safety, let's re-align basis_files_paths if loading from CSV to match the matrix order
            basis_files_paths = [BASIS_FILES_DIR / fn for fn in basis_filenames_from_csv]
            print("Successfully loaded principal angle distance matrix.")
    
    # If not loaded or needs recalculation, proceed to calculate
    # This block will execute if principal_angle_dist_path doesn't exist OR if the above check failed
    if not principal_angle_dist_path.exists() or 'df_principal_angle_dist' not in locals():
        print("\nPrincipal angle distance matrix not found or needs recalculation. Calculating now...")
        loaded_bases_data = []
        for bf_path in tqdm(basis_files_paths, desc="Loading basis files for distance calc"):
            u1, u2 = load_basis_vectors_from_npz(bf_path)
            plane_matrix = None
            if u1 is not None and u2 is not None:
                plane_matrix = np.stack((u1, u2), axis=1)
            loaded_bases_data.append({
                "filepath": bf_path, "filename": bf_path.name,
                "u1": u1, "u2": u2, "plane_matrix": plane_matrix
            })

        principal_angle_distance_matrix = np.full((num_bases, num_bases), np.nan)
        u1_cosine_distance_matrix = np.full((num_bases, num_bases), np.nan) # Also recalc this

        print("\nCalculating pairwise similarities/distances...")
        for i in tqdm(range(num_bases), desc="Outer loop (basis i)"):
            # ... (The same calculation logic as before for both matrices) ...
            data_i = loaded_bases_data[i]
            if data_i["u1"] is None: continue
            u1_i = data_i["u1"]; plane_matrix_i = data_i["plane_matrix"]
            for j in range(i, num_bases):
                data_j = loaded_bases_data[j]
                if data_j["u1"] is None: continue
                u1_j = data_j["u1"]; plane_matrix_j = data_j["plane_matrix"]
                if i == j:
                    u1_cosine_distance_matrix[i, j] = 0.0
                    principal_angle_distance_matrix[i, j] = 0.0
                    continue
                
                # Cosine
                dot_product_u1 = np.dot(u1_i, u1_j)
                acute_angle_rad_u1 = np.arccos(np.clip(np.abs(dot_product_u1), -1.0, 1.0))
                u1_cosine_distance_matrix[i, j] = acute_angle_rad_u1
                u1_cosine_distance_matrix[j, i] = u1_cosine_distance_matrix[i, j]
                
                # Principal Angle
                if plane_matrix_i is not None and plane_matrix_j is not None:
                    try:
                        angles_rad_principal = subspace_angles(plane_matrix_i, plane_matrix_j)
                        principal_angle_distance_matrix[i, j] = angles_rad_principal[0]
                        principal_angle_distance_matrix[j, i] = principal_angle_distance_matrix[i, j]
                    except Exception as e_pa:
                        print(f"Error PA for {data_i['filename']} vs {data_j['filename']}: {e_pa}")
                        principal_angle_distance_matrix[i, j] = np.pi / 2
                        principal_angle_distance_matrix[j, i] = np.pi / 2
                else:
                    principal_angle_distance_matrix[i, j] = np.pi / 2
                    principal_angle_distance_matrix[j, i] = np.pi / 2

        basis_filenames = [data["filename"] for data in loaded_bases_data]
        df_u1_cosine_dist = pd.DataFrame(u1_cosine_distance_matrix, index=basis_filenames, columns=basis_filenames)
        df_principal_angle_dist = pd.DataFrame(principal_angle_distance_matrix, index=basis_filenames, columns=basis_filenames)

        df_u1_cosine_dist.to_csv(u1_cosine_dist_path)
        print(f"\nSaved u1 cosine angle distance matrix to: {u1_cosine_dist_path}")
        df_principal_angle_dist.to_csv(principal_angle_dist_path)
        print(f"Saved first principal angle distance matrix to: {principal_angle_dist_path}")
        
        load_status_df_data = [{"filename": d["filename"], "u1_loaded": d["u1"] is not None, "u2_loaded": d["u2"] is not None, "plane_matrix_valid": d["plane_matrix"] is not None} for d in loaded_bases_data]
        load_status_df = pd.DataFrame(load_status_df_data)
        load_status_df.to_csv(OUTPUT_DIR / "basis_load_status_for_similarity.csv", index=False)
        print(f"Saved basis load status to: {OUTPUT_DIR / 'basis_load_status_for_similarity.csv'}")


    # --- Perform Clustering using the Principal Angle Distances ---
    # We use df_principal_angle_dist which is now guaranteed to be loaded or calculated
    # Angle threshold for clustering (e.g., 15 degrees)
    ANGLE_THRESHOLD_DEGREES_FOR_CLUSTERING = 15.0 # You can tune this
    
    cluster_assignments_df = perform_clustering(
        df_principal_angle_dist, # Pass the DataFrame
        angle_threshold_degrees=ANGLE_THRESHOLD_DEGREES_FOR_CLUSTERING,
        linkage_method='average' # Or 'complete'
    )

    # Save the cluster assignments
    cluster_assignments_path = OUTPUT_DIR / f"basis_geometric_clusters_thresh{int(ANGLE_THRESHOLD_DEGREES_FOR_CLUSTERING)}deg.csv"
    cluster_assignments_df.to_csv(cluster_assignments_path, index=False)
    print(f"\nSaved basis cluster assignments to: {cluster_assignments_path}")

if __name__ == "__main__":
    main()
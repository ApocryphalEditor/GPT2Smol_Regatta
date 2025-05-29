Probing Safety Representations in GPT-2 Small (Layer 11)

This repository contains the Python code and key data outputs for an exploratory investigation into how safety-related concepts are represented within the MLP Layer 11 activation space of the GPT-2 Small model. 

The core methodology involves:
1.  Constructing a diverse set of 608 2D vector basis planes from various human-intuitive and nonsense definitional prompts, using multiple generation techniques (Mean A vs. B, DMEAN, DRND, PCA, Random One-Hot).
2.  Projecting the mean activations of a standardized set of 140 test prompts (covering polar safe/unsafe statements, stylistic variations, abstract tags, and controls) onto these basis planes.
3.  Measuring the geometric similarity (first principal angle) between all pairs of basis planes and performing agglomerative clustering (using a 15-degree threshold) to identify ~512 geometrically distinct directional clusters.
4.  Evaluating each geometric cluster based on:
    *   **Plane Relevance (`r_baseline`):** How much of a test prompt's activation energy is captured by the plane.
    *   **Directional Separation (`theta_polarity_score`):** The plane's ability to angularly separate "Safe-aligned" vs. "Unsafe-aligned" test prompts.
    *   **Statistical Significance:** Using permutation tests (1000 iterations, dual exemplar strategies per cluster) and Benjamini-Hochberg FDR correction to assess if observed `theta_polarity_score`s are greater than chance.

## Key Preliminary Findings

*   **High Geometric Diversity:** Subtle variations in definitional prompts often lead to significantly different ( >15° principal angle) 2D basis planes.
*   **Human-Intuitive "Safe/Unsafe" Axes:** While these planes often show high relevance (`r_baseline`) to safety-themed test prompts, they generally exhibit very low `theta_polarity_score`s, which are not statistically significant after FDR correction.
*   **"Nonsense" Axes:** Some control/nonsense axes show high *descriptive* `theta_polarity_score`s. However, these also largely fail to achieve statistical significance after permutation testing and FDR correction.
*   **Subtle Signals:** A few DMEAN-generated "nonsense" axes (e.g., from stylized Japanese prompts) showed very low raw p-values (<0.001) for weak polarity scores, though these did not survive FDR correction across all 512 cluster tests. This hints at potential subtle, non-obvious geometric biases that warrant further investigation.

**Overall Conclusion (Preliminary):** With the current metrics and 2D plane projection approach, we did not find a simple, statistically robust "safety compass" in Layer 11 that strongly and significantly separates general safe vs. unsafe concepts directionally. The model's mechanisms appear more nuanced.

For a more engaging summary of these findings, see [Link to your Reddit post / Blog post / Future Paper].

## Repository Structure

A detailed manifest of files and directories can be found in [FILE_MANIFEST.md](FILE_MANIFEST.md). Key directories include:
*   `src/` (or root if scripts are in root): Python scripts for basis generation, similarity calculation, analysis, and evaluation.
*   `game_data/vectorbases/`: The 608 `.npz` basis vector files and their JSON metadata in `basesdata/`.
*   `promptsets/`: Text files defining prompts for basis creation and testing.
*   `analysis_results/`: Output data, including cluster definitions, master data CSVs (full set and per chunk), cluster quality evaluations with p-values, plots, and reports.

## Setup

1.  **Clone the repository in environment of your choice.**
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: For GPU usage with PyTorch, ensure appropriate CUDA drivers are installed.)*

## Running the Analysis Pipeline

The main analysis scripts should be run in the following order. Pre-computed output CSVs are already provided in `analysis_results/` if you wish to skip reproduction steps.

1.  **Calculate Basis Similarity & Clusters (Optional - `basis_geometric_clusters_thresh15deg.csv` provided):**
    *   Generates pairwise distances between all `.npz` bases and performs clustering.
    *   Script: `calculate_basis_similarity.py`
    *   Key Output: `analysis_results/basis_similarity_analysis/basis_geometric_clusters_thresh15deg.csv`
    *   Note: 15 degree differentiations are not a magic number just a starting point. A full cross-sweep analysis is on the to-do list. 

2.  **Process Experimental Results & Generate Master Data (Uses provided summary data):**
    *   The raw experimental outputs (~85,000 individual JSON files, one per prompt-basis test) are **not included** in this repository due to size.
    *   Instead, **pre-computed master data summary CSVs are provided** in:
        *   `analysis_results/analysis_results_default_full_set/master_artifact_data_default_full_set_v2.5.1_clusteredHR.csv` (for all 140 test prompts)
        *   `analysis_results/analysis_results_C<N>/master_artifact_data_C<N>_v2.5.1_clusteredHR.csv` (for each chunk C1-C14 of test prompts)
    *   The script `analyze_artifacts.py` is provided to show the logic that *would* process raw JSONs and merge cluster IDs to produce these master CSVs. If you had the raw JSONs, you could run it.

3.  **Evaluate Cluster Quality & Run Permutation Tests:**
    *   This script consumes a `master_artifact_data... .csv` (e.g., for the `default_full_set`) and the cluster definitions to perform cluster-level quality evaluation and statistical significance testing.
    *   Ensure `config.py` has `SCRIPT_VERSION_SUFFIX_FOR_ANALYSIS_FILES` set to match the suffix of the master data CSV you are analyzing (default current version is `"v2.5.1_clusteredHR"`).
    *   Script: `python evaluate_basis_quality.py`
    *   Key Output: `analysis_results/analysis_results_default_full_set/cluster_quality_evaluation_perm_default_full_set_v2.5.1_clusteredHR.csv` (and associated plots/reports).

*(Scripts for generating the initial 608 `.npz` basis vectors are also included for completeness.)*

## Key Result Files

*   **Cluster Definitions:** `analysis_results/basis_similarity_analysis/basis_geometric_clusters_thresh15deg.csv`
*   **Master Data (Full Set):** `analysis_results/analysis_results_default_full_set/master_artifact_data_default_full_set_v2.5.1_clusteredHR.csv`
*   **Cluster Quality & Permutation Test Results (Full Set):** `analysis_results/analysis_results_default_full_set/cluster_quality_evaluation_perm_default_full_set_v2.5.1_clusteredHR.csv`
*   Key plots and text reports are located within the `plots...` subdirectories of `analysis_results/analysis_results_default_full_set/` and the per-chunk directories.

## Further Exploration & Collaboration

This work is exploratory. The provided code and data allow for further investigation, such as:
*   Exploring different clustering thresholds.
*   Testing alternative metrics (e.g., based on activation magnitude, last-token representations).
*   Training linear probes on the full activation space.

Contributions, suggestions, and collaborations are welcome! PyjamaKooka on Reddit. 

## License
MIT
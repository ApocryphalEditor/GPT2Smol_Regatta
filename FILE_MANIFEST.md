# Repository File Manifest

This document provides an overview of the key files and directories within this repository for the "Probing Safety Representations in GPT-2 Small (MLP Layer 11)" project.

## Root Directory

*   **`README.md`**: The main introductory document for the project, outlining purpose, setup, how to run analyses, and key findings.
*   **`requirements.txt`**: A list of Python packages required to run the scripts in this repository. Install using `pip install -r requirements.txt`.
*   **`config.py`**: Contains core configuration settings used by various scripts, such as model names, target layers, default parameters, and file paths for analysis outputs. **Key variable `SCRIPT_VERSION_SUFFIX_FOR_ANALYSIS_FILES` should match the suffix of data files being processed by `evaluate_basis_quality.py`.**
*   **`.gitignore`**: Specifies intentionally untracked files that Git should ignore (e.g., `__pycache__`, local environment files).
*   **`FILE_MANIFEST.md`**: This document.

---

## Code Scripts (primarily in root or a `src/` directory)

*   **Core Experiment & Utility Scripts:**
    *   `utils_game.py`: Contains low-level utility functions for vector math (normalization, differences, means, orthogonality), JSON handling, timestamps, and other helpers used across multiple scripts.
    *   `experiment_runner.py`: Manages the GPT-2 model and tokenizer, handles activation capture, interventions (if enabled), text generation, and calculates projections of prompt activations onto loaded 2D basis planes. This script produces the raw individual artifact JSONs (not included in this repo).
    *   `game_controller.py`: Provides a higher-level interface for running experiments and generating basis vectors. Used by the basis creation scripts.
    *   `artifact_formatter.py`: Formats the raw dictionary output from `experiment_runner.py` into a more structured "artifact" dictionary for display or saving.
    *   `artifact_manager.py`: (Included for completeness of original framework) Handles saving and loading of individual artifact JSONs.

*   **Basis Generation Scripts:**
    *   `create_control_bases_from_file.py`: Generates `.npz` basis vector files (and their JSON metadata) from pairs of definitional prompts (A vs. B) listed in a formatted `.txt` file. Supports MAB, DMEAN, DRND methods.
    *   `create_pca_bases_from_file.py`: Generates `.npz` basis vector files (and metadata) using Principal Component Analysis (PCA) on activations from clusters of A-prompts and B-prompts defined in a structured `.txt` file.
    *   `create_random_bases.py`: Generates `.npz` basis vector files (and metadata) from pairs of random one-hot neuron directions.

*   **Main Analysis Pipeline Scripts:**
    *   `calculate_basis_similarity.py`:
        *   **Input:** Reads all `.npz` basis files from `game_data/vectorbases/`.
        *   **Process:** Calculates pairwise geometric similarity (first principal angle and u1 cosine angle) between all basis planes. Performs agglomerative clustering based on principal angles.
        *   **Key Output:** `analysis_results/basis_similarity_analysis/basis_geometric_clusters_thresh15deg.csv` (mapping basis filenames to cluster IDs) and distance matrix CSVs.
    *   `analyze_artifacts.py`:
        *   **Input:** Typically processes raw artifact JSONs (from `game_data/artifacts/` if they were present) and the `basis_geometric_clusters...csv`. *In this repository, it's intended to be run by users on the provided `master_artifact_data...csv` files if they modify them, or used to understand how these master files were derived from raw data.*
        *   **Process:** Parses artifact data, extracts projection metrics, merges with basis metadata and geometric cluster IDs.
        *   **Key Output:** `master_artifact_data_[source_name]_[version_suffix].csv` files (located in `analysis_results/analysis_results_[source_name]/`) containing processed data for all test prompts and bases. Also generates various summary plots.
    *   `evaluate_basis_quality.py`:
        *   **Input:** A `master_artifact_data_[source_name]_[version_suffix].csv` file (e.g., for `default_full_set`).
        *   **Process:** Calculates quality metrics (e.g., `theta_polarity_score`, relevance ratios) for individual bases, aggregates these to the geometric cluster level. Performs permutation tests for `theta_polarity_score` at the cluster level using dual exemplar strategies, and applies multiple comparison corrections (Bonferroni, FDR).
        *   **Key Output:** `cluster_quality_evaluation_perm_[source_name]_[version_suffix].csv` (in `analysis_results/analysis_results_[source_name]/`) containing detailed quality metrics and p-values for each geometric cluster. Also generates summary plots.

## Input Data Directories

*   **`promptsets/`**:
    *   Contains `.txt` files that define prompts.
    *   `PromptSet_SafeV1_Formatted.txt`: The standardized set of 140 test prompts used for evaluating basis planes in `batch_runner.py` (which generates raw artifacts) and subsequently analyzed by `analyze_artifacts.py` and `evaluate_basis_quality.py`.
    *   Other `.txt` files in this directory are inputs for `create_control_bases_from_file.py` and `create_pca_bases_from_file.py`, specifying the definitional prompts (Prompt A, Prompt B, or prompt clusters) used to construct the 608 basis vector files.

*   **`game_data/vectorbases/`**:
    *   Contains the 608 `.npz` files, each storing a 2D orthonormal basis (two 3072-dimensional vectors, `basis_1` and `basis_2`). These are the core "probes" or "sailboats."
    *   **`basesdata/` (subdirectory):** Contains a `.json` metadata file for each corresponding `.npz` file. These JSONs detail how the basis was generated (method, source prompts, target layer, etc.).

*   **`game_data/artifacts/` (Conceptual - Raw Data Not Included):**
    *   This directory is where raw output JSONs from `batch_runner.py` / `experiment_runner.py` would be stored (one JSON per prompt-basis pair, ~85,000 files).
    *   **These raw JSONs are not included in this repository due to their collective size.** The provided `master_artifact_data... .csv` files in `analysis_results/` are derived from these.

---

## Analysis Results Directory (`analysis_results/`)

This directory contains the processed data and outputs from the analysis pipeline.

*   **`analysis_results/basis_similarity_analysis/`**:
    *   `basis_geometric_clusters_thresh15deg.csv`: Maps each `.npz` basis filename to a geometric `cluster_id` (based on a 15-degree principal angle threshold). **Essential input for subsequent cluster-aware analyses.**
    *   `principal_angle_distances.csv`: Pairwise first principal angle distances between all 608 basis planes.
    *   `u1_cosine_angle_distances.csv`: Pairwise cosine angle distances between the `u1` vectors of all 608 basis planes.
    *   `basis_load_status_for_similarity.csv`: A small utility CSV confirming which bases were loaded for similarity calculation.

*   **`analysis_results/analysis_results_default_full_set/`**:
    *   Contains results from analyzing the **full set of 140 test prompts** against all bases.
    *   `master_artifact_data_default_full_set_[version_suffix].csv`: The primary processed data file from `analyze_artifacts.py`. Contains projection metrics (`r_baseline`, `theta_baseline`, etc.), basis metadata, and merged `cluster_id` for every test prompt projected onto every basis. **Key input for `evaluate_basis_quality.py`.**
    *   `cluster_quality_evaluation_perm_default_full_set_[version_suffix].csv`: The primary output from `evaluate_basis_quality.py`. Contains one row per geometric cluster, with aggregated quality metrics, permutation test p-values (raw and corrected), and exemplar basis information. **This is the main file for interpreting statistically validated findings.**
    *   `plots_default_full_set_[version_suffix]/`: Subdirectory containing plots generated by `analyze_artifacts.py` and `evaluate_basis_quality.py` for the full dataset, including:
        *   Semantic `r_baseline` distributions.
        *   The high-resolution `r_baseline` by geometric cluster plot.
        *   `top_bottom_geometric_clusters_report_default_full_set.txt`: Text report of top/bottom performing clusters.

*   **`analysis_results/analysis_results_C<N>/` (for N from 1 to 14, if included):**
    *   Each directory (e.g., `analysis_results_C1`) contains results from `analyze_artifacts.py` run on a specific **chunk (subset) of the test prompts**. This allows for examining how bases/clusters perform with different types of test inputs.
    *   `master_artifact_data_C<N>_[version_suffix].csv`: Master data for that specific chunk.
    *   `plots_C<N>_[version_suffix]/`: Plots specific to that chunk's analysis, including its own `r_baseline` by geometric cluster plot and top/bottom report.
    *   *(Note: `evaluate_basis_quality.py` is typically run on the `default_full_set` data for robust p-values, but could be adapted for chunks if desired).*

---
Further files can be provided on request including batch_runner.py if you want to generate/reproduce the original steps yourself. Reach out to me on Reddit. 
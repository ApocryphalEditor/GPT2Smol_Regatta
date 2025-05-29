# artifact_manager.py
from pathlib import Path
import utils_game
from config import ARTIFACTS_DIR

# Ensure the directory exists at startup
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

def save_artifact(artifact_data: dict):
    """Saves a single artifact dictionary to a JSON file."""
    artifact_id = artifact_data.get("artifact_id")
    if not artifact_id:
        print("Error: Cannot save artifact without 'artifact_id'.")
        return False
    filepath = ARTIFACTS_DIR / f"{artifact_id}.json"
    print(f"[Artifact Mgr] Saving artifact: {filepath.name}")
    return utils_game.safe_save_json(artifact_data, filepath)

def load_artifact(artifact_id: str) -> dict | None:
    """Loads a single artifact by ID."""
    filepath = ARTIFACTS_DIR / f"{artifact_id}.json"
    return utils_game.safe_load_json(filepath)

def get_all_artifacts() -> list[dict]:
    """Loads all artifacts from the storage directory."""
    artifacts = []
    print(f"[Artifact Mgr] Loading all artifacts from: {ARTIFACTS_DIR}")
    try:
        for filepath in ARTIFACTS_DIR.glob("ART-*.json"):
            artifact = utils_game.safe_load_json(filepath)
            if artifact:
                artifacts.append(artifact)
        # Sort by timestamp (assuming it's in the artifact data)
        artifacts.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        print(f"[Artifact Mgr] Loaded {len(artifacts)} artifacts.")
    except Exception as e:
        print(f"Error scanning artifact directory {ARTIFACTS_DIR}: {e}")
    return artifacts

# Example: Function to get artifacts with a specific tag (for unlocks later)
def get_artifacts_by_tag(tag: str, all_artifacts: list[dict] | None = None) -> list[dict]:
    """Filters artifacts by a specific tag."""
    if all_artifacts is None:
        all_artifacts = get_all_artifacts()
    return [a for a in all_artifacts if tag in a.get("tags", [])]
import json
import yaml
from pathlib import Path

def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f) if path.endswith(".json") else yaml.safe_load(f)
    
def resolve_path_from_config(config: dict, path: str) -> tuple[Path, str, str, Path]:
    dataset_name = config["dataset"]["name"]
    scene = config["dataset"]["config"].get("scene")

    run_dir = Path(path) / dataset_name
    train_dir = "train_" + dataset_name
    val_dir = "val_" + dataset_name
    model_dir = Path("models/weights", dataset_name)
    if scene is not None:
        run_dir = run_dir / scene
        model_dir = model_dir / scene

    return run_dir, train_dir, val_dir, model_dir

# save config in structured (dataset/scene) directory
def save_config(config: dict, path: str) -> None:
    run_dir, _, _, _ = resolve_path_from_config(config, path)
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "config.yaml").open("w", encoding="utf-8") as f:
        yaml.dump(config, f)

    
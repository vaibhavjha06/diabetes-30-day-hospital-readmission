from pathlib import Path
import yaml


def load_config(path: str | Path) -> dict:
    """Load and validate the project's YAML configuration."""
    path = Path(path)

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    return config

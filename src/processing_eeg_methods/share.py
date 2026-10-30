import os
from pathlib import Path


def _root() -> str:
    """Folder that holds Datasets/ and Results/: $EEG_ENSEMBLE_ROOT if set, else the repository
    checkout this package lives in, else the current working directory."""
    if os.environ.get("EEG_ENSEMBLE_ROOT"):
        return str(Path(os.environ["EEG_ENSEMBLE_ROOT"]).resolve())
    checkout = Path(__file__).parent.parent.parent.resolve()
    if (checkout / "pyproject.toml").exists():
        return str(checkout)
    return str(Path.cwd().resolve())


ROOT_VOTING_SYSTEM_PATH: str = _root()
GLOBAL_SEED = 42

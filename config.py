"""Central configuration for the five-class skin image project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW_DATA_DIR = ROOT / "data" / "raw"
PROCESSED_DATA_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"
ARTIFACTS_DIR = ROOT / "artifacts"
RESULTS_DIR = REPORTS_DIR / "results"
PLOTS_DIR = REPORTS_DIR / "plots"

SEED = 42
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_CLASSES = 5
INITIAL_EPOCHS = 10
FINE_TUNE_EPOCHS = 5
LEARNING_RATE = 1e-3
FINE_TUNE_LEARNING_RATE = 1e-5
CLASS_NAMES = ["normal", "wrinkle", "acne", "dark_spot", "large_pores"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# Only explicit, reviewed names belong here. Unknown names are excluded, never guessed.
FOLDER_CLASS_MAP = {
    "acne": "acne", "wrinkle": "wrinkle", "wrinkles": "wrinkle",
    "spots": "dark_spot", "dark spots": "dark_spot", "dark_spots": "dark_spot",
    "normal": "normal", "large pore": "large_pores", "large pores": "large_pores",
    "large_pores": "large_pores",
}
COCO_CLASS_MAP = {
    "Acne": "acne", "Wrinkles": "wrinkle", "Dark-Spots": "dark_spot",
    # The published Roboflow category is misspelled this way.
    "Englarged-Pores": "large_pores",
}

SOURCE_NAMES = (
    "kaggle_acne_wrinkle_spots",
    "kaggle_oily_dry_normal",
    "roboflow_skin_problem_clean3",
)

def locate_source(name: str) -> Path | None:
    """Support both the requested data/raw layout and the repository's current layout."""
    for candidate in (RAW_DATA_DIR / name, ROOT / name):
        if candidate.exists():
            return candidate
    return None

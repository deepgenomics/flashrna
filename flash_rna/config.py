from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent

# Data
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"

BORZOI_HUMAN_TARGETS = DATA_DIR / "borzoi_tracks_mapping_hg38.csv"
BORZOI_MOUSE_TARGETS = DATA_DIR / "borzoi_tracks_mapping_mm10.csv"
FLASH_RNA_HUMAN_TARGETS = DATA_DIR / "flashrna_tracks_mapping_hg38.csv"
FLASH_RNA_MOUSE_TARGETS = DATA_DIR / "flashrna_tracks_mapping_mm10.csv"

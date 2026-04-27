"""Project paths and global constants."""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
UMASS_DIR = RAW_DIR / "Umass"

OUTPUT_DIR = ROOT_DIR / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"

for d in [OUTPUT_DIR, FIGURE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42

INPUT_SIZE = 96
HORIZON = 24
D_MODEL = 64
N_POLYNOMIALS = 3
N_HARMONICS = 5

TRAIN_RATIO = 0.7
VAL_RATIO = 0.1

import os



# centralize paths here so train.py / eval.py stay cleaner

    # data directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

    # checkpoint directory
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
DEFAULT_CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pth")
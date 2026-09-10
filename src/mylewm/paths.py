"""Shared checkout paths; no data, manifest, or output paths are rewritten."""
import os
from pathlib import Path

ROOT = Path(os.environ.get('BT_SIGREG_ROOT', Path(__file__).resolve().parents[2])).expanduser().resolve()
SOURCE = Path(__file__).resolve().parent
CONFIGS = ROOT / 'mylewm' / 'configs'


def source_files():
    """All shipped Python sources, including path/bootstrap and compatibility code."""
    return sorted(SOURCE.rglob('*.py'))


def libero_paths():
    """External runtime locations can be supplied by each host without code edits."""
    return (
        Path(os.environ.get('LIBERO_ROOT', ROOT / 'external/libero')).expanduser(),
        Path(os.environ.get('LIBERO_MUJOCO_PATH', ROOT / '.cache/libero-runtime')).expanduser(),
    )

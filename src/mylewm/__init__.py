"""BT-SIGReg implementation; install this checkout in editable mode."""
import sys
from .paths import ROOT

# Trusted upstream object checkpoints refer to top-level jepa/module classes.
# Keep the comparison implementation and its resources in the repository.
for path in (ROOT, ROOT / 'lewm'):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

"""Narrow compatibility shims for the locked Real-TSQA runners.

The shared Python 3.9 environment exposes a user-site scikit-learn package but
does not expose its SciPy dependency.  Transformers 4.57 detects the package
metadata and imports ``sklearn.metrics`` for optional assisted generation even
though the Real-TSQA runners do not use assisted generation.  Install a small
fail-closed stub only when importing the real package fails.  Any accidental
use of one of these optional metrics raises immediately.
"""

from __future__ import annotations

import importlib.machinery
import sys
import types


def guard_optional_sklearn() -> bool:
    """Return True only when a broken optional sklearn was replaced by a stub."""
    try:
        import sklearn  # noqa: F401
        return False
    except (ImportError, OSError):
        for name in tuple(sys.modules):
            if name == "sklearn" or name.startswith("sklearn."):
                sys.modules.pop(name, None)

    def _unavailable(*_args, **_kwargs):
        raise RuntimeError(
            "scikit-learn is unavailable in this runtime; this optional metric "
            "must not be used by the Real-TSQA inference runners"
        )

    sklearn = types.ModuleType("sklearn")
    sklearn.__file__ = "<real-tsqa-sklearn-stub>"
    sklearn.__path__ = []
    sklearn.__version__ = "0.0.0-real-tsqa-stub"
    sklearn.__spec__ = importlib.machinery.ModuleSpec(
        "sklearn", loader=None, is_package=True
    )
    metrics = types.ModuleType("sklearn.metrics")
    metrics.__file__ = "<real-tsqa-sklearn-metrics-stub>"
    metrics.__spec__ = importlib.machinery.ModuleSpec(
        "sklearn.metrics", loader=None
    )
    metrics.roc_curve = _unavailable
    metrics.f1_score = _unavailable
    metrics.matthews_corrcoef = _unavailable
    sklearn.metrics = metrics
    sys.modules["sklearn"] = sklearn
    sys.modules["sklearn.metrics"] = metrics
    return True


def guard_legacy_dynamic_cache_api() -> tuple[bool, bool, bool]:
    """Restore read-only legacy names used by pinned official remote code."""
    from transformers.cache_utils import DynamicCache

    added_seen_tokens = False
    added_max_length = False
    added_usable_length = False
    if not hasattr(DynamicCache, "seen_tokens"):
        DynamicCache.seen_tokens = property(lambda self: self.get_seq_length())
        added_seen_tokens = True
    if not hasattr(DynamicCache, "get_max_length"):
        def _legacy_max_length(self):
            value = self.get_max_cache_shape()
            return None if value is None or value < 0 else value
        DynamicCache.get_max_length = _legacy_max_length
        added_max_length = True
    if not hasattr(DynamicCache, "get_usable_length"):
        def _legacy_usable_length(self, new_seq_length, layer_idx=0):
            previous = self.get_seq_length(layer_idx)
            maximum = self.get_max_length()
            if maximum is not None and previous + new_seq_length > maximum:
                return maximum - new_seq_length
            return previous
        DynamicCache.get_usable_length = _legacy_usable_length
        added_usable_length = True
    return added_seen_tokens, added_max_length, added_usable_length

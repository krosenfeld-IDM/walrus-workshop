import os
from typing import Optional, Sequence
import numpy as np


class ActivationManager:
    """
    Handles selective saving of activations from GraphCast layers.

    Source: https://github.com/theodoremacmillan/graphcast/blob/sae-hooks/graphcast/deep_typed_graph_net.py
    """

    def __init__(self,
                 enabled: bool = False,
                 save_dir: Optional[str] = None,
                 save_steps: Optional[Sequence[int]] = None,
                 save_node_sets: Optional[Sequence[str]] = None,
                 mode: str = "post_res"):
        self.enabled = enabled
        self.save_dir = save_dir
        self.save_steps = save_steps
        self.save_node_sets = save_node_sets
        self.mode = mode
        self.current_time_str: Optional[str] = None   # <--- NEW
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

    # --- add these small helpers ---
    def set_time(self, time_str: Optional[str]):
        """Set a global time string (e.g. '2021-09-28T06Z') for subsequent saves."""
        self.current_time_str = time_str

    def clear_time(self):
        """Unset the current global time label."""
        self.current_time_str = None

    def _should_save(self, tag: str, step_idx: Optional[int], node_set: str) -> bool:
        """Determine if this activation should be saved."""
        if not self.enabled:
            return False
        if self.mode not in tag and self.mode != "both":
            return False
        if self.save_steps is not None and step_idx not in self.save_steps:
            return False
        if self.save_node_sets is not None and node_set not in self.save_node_sets:
            return False
        return True

    def save(self, tag: str, x, *,
             step_idx: Optional[int] = None,
             node_set: Optional[str] = None,
             time_str: Optional[str] = None):
        """Save activation array x for given step / node set."""
        if not self._should_save(tag, step_idx, node_set):
            return

        # Prefer explicit time_str if provided, else use the global one
        ts = time_str or self.current_time_str
        arr = np.asarray(x).copy()

        safe_tag = tag.replace("/", "_")
        step_prefix = f"layer{step_idx:04d}_" if step_idx is not None else ""
        time_suffix = f"_t{ts}" if ts else ""
        node_suffix = f"_{node_set}" if node_set else ""

        fname = f"{step_prefix}{safe_tag}{time_suffix}.npy"
        np.save(os.path.join(self.save_dir, fname), arr)

    def get_cache(self):
        """Return in-memory cache (if using memory mode)."""
        return self._cache if self._cache is not None else {}

    def clear(self):
        """Clear in-memory cache."""
        if self._cache is not None:
            self._cache.clear()


# Global instance (imported across modules)
_ACT_MANAGER = ActivationManager()

def get_activation_manager():
    """Return global ActivationManager instance."""
    return _ACT_MANAGER
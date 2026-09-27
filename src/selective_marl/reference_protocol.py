"""Mechanism-isolated transforms shared with the pinned reference evaluator."""

from __future__ import annotations

import numpy as np

MOVEMENT_ROTATION = np.asarray((0, 3, 4, 2, 1), dtype=np.int64)
MESSAGE_COUNT = 10


def permute_received_messages(obs: np.ndarray, permutation: np.ndarray) -> np.ndarray:
    """Permute only the received-message suffix of an official MPE observation batch."""

    changed = np.asarray(obs, dtype=np.float32).copy()
    changed[:, -MESSAGE_COUNT:] = changed[:, -MESSAGE_COUNT:][:, permutation]
    return changed


def rotate_teammate_movement(action_indices: np.ndarray) -> np.ndarray:
    """Rotate agent 1's movement branch while retaining both communication branches."""

    changed = np.asarray(action_indices, dtype=np.int64).copy()
    changed[1, 0] = MOVEMENT_ROTATION[changed[1, 0]]
    return changed

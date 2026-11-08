"""Lõi mô phỏng vector hóa (chỉ NumPy, không import gì từ play/)."""
from .catalog import Catalog, load_catalog, DEFAULT_CATALOG
from .state import State, reset
from .actions import (legal_actions, mask_parts, explain, action_name, NUM_ACTIONS,
                      BUY0, SELL0, MOVE0, ROLL, BUY_XP, END_TURN)
from .step import step, IllegalActionError, InvariantError
from .combat import CombatModel, FormulaCombat, board_features
from .pairing import meet_prob, draw_pairs
from .invariants import check_invariants, state_hash
from . import rng

__all__ = [
    "Catalog", "load_catalog", "DEFAULT_CATALOG", "State", "reset", "legal_actions", "mask_parts",
    "explain", "action_name", "NUM_ACTIONS", "BUY0", "SELL0", "MOVE0", "ROLL", "BUY_XP", "END_TURN",
    "step", "IllegalActionError", "InvariantError", "CombatModel", "FormulaCombat", "board_features",
    "meet_prob", "draw_pairs", "check_invariants", "state_hash", "rng",
]

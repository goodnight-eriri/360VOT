"""JSON parameter loading for SiamFC tracker.

Replaces the ``paramparse``-based loader from deep_mdp with a simple
``json.load`` approach.  Only the fields required for inference are loaded.
"""

from __future__ import annotations

import json
import os


def load_json(path: str) -> dict:
    with open(path, 'r') as f:
        return json.load(f)


def load_siamfc_params(params_dir: str | None = None):
    """Load SiamFC hyperparameters and design parameters from JSON files.

    Args:
        params_dir: Directory containing ``hyperparams.json`` and
                    ``design.json``.  Defaults to the ``parameters/``
                    subdirectory next to this file.

    Returns:
        (hyper, design) namespace-like objects.
    """
    if params_dir is None:
        params_dir = os.path.join(os.path.dirname(__file__), 'parameters')

    hyper_path  = os.path.join(params_dir, 'hyperparams.json')
    design_path = os.path.join(params_dir, 'design.json')

    hyper_dict  = load_json(hyper_path)
    design_dict = load_json(design_path)

    class _Namespace:
        def __init__(self, d: dict):
            for k, v in d.items():
                setattr(self, k, v)

        def __repr__(self):
            return f"{self.__class__.__name__}({self.__dict__})"

    return _Namespace(hyper_dict), _Namespace(design_dict)

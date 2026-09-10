"""Site parameters and global random-seed control."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass

SEED = 42


@dataclass(frozen=True)
class Site:
    key: str
    latitude: float = 45.0
    longitude: float = 23.3
    capacity_kwp: float = 50.0


SITES: dict[str, Site] = {
    "site_a": Site("site_a"),
    "site_b": Site("site_b"),
}


def set_seed(seed: int = SEED) -> int:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np
    except ImportError:
        pass
    else:
        np.random.seed(seed)

    try:
        import torch
    except ImportError:
        return seed

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed

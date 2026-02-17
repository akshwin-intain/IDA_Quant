from __future__ import annotations

import pandas as pd


def load_ces_csv(path: str, *, low_memory: bool = False) -> pd.DataFrame:
    """
    Load the raw CES CSV (the one producing mixed dtype warnings in notebooks).
    """
    return pd.read_csv(path, low_memory=low_memory)



"""Percentage momentum; overlap with historical returns is intentional and disclosed."""
import pandas as pd
from src.features.returns import historical_returns


def momentum(close: pd.Series, windows: list[int]) -> pd.DataFrame:
    """Use P_t/P_(t-n)-1, preserving explicitly requested duplicate horizons."""
    result = historical_returns(close, windows)
    return result.rename(columns={f'return_{n}d': f'momentum_{n}d' for n in windows})

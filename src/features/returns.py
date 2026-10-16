"""Backward simple price changes on one validated continuous asset segment."""
import pandas as pd


def historical_returns(close: pd.Series, windows: list[int]) -> pd.DataFrame:
    """Fractional returns, never percentages multiplied by 100 or auto-filled."""
    return pd.DataFrame({f'return_{n}d': close/close.shift(n)-1 for n in windows}, index=close.index)

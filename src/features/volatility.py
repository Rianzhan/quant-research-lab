"""Trailing daily simple-return volatility, without annualization."""
import pandas as pd


def historical_volatility(close: pd.Series, windows: list[int], ddof: int) -> pd.DataFrame:
    """Require n valid one-session returns, hence n+1 continuous prices."""
    daily = close/close.shift(1)-1
    return pd.DataFrame({f'volatility_{n}d': daily.rolling(n,min_periods=n,center=False).std(ddof=ddof)
                         for n in windows}, index=close.index)

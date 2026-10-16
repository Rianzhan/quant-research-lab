"""Strict missing-volume propagation without replacement or interpolation."""
import pandas as pd


def volume_features(volume: pd.Series, change_periods: int, average_window: int) -> pd.DataFrame:
    """All dependency observations must be valid, including the current date."""
    p, n = change_periods, average_window
    clean = volume.where(volume.gt(0))
    complete = clean.notna().rolling(p+1,min_periods=p+1).sum().eq(p+1)
    change = (clean/clean.shift(p)-1).where(complete)
    ratio = clean/clean.rolling(n,min_periods=n,center=False).mean()
    return pd.DataFrame({f'volume_change_{p}d': change, f'volume_ratio_{n}d': ratio}, index=volume.index)

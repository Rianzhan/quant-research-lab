"""Explicit moving-average, Wilder RSI and arithmetic-seeded EMA/MACD rules."""
import numpy as np
import pandas as pd


def moving_averages(close: pd.Series, windows: list[int], pairs: list[list[int]]) -> pd.DataFrame:
    """Right-aligned price-unit means and short-minus-long differences."""
    means = {n: close.rolling(n,min_periods=n,center=False).mean() for n in windows}
    result = {f'ma_{n}d': value for n,value in means.items()}
    result.update({f'ma_diff_{a}_{b}': means[a]-means[b] for a,b in pairs})
    return pd.DataFrame(result,index=close.index)


def _seeded_average(values: pd.Series, period: int, alpha: float) -> pd.Series:
    """Seed from period consecutive values; missing input resets all state."""
    if type(period) is not int or period<1:
        raise ValueError('Smoothing period must be a positive integer')
    result = np.full(len(values),np.nan)
    seed, state = [], None
    for i,value in enumerate(values.to_numpy(dtype=float)):
        if not np.isfinite(value):
            seed, state = [], None
            continue
        if state is None:
            seed.append(value)
            if len(seed)<period:
                continue
            state = float(np.mean(seed))
            seed = []
        else:
            state = alpha*value+(1-alpha)*state
        result[i] = state
    return pd.Series(result,index=values.index)


def seeded_ema(values: pd.Series, period: int) -> pd.Series:
    """EMA alpha=2/(period+1), first value is the first complete arithmetic mean."""
    return _seeded_average(values,period,2/(period+1))


def wilder_rsi(close: pd.Series, period: int) -> pd.Series:
    """Wilder alpha=1/period on gains/losses; a fully flat initialized RSI is 50."""
    change = close.diff()
    gain = _seeded_average(change.clip(lower=0),period,1/period)
    loss = _seeded_average((-change).clip(lower=0),period,1/period)
    total = gain+loss
    result = pd.Series(np.nan,index=close.index,dtype=float)
    result.loc[total.gt(0)] = 100*gain.loc[total.gt(0)]/total.loc[total.gt(0)]
    result.loc[total.eq(0)] = 50.0
    return result


def macd(close: pd.Series, fast: int, slow: int, signal: int) -> pd.DataFrame:
    """Return line, signal and unscaled line-minus-signal histogram."""
    line = seeded_ema(close,fast)-seeded_ema(close,slow)
    signal_line = seeded_ema(line,signal)
    return pd.DataFrame({'macd_line': line,'macd_signal': signal_line,'macd_hist': line-signal_line},index=close.index)

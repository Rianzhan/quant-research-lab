"""Describe gaps and anomalies without inferring listing or delisting from missing rows."""
from __future__ import annotations
import numpy as np
import pandas as pd

from src.data.cleaner import DataQualityError, PRICES, date_strings


def audit_panel(panel: pd.DataFrame, calendar: list[str], assets: list[dict],
                start_date: str, end_date: str, *, vwap_tolerance: float = 0.005,
                price_break_threshold: float = 0.30) -> dict:
    """Audit domestic sessions and unbalanced asset histories; never fill missing dates."""
    sessions = date_strings(pd.Series(calendar)).tolist()
    if not sessions or len(set(sessions)) != len(sessions) or sessions != sorted(sessions):
        raise DataQualityError('Calendar must be nonempty, unique and increasing')
    if start_date < sessions[0] or end_date > sessions[-1]:
        raise DataQualityError('Requested range exceeds calendar coverage')
    expected = {d for d in sessions if start_date <= d <= end_date}
    report = {'assets': {}, 'common_dates': [], 'noncommon_dates': [], 'calendar_is_vendor_supplied': True}
    date_sets = []
    for asset in assets:
        frame = panel.loc[(panel.asset_code == asset['code']) & (panel.exchange == asset['exchange'])]
        observed = set(frame.date); date_sets.append(observed)
        outside = sorted(observed-expected)
        if outside:
            raise DataQualityError(f'Rows outside requested domestic trading sessions: {outside[:5]}')
        first, last = (min(observed), max(observed)) if observed else (None, None)
        internal = sorted(d for d in expected-observed if first is not None and first <= d <= last)
        positive_volume = frame['volume'] > 0
        vwap = frame.loc[positive_volume, 'amount']/frame.loc[positive_volume, 'volume']
        mismatch = (vwap < frame.loc[positive_volume,'low']-vwap_tolerance) | (vwap > frame.loc[positive_volume,'high']+vwap_tolerance)
        # Diagnostic only: observed-price discontinuities are not model features or economic returns.
        breaks = (frame['close']/frame['close'].shift(1)-1).abs() > price_break_threshold
        report['assets'][asset['exchange']+':'+asset['code']] = {
            'first_observed_date': first, 'last_observed_date': last, 'rows': len(frame),
            'duplicate_dates': int(frame.date.duplicated().sum()), 'monotonic': bool(frame.date.is_monotonic_increasing),
            'internal_missing_dates': internal,
            'leading_unobserved_dates': sorted(d for d in expected if first is None or d < first),
            'trailing_unobserved_dates': sorted(d for d in expected if last is not None and d > last),
            'zero_volume_dates': frame.loc[~positive_volume,'date'].tolist(),
            'large_price_break_dates': frame.loc[breaks,'date'].tolist(),
            'missing_amount_rows': int(frame.amount.isna().sum()),
            'amount_volume_outside_ohlc_dates': frame.loc[vwap.index[mismatch],'date'].tolist(),
            'listing_status': 'unverified', 'delisting_status': 'unverified',
            'suspension_status': 'unverified; missing/zero-volume observations are not proof',
        }
    common = set.intersection(*date_sets) if date_sets else set()
    union = set.union(*date_sets) if date_sets else set()
    report['common_dates'] = sorted(common)
    report['noncommon_dates'] = sorted(union-common)
    return report


def compare_sources(left: pd.DataFrame, right: pd.DataFrame, *, price_tolerance: float,
                    volume_relative_tolerance: float) -> dict:
    """Join by asset AND date, and surface conflicts instead of selecting better-looking prices."""
    keys = ['date','exchange','asset_code']
    if left.duplicated(keys).any() or right.duplicated(keys).any():
        raise DataQualityError('Cannot compare duplicate observations')
    joined = left.merge(right, on=keys, suffixes=('_a','_b'), validate='one_to_one')
    price_bad = pd.Series(False, index=joined.index)
    max_price = 0.0
    for col in PRICES:
        diff = (joined[col+'_a']-joined[col+'_b']).abs()
        price_bad |= diff > price_tolerance+1e-12
        if len(diff): max_price = max(max_price, float(diff.max()))
    denominator = np.maximum(np.maximum(joined.volume_a.abs(), joined.volume_b.abs()), 1)
    relative_volume = (joined.volume_a-joined.volume_b).abs()/denominator
    volume_bad = relative_volume > volume_relative_tolerance
    return {'conflict': bool((price_bad | volume_bad).any()), 'overlap_rows': len(joined),
            'max_absolute_price_difference': max_price,
            'price_conflict_dates': joined.loc[price_bad,'date'].tolist(),
            'volume_conflict_dates': joined.loc[volume_bad,'date'].tolist(),
            'no_overlap': joined.empty}

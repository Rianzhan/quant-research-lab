"""Validate observations without imputing dates, prices or asset identities."""
from __future__ import annotations
import re

import numpy as np
import pandas as pd


class DataQualityError(ValueError):
    """Invalid observations must be inspected, never silently repaired or dropped."""


CORE = ['date', 'asset_code', 'exchange', 'asset_name', 'category', 'open', 'high', 'low',
        'close', 'volume', 'amount', 'source', 'snapshot_id', 'price_basis']
PRICES = ['open', 'high', 'low', 'close']


def date_strings(values: pd.Series) -> pd.Series:
    """Require date-only domestic session labels; never shift time zones implicitly."""
    result = values.astype(str)
    if not result.str.fullmatch(r'\d{4}-\d{2}-\d{2}').all():
        raise DataQualityError('Invalid date-only trading session')
    try:
        pd.to_datetime(result, format='%Y-%m-%d', errors='raise')
    except (ValueError, OverflowError) as exc:
        raise DataQualityError(f'Invalid trading date: {exc}') from exc
    return result


def _validate_values(frame: pd.DataFrame) -> None:
    for col in PRICES+['volume']:
        if not np.isfinite(frame[col].to_numpy(dtype=float)).all():
            raise DataQualityError(f'Missing or nonfinite {col}')
    amount = frame['amount'].dropna().to_numpy(dtype=float)
    if not np.isfinite(amount).all() or (amount < 0).any():
        raise DataQualityError('Invalid amount')
    if (frame[PRICES] <= 0).any().any() or (frame['volume'] < 0).any():
        raise DataQualityError('Nonpositive price or negative volume')
    if ((frame['high'] < frame[PRICES].max(axis=1)) | (frame['low'] > frame[PRICES].min(axis=1))).any():
        raise DataQualityError('OHLC bounds are inconsistent')


def clean_prices(raw: pd.DataFrame, asset: dict, source: str, snapshot_id: str, *,
                 price_basis: str = 'none', listing_date: str | None = None) -> pd.DataFrame:
    """Normalize one asset; source volume units are converted, missing rows never added."""
    if price_basis != 'none':
        raise DataQualityError('Adjusted price policy is not approved; keep adjustment probes separate')
    if source not in ('akshare_em', 'akshare_sina', 'baostock'):
        raise DataQualityError('Unknown source')
    if not re.fullmatch(r'[0-9]{6}', str(asset['code'])) or asset['exchange'] not in ('SSE', 'SZSE') or not snapshot_id:
        raise DataQualityError('Invalid asset identity or snapshot')
    if raw.empty or raw.columns.duplicated().any():
        raise DataQualityError('Empty data or duplicate fields')
    frame = raw.copy(deep=True)
    if source == 'akshare_em':
        frame = frame.rename(columns={'日期':'date','开盘':'open','最高':'high','最低':'low',
                                      '收盘':'close','成交量':'volume','成交额':'amount'})
    expected = ('sh.' if asset['exchange'] == 'SSE' else 'sz.')+asset['code']
    for col in ('code', 'asset_code'):
        if col in frame and not frame[col].astype(str).isin([asset['code'], expected]).all():
            raise DataQualityError('Provider asset identity mismatch')
    if 'adjustflag' in frame and not frame['adjustflag'].astype(str).eq('3').all():
        raise DataQualityError('Provider adjustment differs from unadjusted request')
    required = {'date', *PRICES, 'volume'}
    if not required <= set(frame):
        raise DataQualityError(f'Missing fields: {sorted(required-set(frame))}')
    frame['date'] = date_strings(frame['date'])
    if frame['date'].duplicated().any():
        raise DataQualityError('duplicate dates within asset')
    if listing_date and (frame['date'] < listing_date).any():
        raise DataQualityError('Observed rows precede independently verified listing date')
    if 'amount' not in frame:
        frame['amount'] = np.nan
    try:
        for col in PRICES+['volume','amount']:
            frame[col] = pd.to_numeric(frame[col].replace('', np.nan), errors='raise')
    except (ValueError, TypeError) as exc:
        raise DataQualityError(f'Invalid numeric field: {exc}') from exc
    if source == 'akshare_em':
        frame['volume'] = frame['volume'] * 100
    _validate_values(frame)
    frame = frame.assign(asset_code=asset['code'], exchange=asset['exchange'], asset_name=asset['name'],
                         category=asset['category'], source=source, snapshot_id=snapshot_id, price_basis=price_basis)
    return frame[CORE].sort_values('date', kind='stable').reset_index(drop=True)


def combine_assets(frames: list[pd.DataFrame], assets: list[dict]) -> pd.DataFrame:
    """Concatenate a non-balanced panel; validate membership without reindex/fill."""
    expected = {(a['exchange'], a['code']): a for a in assets}
    if len(expected) != len(assets) or not frames:
        raise DataQualityError('Missing or duplicate expected asset')
    found = set()
    for frame in frames:
        if frame.empty or not set(CORE) <= set(frame):
            raise DataQualityError('Empty asset or missing identity fields')
        ids = set(zip(frame['exchange'], frame['asset_code']))
        if len(ids) != 1 or not ids <= expected.keys() or found & ids:
            raise DataQualityError('Mixed, duplicate or unexpected asset metadata')
        identity = next(iter(ids)); found.add(identity)
        asset = expected[identity]
        for col, key in (('asset_name','name'), ('category','category')):
            if not frame[col].eq(asset[key]).all():
                raise DataQualityError('Asset metadata mismatch')
        if frame[['source','snapshot_id','price_basis']].isna().any().any() or any(frame[c].astype(str).eq('').any() for c in ('source','snapshot_id')):
            raise DataQualityError('Missing source metadata')
        if not frame['price_basis'].eq('none').all():
            raise DataQualityError('Unapproved price basis')
        dates = date_strings(frame['date'])
        if dates.duplicated().any() or not dates.is_monotonic_increasing:
            raise DataQualityError('Asset dates must be unique and increasing')
        _validate_values(frame)
    if found != expected.keys():
        raise DataQualityError('Missing expected asset; refusing to publish partial panel')
    return pd.concat(frames, ignore_index=True)[CORE].sort_values(['date','exchange','asset_code'], kind='stable').reset_index(drop=True)

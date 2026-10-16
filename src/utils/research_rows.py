"""Shared development-only row and domestic-session contract; no financial features."""
from __future__ import annotations
import numpy as np
import pandas as pd
from src.data.cleaner import date_strings
from src.data.research_policy import require
from src.utils.config import validate_config

ASSET = ['exchange', 'asset_code']
KEYS = ['date', *ASSET, 'price_segment']


def validate_calendar(calendar: list[str], config: dict) -> list[str]:
    """Validate an explicitly supplied, ordered development-session calendar."""
    dates = date_strings(pd.Series(calendar, dtype=str))
    require(len(dates)>0 and dates.is_unique and dates.is_monotonic_increasing, 'Invalid session calendar order')
    require(dates.between(config['data']['start_date'], config['data']['end_date']).all(),
            'Calendar includes dates outside development')
    return dates.tolist()


def prepare_rows(panel: pd.DataFrame, config: dict, calendar: list[str]) -> pd.DataFrame:
    """Validate and canonically sort observed rows; mark causal calculation runs.

    An exchange-session gap starts a new calculation run, without creating any
    missing price row. Positions are audit metadata and never model features.
    """
    validate_config(config)
    calendar = validate_calendar(calendar, config)
    required = {*KEYS, 'asset_name', 'category', 'close', 'research_close', 'volume',
                'research_volume', 'volume_valid', 'price_basis', 'research_price_policy'}
    require(not panel.empty and panel.columns.is_unique and required <= set(panel), 'Missing research columns/rows')
    frame = panel.copy(deep=True)
    # pandas groupby drops null keys by default; reject them before any grouping.
    require(frame[ASSET].notna().all().all(), 'Missing asset identity')
    frame['date'] = date_strings(frame.date)
    require(frame.date.between(config['data']['start_date'], config['data']['end_date']).all(),
            'Holdout/out-of-development observations are forbidden')
    require(frame.date.isin(calendar).all(), 'Quote date is not in supplied domestic calendar')
    require(not frame.duplicated(['date', *ASSET]).any(), 'Duplicate asset date')
    require(frame.price_basis.eq('none').all() and frame.research_price_policy.eq('unadjusted_event_segmented_v1').all(),
            'Unapproved research price basis')
    for col in ['close', 'research_close', 'volume', 'research_volume', 'price_segment']:
        frame[col] = pd.to_numeric(frame[col], errors='raise')
    require(np.isfinite(frame[['close','research_close','volume','price_segment']]).all().all(), 'Nonfinite required field')
    require(frame.close.gt(0).all() and frame.research_close.eq(frame.close).all(), 'Research price must equal positive raw close')
    require(frame.volume.ge(0).all(), 'Negative raw volume')
    require(frame.price_segment.ge(0).all() and frame.price_segment.eq(np.floor(frame.price_segment)).all(), 'Invalid price segment')
    require(frame.volume_valid.map(lambda v: isinstance(v,(bool,np.bool_))).all(), 'volume_valid must be Boolean')
    frame['price_segment'] = frame.price_segment.astype('int64')
    valid_volume = frame.volume_valid & np.isfinite(frame.research_volume) & frame.research_volume.gt(0)
    require(frame.loc[valid_volume,'research_volume'].eq(frame.loc[valid_volume,'volume']).all(), 'Research volume was silently replaced')
    frame['_effective_volume'] = frame.research_volume.where(valid_volume, np.nan)
    frame = frame.sort_values(['date', *ASSET], kind='stable').reset_index(drop=True)
    expected = {(a['exchange'],a['code']): a for a in config['universe']['assets']}
    positions = dict(zip(calendar, range(len(calendar))))
    for name in ['_asset_position', '_segment_position', '_run_id', '_run_position']:
        frame[name] = 0
    for identity, group in frame.groupby(ASSET, sort=False):
        require(identity in expected, 'Unknown research asset')
        asset = expected[identity]
        require(group.asset_name.eq(asset['name']).all() and group.category.eq(asset['category']).all(), 'Asset metadata conflict')
        require(group.date.ge(asset['data_available_from']).all(), 'Prelisting/pre-observation values')
        require(group.price_segment.is_monotonic_increasing, 'Corporate-action segments must not reverse or recur')
        changed = group.price_segment.diff().ne(0) | group.date.map(positions).diff().ne(1)
        run = changed.cumsum()-1
        frame.loc[group.index, '_asset_position'] = np.arange(len(group))
        frame.loc[group.index, '_segment_position'] = group.groupby('price_segment',sort=False).cumcount()
        frame.loc[group.index, '_run_id'] = run
        frame.loc[group.index, '_run_position'] = group.groupby(run,sort=False).cumcount()
    return frame

"""Conservative research annotations, never adjusted prices or technical features."""
from __future__ import annotations

from datetime import date
from pathlib import PureWindowsPath, PurePosixPath
import re

import numpy as np
import pandas as pd

from src.data.cleaner import PRICES, combine_assets


class PolicyError(ValueError):
    """Evidence or research-data policy cannot support the requested use."""


def require(condition: bool, message: str) -> None:
    """Fail closed rather than infer missing policy decisions."""
    if not condition:
        raise PolicyError(message)


def relative_path(value: str) -> str:
    """Portable evidence paths must stay inside the project on Windows and POSIX."""
    require(isinstance(value, str) and bool(value), 'Missing relative path')
    for path in (PureWindowsPath(value), PurePosixPath(value)):
        require(not path.is_absolute() and not path.drive and '..' not in path.parts,
                'Path must stay inside project')
    return value


def iso_day(value: str) -> str:
    """Validate a date-only event/session label."""
    require(isinstance(value, str) and re.fullmatch(r'\d{4}-\d{2}-\d{2}', value) is not None,
            'Expected an ISO date')
    date.fromisoformat(value)
    return value


def validate_policy(policy: dict, assets: list[dict]) -> None:
    """Validate event identities, evidence references and information timing."""
    try:
        require(policy['schema_version'] == 1, 'Unsupported policy schema')
        require(policy['price_policy'] == 'unadjusted_event_segmented_v1', 'Unsupported price policy')
        ids = {(a['exchange'], a['code']) for a in assets}
        require(len(ids) == len(assets), 'Duplicate assets')
        evidence_ids = set()
        for record in policy['evidence']:
            eid = record['evidence_id']
            require(bool(eid) and eid not in evidence_ids, 'Duplicate/empty evidence identity')
            evidence_ids.add(eid)
            relative_path(record['path'])
            require(re.fullmatch(r'[0-9a-f]{64}', record['sha256']) is not None, 'Invalid evidence SHA-256')
            require(record['url'].startswith('https://'), 'Evidence requires its public source URL')
        def references(record: dict) -> None:
            refs = record['evidence_ids']
            require(isinstance(refs, list) and bool(refs) and set(refs) <= evidence_ids,
                    'Missing or unknown evidence reference')
        reviews = policy['asset_reviews']
        require(len(reviews) == len(ids) and {(r['exchange'], r['asset_code']) for r in reviews} == ids,
                'Every frozen asset needs an event review')
        for record in reviews:
            require(record['status'] == 'reviewed', 'Unresolved asset action review')
            references(record)
            asset = next(a for a in assets if (a['exchange'], a['code']) == (record['exchange'], record['asset_code']))
            require(iso_day(record['covered_start']) <= max(policy['development']['start'], asset['data_available_from'])
                    and iso_day(record['covered_end']) >= policy['development']['end'], 'Incomplete action review window')
            cash_count = sum(e['kind'] == 'cash_dividend' and (e['exchange'], e['asset_code']) ==
                             (record['exchange'], record['asset_code']) and
                             policy['development']['start'] <= e['price_effective_date'] <= policy['development']['end']
                             for e in policy['events'])
            require(record['cash_event_count'] == cash_count, 'Cash events disagree with completeness audit')
        seen = set()
        for event in policy['events']:
            require((event['exchange'], event['asset_code']) in ids, 'Unknown event asset')
            require(event['event_id'] not in seen, 'Duplicate event identity')
            seen.add(event['event_id'])
            require(event['kind'] in ('cash_dividend', 'split', 'conversion'), 'Unknown corporate action')
            announced = iso_day(event['announced_date'])
            action = iso_day(event['action_date'])
            effective = iso_day(event['price_effective_date'])
            require(announced <= action <= effective, 'Event timing is late or inconsistent')
            references(event)
        seen = set()
        for anomaly in policy['volume_anomalies']:
            key = (anomaly['exchange'], anomaly['asset_code'], iso_day(anomaly['date']))
            require(key[:2] in ids and key not in seen, 'Unknown/duplicate volume anomaly')
            seen.add(key)
            require(bool(anomaly['reason'].strip()), 'Missing anomaly reason')
            references(anomaly)
        dev, hold = policy['development'], policy['holdout']
        require(iso_day(dev['start']) <= iso_day(dev['end']) < iso_day(hold['start']) <= iso_day(hold['end']),
                'Development and holdout dates must be disjoint and ordered')
        require(hold['status'] == 'sealed_candidate', 'Holdout must remain sealed')
    except (KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, PolicyError):
            raise
        raise PolicyError(f'Malformed research policy: {exc}') from exc


def apply_policy(panel: pd.DataFrame, policy: dict, assets: list[dict]) -> pd.DataFrame:
    """Keep all observed rows; add event regimes and missing research-volume values.

    Research OHLC equals unadjusted OHLC. Segment identifiers are grouping metadata,
    not predictors; downstream windows must stay within asset AND segment. No return,
    moving average, oscillator or other technical feature is computed here.
    """
    validate_policy(policy, assets)
    frames = [panel.loc[(panel.exchange == a['exchange']) & (panel.asset_code == a['code'])].copy()
              for a in assets]
    require(sum(len(f) for f in frames) == len(panel), 'Unexpected asset observations')
    result = combine_assets(frames, assets)
    for asset in assets:
        dates = result.loc[(result.exchange == asset['exchange']) & (result.asset_code == asset['code']), 'date']
        require(not (dates < asset['data_available_from']).any(), 'Rows precede registered available history')
    for col in PRICES:
        result['research_'+col] = result[col]
    result['price_segment'] = 0
    result['price_event'] = False
    for event in policy['events']:
        same_asset = (result.exchange == event['exchange']) & (result.asset_code == event['asset_code'])
        eligible = same_asset & (result.date >= event['price_effective_date'])
        result.loc[eligible, 'price_segment'] += 1
        # If the action date has no price (e.g. a suspension), mark the first observed session after it.
        indices = result.index[eligible]
        if len(indices):
            result.loc[indices[0], 'price_event'] = True
    result['volume_valid'] = np.isfinite(result.volume) & (result.volume > 0)
    result['volume_anomaly'] = np.where(result.volume_valid, '', 'nonpositive_or_nonfinite_volume')
    for anomaly in policy['volume_anomalies']:
        mask = ((result.exchange == anomaly['exchange']) & (result.asset_code == anomaly['asset_code'])
                & (result.date == anomaly['date']))
        result.loc[mask, 'volume_valid'] = False
        result.loc[mask, 'volume_anomaly'] = anomaly['reason']
    result['research_volume'] = result.volume.where(result.volume_valid, np.nan)
    result['research_price_policy'] = policy['price_policy']
    return result

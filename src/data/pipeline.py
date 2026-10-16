"""Probe, publish and replay data-only experiments from immutable input snapshots."""
from __future__ import annotations
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Callable
from uuid import uuid4

import pandas as pd

from src.data.cleaner import clean_prices, combine_assets, date_strings
from src.data.contracts import DataRequest
from src.data.loader import SnapshotStore, digest, json_bytes, write_bytes, revision_details
from src.data.quality import audit_panel
from src.utils.config import config_digest, resolve_paths, validate_config

Fetch = Callable[[DataRequest], pd.DataFrame]


class PipelineError(RuntimeError):
    """A failed gate or corrupted artifact cannot be called a complete dataset."""


def read_audit(path: Path) -> dict:
    """Read a manifest only if its detached checksum still matches."""
    try:
        payload = path.read_bytes()
        if digest(payload) != path.with_suffix('.sha256').read_text(encoding='ascii').strip():
            raise PipelineError(f'Audit hash mismatch: {path}')
        return json.loads(payload)
    except (OSError, ValueError) as exc:
        raise PipelineError(f'Cannot read audit {path}: {exc}') from exc


def _audit(path: Path, report: dict) -> Path:
    payload = json_bytes(report)
    write_bytes(path, payload)
    write_bytes(path.with_suffix('.sha256'), (digest(payload)+'\n').encode('ascii'))
    return path


def _run_dir(config: dict, root: Path) -> Path:
    validate_config(config)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'-'+uuid4().hex[:8]
    target = resolve_paths(config, root)['results']/'reports'/'data'/stamp
    target.mkdir(parents=True)
    return target


def _request(config: dict, asset: dict, purpose: str) -> DataRequest:
    cfg = config['data']['download']
    start, end = (cfg['probe_start_date'], cfg['probe_end_date']) if purpose == 'probe' else (cfg['coverage_start_date'], cfg['as_of_date'])
    return DataRequest(asset['code'], asset['exchange'], cfg['source'], start, end, purpose=purpose)


def _input(snapshot, root: Path) -> dict:
    return {'path': snapshot.path.relative_to(root).as_posix(), 'snapshot_id': snapshot.snapshot_id,
            'manifest_sha256': digest((snapshot.path/'manifest.json').read_bytes()), 'cache_hit': snapshot.cache_hit}


def run_probe(config: dict, root: Path, fetch: Fetch, *, refresh: bool = False) -> Path:
    """Probe every candidate explicitly; a failed/empty sample blocks publication."""
    root = Path(root).resolve(); out = _run_dir(config, root)
    paths = resolve_paths(config, root); store = SnapshotStore(paths['raw'], paths['cache'])
    rows = []
    for asset in config['universe']['assets']:
        record = {'asset_code': asset['code'], 'exchange': asset['exchange'], 'status': 'failed'}
        try:
            request = _request(config, asset, 'probe')
            snapshot = store.get(request, fetch, refresh=refresh)
            clean = clean_prices(snapshot.frame, asset, request.source, snapshot.snapshot_id)
            sample = clean.loc[clean.date.between(request.start_date, request.end_date)]
            if sample.empty:
                raise PipelineError('Probe window has no observations; asset coverage remains unknown')
            record.update(status='ok', input=_input(snapshot, root), sample_rows=len(sample),
                          first_available=str(clean.date.min()), last_available=str(clean.date.max()), total_returned_rows=len(clean))
        except Exception as exc:
            record['error'] = f'{type(exc).__name__}: {exc}'
        rows.append(record)
        time.sleep(config['data']['download']['request_pause_seconds'])
    return _audit(out/'probe.json', {'status': 'complete' if all(r['status']=='ok' for r in rows) else 'failed',
                  'kind': 'probe', 'config_sha256': config_digest(config), 'source': config['data']['download']['source'],
                  'created_at': datetime.now(timezone.utc).isoformat(), 'assets': rows})


def _verify_probe(config: dict, root: Path, path: Path, store: SnapshotStore) -> dict:
    probe = read_audit(Path(path))
    expected = {(a['exchange'], a['code']) for a in config['universe']['assets']}
    seen = {(a['exchange'], a['asset_code']) for a in probe.get('assets', [])}
    if probe.get('kind') != 'probe' or probe.get('status') != 'complete' or probe.get('config_sha256') != config_digest(config) or seen != expected or len(probe['assets']) != len(expected):
        raise PipelineError('A successful matching probe is required before formal download')
    for entry in probe['assets']:
        snapshot = store.read(root/entry['input']['path'], entry['input']['manifest_sha256'])
        asset = next(a for a in config['universe']['assets'] if (a['exchange'], a['code']) == (entry['exchange'], entry['asset_code']))
        if snapshot.manifest['request'] != asdict(_request(config, asset, 'probe')) or entry['status'] != 'ok':
            raise PipelineError('Invalid probe asset binding')
    return probe


def _slice(snapshot, asset: dict, config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    request = DataRequest(**snapshot.manifest['request'])
    if (request.asset_code, request.exchange) != (asset['code'], asset['exchange']):
        raise PipelineError('Snapshot is bound to a different asset')
    clean = clean_prices(snapshot.frame, asset, request.source, snapshot.snapshot_id, price_basis=request.adjustment)
    data = config['data']; cfg = data['download']
    available = clean.loc[clean.date.between(cfg['coverage_start_date'], cfg['as_of_date'])].reset_index(drop=True)
    research = available.loc[available.date.between(data['start_date'], data['end_date'])].reset_index(drop=True)
    if research.empty:
        raise PipelineError('No observations in research window; do not drop or fabricate the asset')
    return available, research


def _panel_bytes(panel: pd.DataFrame) -> bytes:
    return panel.to_csv(index=False, float_format='%.17g', lineterminator='\n').encode('utf-8')


def run_download(config: dict, root: Path, probe_path: Path, fetch: Fetch, *, refresh: bool = False) -> Path:
    """Fetch coverage histories, retain their early years, publish a bounded research panel."""
    root = Path(root).resolve(); paths = resolve_paths(config, root)
    store = SnapshotStore(paths['raw'], paths['cache'])
    probe = _verify_probe(config, root, probe_path, store)
    out = _run_dir(config, root)
    report = {'status': 'failed', 'kind': 'download', 'created_at': datetime.now(timezone.utc).isoformat(),
              'config_sha256': config_digest(config), 'assets': [], 'dataset_manifest': None}
    frames, coverage_frames, inputs = [], [], []
    for asset in config['universe']['assets']:
        entry = {'asset_code': asset['code'], 'exchange': asset['exchange'], 'status': 'failed'}
        try:
            snapshot = store.get(_request(config, asset, 'coverage'), fetch, refresh=refresh)
            probe_entry = next(p for p in probe['assets'] if (p['exchange'], p['asset_code']) == (asset['exchange'], asset['code']))
            previous = store.read(root/probe_entry['input']['path'], probe_entry['input']['manifest_sha256'])
            comparison = revision_details(previous.frame, snapshot.frame)
            comparison['scope_note'] = 'full response; added/removed dates may reflect differing API request windows'
            available, research = _slice(snapshot, asset, config)
            frames.append(research); coverage_frames.append(available); inputs.append(_input(snapshot, root))
            entry.update(status='ok', source=config['data']['download']['source'], input=inputs[-1],
                         available_first=available.date.min(), available_last=available.date.max(), available_rows=len(available),
                         research_first=research.date.min(), research_last=research.date.max(), research_rows=len(research),
                         revision=snapshot.manifest['revision'], revision_since_probe=comparison,
                         source_dates_monotonic=bool(snapshot.frame[next(c for c in ('date','日期') if c in snapshot.frame)].is_monotonic_increasing),
                         extra_source_rows_outside_coverage=len(snapshot.frame)-len(available))
        except Exception as exc:
            entry['error'] = f'{type(exc).__name__}: {exc}'
        report['assets'].append(entry)
        time.sleep(config['data']['download']['request_pause_seconds'])
    if any(row['status'] != 'ok' for row in report['assets']):
        return _audit(out/'download.json', report)
    try:
        cfg = config['data']['download']
        calendar_request = DataRequest('CALENDAR','CN','akshare_calendar',cfg['coverage_start_date'],cfg['as_of_date'],purpose='calendar')
        calendar_snapshot = store.get(calendar_request, fetch, refresh=refresh)
        calendar = date_strings(calendar_snapshot.frame['trade_date']).tolist()
        panel = combine_assets(frames, config['universe']['assets'])
        coverage = combine_assets(coverage_frames, config['universe']['assets'])
        audit_args = {'vwap_tolerance': cfg['amount_price_tolerance'], 'price_break_threshold': cfg['suspicious_price_change_fraction']}
        audit = audit_panel(panel, calendar, config['universe']['assets'], config['data']['start_date'], config['data']['end_date'], **audit_args)
        coverage_audit = audit_panel(coverage, calendar, config['universe']['assets'], max(cfg['coverage_start_date'],min(calendar)), cfg['as_of_date'], **audit_args)
        latest_session = max(d for d in calendar if d <= cfg['as_of_date'])
        for entry in report['assets']:
            entry['latest_expected_session'] = latest_session
            entry['continuous_quotes_as_of'] = entry['available_last'] == latest_session
        parent = paths['processed']/'unadjusted'; parent.mkdir(parents=True, exist_ok=True)
        pending = parent/('.pending-'+out.name); pending.mkdir()
        payload = _panel_bytes(panel); coverage_payload = _panel_bytes(coverage)
        write_bytes(pending/'prices.csv', payload)
        write_bytes(pending/'coverage_prices.csv', coverage_payload)
        write_bytes(pending/'quality.json', json_bytes(audit))
        write_bytes(pending/'coverage_quality.json', json_bytes(coverage_audit))
        manifest = {'status': 'complete', 'kind': 'unadjusted_panel', 'schema_version': 1,
                    'created_at': datetime.now(timezone.utc).isoformat(), 'config': config,
                    'config_sha256': config_digest(config), 'inputs': inputs, 'calendar_input': _input(calendar_snapshot,root),
                    'price_policy': 'unadjusted observations only; feature price policy pending',
                    'research_ready': False, 'research_blockers': ['research_price_policy_pending', 'review_quality_reports_before_features'],
                    'files': {p.name: digest(p.read_bytes()) for p in pending.iterdir()},
                    'rows': len(panel), 'coverage_rows': len(coverage), 'cleaner_version': 1}
        _audit(pending/'manifest.json', manifest)
        final = parent/out.name; pending.rename(final)
        report.update(status='complete', dataset_manifest=(final/'manifest.json').relative_to(root).as_posix(),
                      dataset_sha256=digest(payload), dataset_rows=len(panel), coverage_rows=len(coverage))
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
    return _audit(out/'download.json', report)


def replay_dataset(manifest_path: Path, root: Path) -> pd.DataFrame:
    """Verify and rebuild the exact dataset offline; no provider function is accepted."""
    root = Path(root).resolve(); manifest_path = Path(manifest_path).resolve()
    meta = read_audit(manifest_path)
    if meta.get('status') != 'complete' or meta.get('kind') != 'unadjusted_panel' or meta.get('cleaner_version') != 1:
        raise PipelineError('Unsupported/incomplete dataset')
    config = meta['config']; paths = resolve_paths(config, root)
    if not manifest_path.is_relative_to(paths['processed']) or any(p.startswith('.pending-') for p in manifest_path.parts):
        raise PipelineError('Dataset path is outside processed storage or incomplete')
    if meta['config_sha256'] != config_digest(config):
        raise PipelineError('Config hash mismatch')
    for name, expected in meta['files'].items():
        if Path(name).name != name or digest((manifest_path.parent/name).read_bytes()) != expected:
            raise PipelineError('Dataset file hash mismatch')
    store = SnapshotStore(paths['raw'], paths['cache']); frames, coverage_frames = [], []
    if len(meta['inputs']) != len(config['universe']['assets']):
        raise PipelineError('Dataset has missing asset inputs')
    for asset, entry in zip(config['universe']['assets'], meta['inputs'], strict=True):
        snapshot = store.read(root/entry['path'], entry['manifest_sha256'])
        available, research = _slice(snapshot, asset, config)
        frames.append(research); coverage_frames.append(available)
    calendar_entry = meta['calendar_input']
    calendar = store.read(root/calendar_entry['path'], calendar_entry['manifest_sha256'])
    panel = combine_assets(frames,config['universe']['assets'])
    cfg = config['data']['download']
    quality = audit_panel(panel, calendar.frame.trade_date.tolist(), config['universe']['assets'], config['data']['start_date'], config['data']['end_date'],
                          vwap_tolerance=cfg['amount_price_tolerance'], price_break_threshold=cfg['suspicious_price_change_fraction'])
    if digest(_panel_bytes(panel)) != meta['files']['prices.csv'] or digest(json_bytes(quality)) != meta['files']['quality.json']:
        raise PipelineError('Rebuilt dataset/quality hash differs from published artifact')
    coverage = combine_assets(coverage_frames,config['universe']['assets'])
    if digest(_panel_bytes(coverage)) != meta['files']['coverage_prices.csv']:
        raise PipelineError('Rebuilt coverage hash differs')
    return panel

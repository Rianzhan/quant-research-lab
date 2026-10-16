"""Immutable SDK response snapshots; a complete manifest is the publication gate."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib.metadata import version
import hashlib
import io
import json
import logging
import os
from pathlib import Path
from typing import Callable
from uuid import uuid4

import pandas as pd

from src.data.cleaner import DataQualityError, clean_prices, date_strings
from src.data.contracts import DataRequest

LOG = logging.getLogger(__name__)


class SnapshotError(RuntimeError):
    """A network, integrity or publication failure; never silently redownload."""


def digest(payload: bytes) -> str:
    """SHA-256 of the exact saved file bytes."""
    return hashlib.sha256(payload).hexdigest()


def json_bytes(value: dict) -> bytes:
    """Stable UTF-8 JSON; NaN is forbidden in audit manifests."""
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)+'\n').encode('utf-8')


def write_bytes(path: Path, payload: bytes) -> None:
    """Create a new file exclusively and flush before publishing its directory."""
    with path.open('xb') as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


@dataclass
class Snapshot:
    """A verified raw response with an immutable location and request binding."""
    path: Path
    manifest: dict
    frame: pd.DataFrame
    cache_hit: bool = False

    @property
    def snapshot_id(self) -> str:
        return self.manifest['snapshot_id']


def revision_details(old: pd.DataFrame, new: pd.DataFrame) -> dict:
    """Report changed observation dates without silently reconciling revisions."""
    date_column = next((c for c in ('date', '日期', 'trade_date') if c in old and c in new), None)
    same = old.equals(new)
    result = {'changed': not same, 'schema_changed': list(old.columns) != list(new.columns),
              'old_rows': len(old), 'new_rows': len(new), 'added_dates': [], 'removed_dates': [], 'changed_dates': []}
    if date_column is not None:
        old_dates, new_dates = set(old[date_column]), set(new[date_column])
        result['added_dates'] = sorted(new_dates-old_dates)
        result['removed_dates'] = sorted(old_dates-new_dates)
        for day in sorted(old_dates & new_dates):
            left = old.loc[old[date_column] == day].reset_index(drop=True)
            right = new.loc[new[date_column] == day].reset_index(drop=True)
            if not left.equals(right):
                result['changed_dates'].append(day)
    return result


def validate_response(frame: pd.DataFrame, request: DataRequest) -> None:
    """Gate cache publication on structure and values, without altering the raw response.

    Adjustment probes receive only structural checks here, never research approval.
    Market-calendar gaps and event anomalies remain downstream audit responsibilities.
    """
    if frame.empty or frame.columns.duplicated().any():
        raise DataQualityError('Empty response or duplicate fields')
    if request.source == 'akshare_calendar':
        if 'trade_date' not in frame:
            raise DataQualityError('Calendar is missing trade_date')
        dates = date_strings(frame['trade_date'])
        if dates.duplicated().any() or not dates.is_monotonic_increasing:
            raise DataQualityError('Calendar dates must be unique and increasing')
        return
    identity = {'code': request.asset_code, 'exchange': request.exchange,
                'name': request.asset_code, 'category': 'validation_only'}
    checked = clean_prices(frame, identity, request.source, 'validation_only')
    if not request.interface()['returns_full_history']:
        if not checked.date.between(request.start_date, request.end_date).all():
            raise DataQualityError('Provider returned dates outside requested range')


class SnapshotStore:
    """One immutable directory per response; only a small cache pointer is replaceable."""

    def __init__(self, raw_root: Path, cache_root: Path):
        self.raw_root = Path(raw_root).resolve()
        self.cache_root = Path(cache_root).resolve()
        self.raw_root.mkdir(parents=True, exist_ok=True)
        (self.cache_root/'requests').mkdir(parents=True, exist_ok=True)
        (self.cache_root/'failures').mkdir(parents=True, exist_ok=True)

    def read(self, path: Path, expected_manifest_hash: str | None = None) -> Snapshot:
        """Refuse partial, relocated or modified snapshots before parsing prices."""
        try:
            path = Path(path).resolve()
            if not path.is_relative_to(self.raw_root) or any(p.startswith('.pending-') for p in path.parts):
                raise SnapshotError('Snapshot path is outside the store or incomplete')
            meta_bytes = (path/'manifest.json').read_bytes()
            checksum = (path/'manifest.sha256').read_text(encoding='ascii').strip()
            if digest(meta_bytes) != checksum or (expected_manifest_hash is not None and checksum != expected_manifest_hash):
                raise SnapshotError('Manifest hash mismatch')
            manifest = json.loads(meta_bytes)
            payload = (path/'raw.csv').read_bytes()
            if manifest['status'] != 'complete' or digest(payload) != manifest['raw_sha256']:
                raise SnapshotError('Raw snapshot hash mismatch or incomplete snapshot')
            req = DataRequest(**manifest['request'])
            if path.name != manifest['snapshot_id'] or path.parent.name != req.key:
                raise SnapshotError('Snapshot request/path identity mismatch')
            frame = pd.read_csv(io.BytesIO(payload), dtype=str, keep_default_na=False)
            if len(frame) != manifest['row_count'] or list(frame.columns) != manifest['columns']:
                raise SnapshotError('Snapshot row count or schema mismatch')
            validate_response(frame, req)
            return Snapshot(path, manifest, frame)
        except (OSError, ValueError, KeyError, pd.errors.ParserError) as exc:
            raise SnapshotError(f'Cannot read snapshot {path}: {exc}') from exc

    def _cached(self, request: DataRequest) -> Snapshot | None:
        pointer = self.cache_root/'requests'/f'{request.key}.json'
        if not pointer.exists():
            return None
        try:
            index = json.loads(pointer.read_bytes())
            relative = Path(index['relative_path'])
            if relative.is_absolute() or '..' in relative.parts:
                raise SnapshotError('Invalid cache pointer path')
            result = self.read(self.raw_root/relative, index['manifest_sha256'])
            if result.manifest['request'] != asdict(request):
                raise SnapshotError('Cache request identity mismatch')
            result.cache_hit = True
            return result
        except (OSError, ValueError, KeyError) as exc:
            raise SnapshotError(f'Invalid cache pointer: {exc}') from exc

    def _quarantine(self, request: DataRequest, snapshot_id: str, payload: bytes,
                    started_at: datetime, reason: str) -> str:
        """Keep rejected provider bytes outside complete snapshots and cache pointers."""
        parent = self.raw_root/'quarantine'/request.key
        parent.mkdir(parents=True, exist_ok=True)
        pending = parent/('.pending-'+snapshot_id)
        pending.mkdir()
        write_bytes(pending/'raw.csv', payload)
        rejected = {'status': 'quarantined', 'attempt_id': snapshot_id,
                    'download_started_at': started_at.isoformat(),
                    'recorded_at': datetime.now(timezone.utc).isoformat(),
                    'request': asdict(request), 'interface': request.interface(),
                    'raw_sha256': digest(payload), 'error': reason}
        metadata = json_bytes(rejected)
        write_bytes(pending/'rejected.json', metadata)
        write_bytes(pending/'rejected.sha256', (digest(metadata)+'\n').encode('ascii'))
        final = parent/snapshot_id
        pending.rename(final)
        return final.relative_to(self.raw_root).as_posix()

    def get(self, request: DataRequest, fetch: Callable[[DataRequest], pd.DataFrame], *, refresh: bool = False) -> Snapshot:
        """Fetch explicitly, or reuse a verified snapshot without calling the provider."""
        previous = self._cached(request)
        if previous is not None and not refresh:
            return previous
        stamp = datetime.now(timezone.utc)
        snapshot_id = stamp.strftime('%Y%m%dT%H%M%S%fZ')+'-'+uuid4().hex[:12]
        quarantine_path = None
        try:
            frame = fetch(request)
            if not isinstance(frame, pd.DataFrame) or frame.empty:
                raise SnapshotError('Provider returned an empty response; coverage is unknown')
            payload = frame.to_csv(index=False, lineterminator='\n', float_format='%.17g').encode('utf-8')
            try:
                if frame.columns.duplicated().any():
                    raise DataQualityError('Provider returned duplicate field names')
                reread = pd.read_csv(io.BytesIO(payload), dtype=str, keep_default_na=False)
                validate_response(reread, request)
            except (ValueError, pd.errors.ParserError) as exc:
                quarantine_path = self._quarantine(request, snapshot_id, payload, stamp, str(exc))
                raise
            revision = revision_details(previous.frame, reread) if previous else None
            manifest = {
                'schema_version': 1, 'status': 'complete', 'snapshot_id': snapshot_id,
                'download_started_at': stamp.isoformat(), 'downloaded_at': datetime.now(timezone.utc).isoformat(),
                'request': asdict(request), 'interface': request.interface(), 'row_count': len(frame),
                'columns': list(frame.columns), 'raw_sha256': digest(payload),
                'serialization': 'SDK-returned DataFrame CSV UTF-8; not original HTTP wire bytes',
                'package_version': version('baostock' if request.source == 'baostock' else 'akshare'),
                'provider_notes': frame.attrs.get('provider_notes', []),
                'units': {'ohlc': 'CNY per ETF share', 'volume': 'lots of 100 shares' if request.source == 'akshare_em' else 'shares', 'amount': 'CNY if supplied'},
                'previous_snapshot_id': previous.snapshot_id if previous else None, 'revision': revision,
            }
            parent = self.raw_root/request.key
            parent.mkdir(exist_ok=True)
            pending = parent/('.pending-'+snapshot_id)
            pending.mkdir()
            write_bytes(pending/'raw.csv', payload)
            meta_bytes = json_bytes(manifest)
            write_bytes(pending/'manifest.json', meta_bytes)
            write_bytes(pending/'manifest.sha256', (digest(meta_bytes)+'\n').encode('ascii'))
            final = parent/snapshot_id
            pending.rename(final)
            result = self.read(final)
            pointer = {'relative_path': final.relative_to(self.raw_root).as_posix(), 'manifest_sha256': digest(meta_bytes)}
            temporary = self.cache_root/'requests'/f'.{uuid4().hex}.tmp'
            write_bytes(temporary, json_bytes(pointer))
            os.replace(temporary, self.cache_root/'requests'/f'{request.key}.json')
            if revision and revision['changed']:
                LOG.warning('History revision detected for %s: %s', request.asset_code, revision)
            return result
        except Exception as exc:
            failure = {'status': 'failed', 'attempt_id': snapshot_id, 'request': asdict(request),
                       'recorded_at': datetime.now(timezone.utc).isoformat(), 'error_type': type(exc).__name__,
                       'error': str(exc), 'quarantine_path': quarantine_path}
            try:
                write_bytes(self.cache_root/'failures'/f'{snapshot_id}.json', json_bytes(failure))
            except OSError:
                LOG.exception('Could not save failure audit record')
            raise SnapshotError(f'{request.source} {request.asset_code}: {exc}') from exc

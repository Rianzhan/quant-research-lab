"""Publish and verify a conservative development view; never release the holdout."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from uuid import uuid4

import pandas as pd

from src.data.loader import digest, json_bytes, write_bytes
from src.data.pipeline import PipelineError, read_audit, replay_dataset
from src.data.research_policy import PolicyError, apply_policy, relative_path, require, validate_policy
from src.utils.config import config_digest, load_config, validate_config


def _inside(root: Path, relative: str) -> Path:
    """Resolve portable project paths, including a symlink containment check."""
    path = (root/relative_path(relative)).resolve()
    require(path.is_relative_to(root) and path != root, 'Path escapes project')
    require(not any(part.startswith('.pending-') for part in path.parts), 'Incomplete publication')
    return path


def _policy(config: dict, root: Path) -> dict:
    validate_config(config)
    require(config['schema_version'] == 2, 'Research requires schema version 2')
    g = config['governance']
    payload = _inside(root, g['policy_file']).read_bytes()
    require(digest(payload) == g['policy_sha256'], 'Policy hash mismatch')
    policy = json.loads(payload)
    validate_policy(policy, config['universe']['assets'])
    require(policy['development'] == {'start': config['data']['start_date'], 'end': config['data']['end_date']},
            'Development window differs between policy and config')
    require(policy['holdout'] == {'start': g['holdout_start_date'], 'end': g['holdout_end_date'],
                                  'status': g['holdout_status']}, 'Holdout window differs')
    for evidence in policy['evidence']:
        require(digest(_inside(root, evidence['path']).read_bytes()) == evidence['sha256'],
                'Evidence hash mismatch: '+evidence['evidence_id'])
    return policy


def partition_windows(panel: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select by date before applying research logic; retain older history in Phase 2."""
    data, g = config['data'], config['governance']
    development = panel.loc[panel.date.between(data['start_date'], data['end_date'])].copy().reset_index(drop=True)
    candidate = panel.loc[panel.date.between(g['holdout_start_date'], g['holdout_end_date'])].copy().reset_index(drop=True)
    candidate['dataset_role'] = 'final_holdout_candidate'
    candidate['research_use_allowed'] = False
    return development, candidate


def _csv(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, float_format='%.17g', lineterminator='\n').encode('utf-8')


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={'date': str, 'asset_code': str}, float_precision='round_trip',
                       keep_default_na=False, na_values=[''])


def _source(config: dict, root: Path) -> tuple[Path, dict]:
    path = _inside(root, config['governance']['phase2_manifest'])
    meta = read_audit(path)
    require(meta.get('kind') == 'unadjusted_panel' and meta.get('status') == 'complete', 'Incomplete Phase 2 input')
    original = meta['config']['universe']['assets']
    current = config['universe']['assets']
    fields = ('code', 'exchange', 'name', 'category', 'selection_reason')
    require([{k: a[k] for k in fields} for a in original] == [{k: a[k] for k in fields} for a in current],
            'Frozen assets differ from pre-registered Phase 2 candidates')
    return path, meta


def _verify_raw_chain(meta: dict, root: Path) -> None:
    """Verify immutable raw bytes and calendar evidence without interpreting prices."""
    for entry in [*meta['inputs'], meta['calendar_input']]:
        manifest = _inside(root, entry['path']+'/manifest.json')
        raw = _inside(root, entry['path']+'/raw.csv')
        require(digest(manifest.read_bytes()) == entry['manifest_sha256'], 'Raw manifest hash changed')
        snapshot = read_audit(manifest)
        require(snapshot['status'] == 'complete' and snapshot['snapshot_id'] == entry['snapshot_id'],
                'Incomplete or mismatched raw snapshot')
        require(digest(raw.read_bytes()) == snapshot['raw_sha256'], 'Raw snapshot bytes changed')


def publish_research_dataset(config: dict, root: Path) -> Path:
    """Validate evidence and Phase 2 bytes, then atomically publish an immutable view.

    On failure a uniquely named pending directory can remain for diagnosis; it is
    never a valid manifest path. Existing releases and raw observations are untouched.
    """
    root = Path(root).resolve()
    policy = _policy(config, root)
    target = _inside(root, config['governance']['dataset_manifest'])
    require(target.name == 'manifest.json', 'Expected manifest.json publication path')
    require(not target.parent.exists(), 'Publication already exists; use a new version')
    source, source_meta = _source(config, root)
    original_development = replay_dataset(source, root)  # offline hashes and development-only quality
    coverage = _read_csv(source.parent/'coverage_prices.csv')
    development, candidate = partition_windows(coverage, config)
    require(digest(_csv(development)) == digest(_csv(original_development)), 'Development window/source mismatch')
    research = apply_policy(development, policy, config['universe']['assets'])
    require(not candidate.empty, 'Expected existing final holdout candidate observations')
    parent = target.parent.parent
    parent.mkdir(parents=True, exist_ok=True)
    pending = parent/('.pending-'+uuid4().hex)
    pending.mkdir()
    write_bytes(pending/'development.csv', _csv(research))
    write_bytes(pending/'holdout_candidate.csv', _csv(candidate))
    meta = {
        'kind': 'research_policy_dataset', 'status': 'complete', 'transform_version': 1,
        'created_at': datetime.now(timezone.utc).isoformat(),
        'config_sha256': config_digest(config), 'policy_sha256': config['governance']['policy_sha256'],
        'universe_sha256': config['universe']['freeze_sha256'],
        'phase2_manifest': config['governance']['phase2_manifest'], 'phase2_manifest_sha256': digest(source.read_bytes()),
        'scope': 'development_price_features', 'holdout_status': 'sealed_candidate',
        'backtest_total_return_ready': False,
        'rows': {'development': len(research), 'holdout_candidate': len(candidate)},
        'files': {name: digest((pending/name).read_bytes()) for name in ('development.csv', 'holdout_candidate.csv')},
    }
    payload = json_bytes(meta)
    write_bytes(pending/'manifest.json', payload)
    write_bytes(pending/'manifest.sha256', (digest(payload)+'\n').encode('ascii'))
    pending.rename(target.parent)
    return target


def _verified(config: dict, root: Path) -> tuple[dict, Path]:
    """Fail closed on changed policy, evidence, input snapshot or published bytes."""
    _policy(config, root)
    manifest = _inside(root, config['governance']['dataset_manifest'])
    meta = read_audit(manifest)
    require(meta.get('kind') == 'research_policy_dataset' and meta.get('status') == 'complete'
            and meta.get('transform_version') == 1, 'Unsupported/incomplete research dataset')
    require(meta['config_sha256'] == config_digest(config), 'Research configuration hash changed')
    require(meta['policy_sha256'] == config['governance']['policy_sha256'], 'Dataset policy hash changed')
    require(meta['universe_sha256'] == config['universe']['freeze_sha256'], 'Dataset universe hash changed')
    source, source_meta = _source(config, root)
    require(meta['phase2_manifest'] == config['governance']['phase2_manifest']
            and meta['phase2_manifest_sha256'] == digest(source.read_bytes()), 'Phase 2 source hash changed')
    _verify_raw_chain(source_meta, root)
    for name, sha in source_meta['files'].items():
        require(Path(name).name == name and digest((source.parent/name).read_bytes()) == sha, 'Phase 2 file hash changed')
    require(set(meta['files']) == {'development.csv', 'holdout_candidate.csv'}, 'Dataset file set changed')
    for name, sha in meta['files'].items():
        require(digest((manifest.parent/name).read_bytes()) == sha, 'Research file hash mismatch: '+name)
    require(meta['scope'] == 'development_price_features' and meta['holdout_status'] == 'sealed_candidate'
            and meta['backtest_total_return_ready'] is False, 'Research scope or holdout state changed')
    frame = _read_csv(manifest.parent/'development.csv')
    require(len(frame) == meta['rows']['development'] and not frame.empty, 'Development row count mismatch')
    require(frame.date.between(config['data']['start_date'], config['data']['end_date']).all(), 'Development date leakage')
    ids = {(a['exchange'], a['code']) for a in config['universe']['assets']}
    require(set(zip(frame.exchange, frame.asset_code)) == ids, 'Development asset identity mismatch')
    return meta, manifest


def check_research_gate(config: dict, root: Path) -> dict:
    """Return a scoped verification result; no Boolean configuration can open it."""
    result = {'research_ready': False, 'scope': 'development_price_features',
              'holdout_status': 'sealed_candidate', 'backtest_total_return_ready': False, 'blockers': []}
    try:
        meta, _ = _verified(config, Path(root).resolve())
        result.update(research_ready=True, rows=meta['rows'])
    except (OSError, ValueError, KeyError, TypeError, PipelineError) as exc:
        result['blockers'].append(f'{type(exc).__name__}: {exc}')
    return result


def load_development(config: dict, root: Path) -> pd.DataFrame:
    """Only approved 2018–2025 observations; there is deliberately no holdout loader."""
    gate = check_research_gate(config, root)
    require(gate['research_ready'], '; '.join(gate['blockers']))
    return _read_csv(_inside(Path(root).resolve(), config['governance']['dataset_manifest']).parent/'development.csv')


def main(argv: list[str] | None = None) -> int:
    """Offline publication/check commands only, without research computations."""
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('publish', 'check'))
    parser.add_argument('--config', type=Path, default=root/'config/config.yaml')
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        if args.command == 'publish':
            print(str(publish_research_dataset(config, root)))
        gate = check_research_gate(config, root)
        print(json.dumps(gate, ensure_ascii=False, indent=2))
        return 0 if gate['research_ready'] else 2
    except (OSError, ValueError, KeyError, TypeError, PipelineError) as exc:
        print(json.dumps({'research_ready': False, 'error': str(exc)}, ensure_ascii=False))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())

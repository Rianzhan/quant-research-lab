"""Guarded Phase 6A demonstrations; all strategy, prediction and holdout APIs absent."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
from src.backtest.actions import readiness
from src.backtest.contracts import require, stable_bytes
from src.backtest.inputs import load_inputs, read_policy, verify_preserved_code
from src.backtest.pipeline import artifact_bytes, historical_cases, publish_bundle, synthetic_cases, verify_bundle


def _execution_snapshot(root: Path, config: Path) -> dict:
    """Capture execution inputs before computation; never label old results with new bytes."""
    return {'execution_config_sha256':sha256(config.read_bytes()).hexdigest(),
            'code_sha256':{p.relative_to(root).as_posix():sha256(p.read_bytes().replace(b'\r\n',b'\n')).hexdigest()
                          for p in sorted((root/'src/backtest').glob('*.py'))}}


def build_bundle(root: Path, config: Path, reverse_inputs: bool=False) -> dict:
    """Verify frozen inputs before/after all synthetic and manual historical audits."""
    snapshot=_execution_snapshot(root,config)
    raw,policy=read_policy(config)
    loaded=load_inputs(root,raw)
    def verify_sources() -> None:
        require(_execution_snapshot(root,config)==snapshot,'Execution configuration/code changed during replay or publication')
        if 'source_lock' in raw:
            require(sha256((root/raw['source_lock']).read_bytes()).hexdigest()==raw['source_lock_sha256'],
                    'Source lock changed during replay or publication')
        loaded['reader'].verify_all()
        verify_preserved_code(root,loaded['source_lock']['preserved_code_sha256'])
    verify_sources()
    if reverse_inputs:
        for name in ['assets','calendar','quotes','actions']:
            # The engine normalizes sessions; historical window extraction needs a sorted calendar.
            if name!='calendar':loaded[name]=list(reversed(loaded[name]))
    cases=synthetic_cases(policy)
    real,availability=historical_cases(loaded,policy);cases.update(real)
    verify_sources()
    gates=readiness(loaded['actions'],engine_verified=True,assets=loaded['assets'])
    audit={'source_files_checked':len(loaded['reader'].files),
        'opened_paths':sorted(set(loaded['reader'].opened_paths)),
        'source_files_unchanged':True,'sealed_payloads_rehashed':False,
        'holdout_files_opened':False,'models_deserialized':False,'predictions_parsed':False,
        'model_signals_used':False,'strategy_backtest_performed':False,
        'cases':{name:{'fingerprint':result.fingerprint(),'sessions':len(result.tables['daily']),
                      'fills':len(result.tables['fills'])} for name,result in sorted(cases.items())}}
    return {'cases':cases,'actions':loaded['action_table'],'gates':gates,'availability':availability,'_verify_sources':verify_sources,
        'policy':raw,'audit':audit,'bindings':{'source_lock_sha256':loaded['source_lock_sha256'],
            **snapshot,
            'previous_commit':loaded['source_lock']['accepted_prior_commit'],
            'source_files_sha256':loaded['reader'].files,'holdout_files_opened':False}}


def install_holdout_guard(root: Path, config: Path) -> dict:
    """Process-wide file audit, including legacy/open bypasses of FrozenReader."""
    root=root.resolve();raw,_=read_policy(config)
    source=root/raw['source_lock'];payload=source.read_bytes()
    require(sha256(payload).hexdigest()==raw['source_lock_sha256'],'Source lock hash changed')
    lock=json.loads(payload);allowed=set(lock['files_sha256'])
    audit={'forbidden_access_attempts':[],'market_files_opened':[]}
    def hook(event: str, args: tuple) -> None:
        if event!='open' or not args or not isinstance(args[0],(str,bytes)):
            return
        path=Path(args[0].decode() if isinstance(args[0],bytes) else args[0]).resolve()
        if not path.is_relative_to(root):
            return
        rel=path.relative_to(root).as_posix()
        # This guard is stricter than the file reader for any legacy/raw-price path.
        if rel.startswith(('data/raw/','data/processed/')):
            if rel not in allowed or 'holdout' in rel.lower() or 'coverage_prices' in rel.lower():
                audit['forbidden_access_attempts'].append(rel)
                raise ValueError('2026/sealed or unlisted market file access forbidden: '+rel)
            if rel.endswith('.csv'):
                audit['market_files_opened'].append(rel)
    sys.addaudithook(hook)
    return audit


def main(argv: list[str] | None=None) -> int:
    """Offline publish/replay verification; no model or order-generation option."""
    root=Path(__file__).resolve().parents[2]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['publish','verify'])
    parser.add_argument('--manifest',type=Path)
    args=parser.parse_args(argv);config=root/'config/execution.yaml'
    try:
        access=install_holdout_guard(root,config)
        bundle=build_bundle(root,config)
        if args.command=='publish':
            manifest=publish_bundle(root/'results/reports',bundle)
        else:
            require(args.manifest is not None,'--manifest required')
            manifest=args.manifest;meta=verify_bundle(manifest)
            hashes={name:sha256(payload).hexdigest() for name,payload in artifact_bytes(bundle).items()}
            require(hashes==meta['files'] and bundle['bindings']==meta['bindings'],
                    'Replay output differs from frozen publication')
        meta=verify_bundle(manifest)
        require(not access['forbidden_access_attempts'],'Forbidden access attempted')
        print(json.dumps({'manifest':str(manifest),'build_key':meta['build_key'],'cases':len(bundle['cases']),
                          'gates':bundle['gates'],'forbidden_access_attempts':access['forbidden_access_attempts']},ensure_ascii=False,indent=2))
        return 0
    except (OSError,ValueError,KeyError,TypeError) as exc:
        print(json.dumps({'status':'failed','error':str(exc),'complete_result_published':False},ensure_ascii=False))
        return 2


if __name__=='__main__':
    raise SystemExit(main())

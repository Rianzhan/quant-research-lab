"""Preserve the first official attempt, checkpoints, and immutable reconciled output."""
from datetime import datetime,timezone
from hashlib import sha256
import json
from pathlib import Path
from typing import Callable
from uuid import uuid4
from src.backtest.contracts import require,stable_bytes
from src.backtest.pipeline import write_file
from src.portfolio.rules import Rules

VERSION='phase7-v1'
ACCOUNT_FILES={'daily_ledger.csv','orders.csv','fills.csv','holdings_ledger.csv','dividend_receivable_ledger.csv',
    'event_journal.csv','costs.csv','turnover.csv','corporate_action_journal.csv','cancelled_unfilled_orders.csv',
    'signals.csv','decisions.csv','order_provenance.json','holding_count_distribution.csv','summary.json'}
GLOBAL_FILES={'execution_summary.csv','coverage_audit.csv','coverage_summary.csv','gates.json','strategy_spec.json',
              'source_audit.json','corporate_actions.json','schemas.json','account_hashes.json'}

def account_id(strategy: str,fee: str) -> str:
    """Fixed account names contain no score/ranking information."""
    require(strategy in {'A','B','C'} and fee in Rules().fee_scenarios,'Unregistered account')
    return strategy+'_fee_'+fee.replace('.','p')

def expected_files() -> set[str]:
    return GLOBAL_FILES|{f'accounts/{account_id(s,f)}/{name}' for s in ['A','B','C']
                       for f in Rules().fee_scenarios for name in ACCOUNT_FILES}

def check_outputs(files: dict[str,bytes]) -> None:
    """All 12 accounts and required audit tables must be present together."""
    require(set(files)==expected_files(),'Incomplete or extra Phase7 account/output set')
    gates=json.loads(files['gates.json'])
    for name in ['strategy_backtest_performed','historical_strategy_orders_generated','strategy_pnl_computed','phase7_reconciled']:
        require(gates.get(name) is True,'Incomplete accounting gate: '+name)
    for name in ['holdout_files_opened','live_execution_ready']:
        require(gates.get(name) is False,'Forbidden Phase7 gate: '+name)

def verify(manifest: Path) -> dict:
    """Check every exact byte and reject a pending directory, even with a manifest."""
    manifest=Path(manifest);meta=json.loads(manifest.read_bytes())
    require(manifest.name=='manifest.json' and meta['status']=='complete','Incomplete publication')
    require(set(meta['files'])==expected_files(),'Incomplete publication files')
    files={name:(manifest.parent/name).read_bytes() for name in sorted(meta['files'])}
    require(all(sha256(b).hexdigest()==meta['files'][n] for n,b in files.items()),'Published file hash changed')
    check_outputs(files)
    key=sha256(stable_bytes({'version':VERSION,'bindings':meta['bindings'],'files':meta['files']})).hexdigest()
    require(meta['build_key']==key and manifest.parent.name==VERSION+'-'+key[:16],'Publication identity changed')
    return meta

def publish(parent: Path,bindings: dict,produce: Callable,verify_sources: Callable) -> Path:
    """Never rerun an existing first attempt. Verification has a separate entrypoint."""
    parent=Path(parent);parent.mkdir(parents=True,exist_ok=True);verify_sources()
    first=parent/'phase7-first-run.json';done=parent/'phase7-first-run-complete.json'
    bh=sha256(stable_bytes(bindings)).hexdigest()
    if first.exists():
        receipt=json.loads(first.read_bytes())
        require(receipt['bindings_hash']==bh,'First run inputs/code differ; never replace the initial result')
        require(done.exists(),'First attempt incomplete; preserve evidence and investigate, do not silently rerun')
        final=json.loads(done.read_bytes())
        require(final['first_receipt_sha256']==sha256(first.read_bytes()).hexdigest(),'First receipt changed')
        path=parent/final['manifest']
        require(path.resolve().is_relative_to(parent.resolve()),'Completion path escapes')
        meta=verify(path);require(meta['bindings']==bindings and meta['build_key']==final['build_key'],'First run binding changed')
        return path
    pending=parent/('.pending-phase7-'+uuid4().hex);pending.mkdir()
    receipt={'version':VERSION,'attempt_directory':pending.name,'bindings_hash':bh,
             'started_at_utc':datetime.now(timezone.utc).isoformat(),'first_official_attempt':True}
    write_file(first,stable_bytes(receipt))
    def save(name: str,payload: bytes):
        require(name in expected_files(),'Unregistered checkpoint path')
        path=pending/name
        if path.exists():require(path.read_bytes()==payload,'Checkpoint changed');return
        write_file(path,payload)
    try:
        files=produce(save);check_outputs(files)
        for name,payload in sorted(files.items()):save(name,payload)
        hashes={n:sha256(b).hexdigest() for n,b in files.items()}
        require(all(sha256((pending/n).read_bytes()).hexdigest()==h for n,h in hashes.items()),'Checkpoint readback mismatch')
        key=sha256(stable_bytes({'version':VERSION,'bindings':bindings,'files':hashes})).hexdigest()
        meta={'version':VERSION,'status':'complete','build_key':key,'bindings':bindings,'files':hashes}
        write_file(pending/'manifest.json',stable_bytes(meta));verify_sources()
        target=parent/(VERSION+'-'+key[:16]);require(not target.exists(),'Refuse to replace prior result')
        pending.rename(target)
        verify(target/'manifest.json')
        write_file(done,stable_bytes({'manifest':target.name+'/manifest.json','build_key':key,
                                    'first_receipt_sha256':sha256(first.read_bytes()).hexdigest()}))
        return target/'manifest.json'
    except Exception as exc:
        if pending.exists():
            try:write_file(pending/'failure.json',stable_bytes({'error':str(exc),'complete_result_published':False}))
            except OSError:pass
        raise

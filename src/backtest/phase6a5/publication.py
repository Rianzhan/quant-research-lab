"""Independent research gates and immutable Phase 6A.5 publication."""
from hashlib import sha256
from decimal import Decimal, localcontext
import json
from pathlib import Path
from uuid import uuid4
from src.backtest.actions import applicability_issues
from src.backtest.contracts import require, stable_bytes
from src.backtest.phase6a5.engine import DelayedShareAction
from src.backtest.pipeline import validate_result, write_file

VERSION='phase6a5-v1'


def validate_research(result,initial_cash: Decimal | None=None) -> None:
    """Use the same 40-digit arithmetic for validation as for event posting."""
    with localcontext() as ctx:
        ctx.prec=40
        validate_result(result,initial_cash=initial_cash)


def research_gates(actions,assets,full_window,calendar) -> dict:
    """Never infer research completeness from live/broker gates or one event test."""
    blockers=[]
    for a in actions:
        issues=a.issues() if a.applicable else applicability_issues(a,assets)
        if isinstance(a,DelayedShareAction) and not a.known_at:issues=issues+['final_ratio_unknown']
        if issues:blockers.append({'event_id':a.event_id,'issues':issues})
    complete=bool(actions) and not blockers
    reconciled=False
    if full_window is not None:
        validate_research(full_window.result)
        dates=[r['date'] for r in full_window.result.tables['daily']]
        keys={(r['date'],r['asset_code']) for r in full_window.result.tables['holdings']}
        expected={(d,a.asset_code) for d in calendar for a in assets}
        reconciled=dates==sorted(calendar) and keys==expected and full_window.resolved
    return {'accounting_engine_ready':full_window is not None,
        'corporate_action_accounting_ready':complete,
        'historical_research_execution_ready':complete and reconciled,
        'historical_total_return_research_ready':complete and reconciled,
        'historical_execution_ready':complete and reconciled,'backtest_total_return_ready':complete and reconciled,
        'broker_rules_ready':False,'broker_realism_ready':False,'live_execution_ready':False,
        'model_signals_used':False,'strategy_backtest_performed':False,'holdout_files_opened':False,
        'full_development_ledger_reconciled':reconciled,'blockers':blockers,
        'scope':'registered_2018_2025_eight_asset_research_assumptions_not_verified_broker_execution'}


def verify(manifest: Path) -> dict:
    """Reject partial, tampered or escaping publications without repairing files."""
    manifest=Path(manifest)
    require(manifest.name=='manifest.json' and manifest.parent.name.startswith(VERSION+'-'),'Incomplete publication')
    meta=json.loads(manifest.read_bytes())
    require(meta['status']=='complete' and meta['holdout_files_opened'] is False,'Invalid completion flags')
    for name,h in meta['files'].items():
        p=(manifest.parent/name).resolve()
        require(p.is_relative_to(manifest.parent.resolve()),'Escaping publication member')
        require(sha256(p.read_bytes()).hexdigest()==h,'Publication file changed: '+name)
    key=sha256(stable_bytes({'version':VERSION,'bindings':meta['bindings'],'files':meta['files']})).hexdigest()
    require(meta['build_key']==key and manifest.parent.name==VERSION+'-'+key[:16],'Publication identity changed')
    return meta


def publish(parent: Path,files: dict[str,bytes],bindings: dict,verify_sources) -> Path:
    """Durable write, readback, source recheck, then one final directory rename."""
    verify_sources()
    hashes={n:sha256(b).hexdigest() for n,b in files.items()}
    key=sha256(stable_bytes({'version':VERSION,'bindings':bindings,'files':hashes})).hexdigest()
    parent=Path(parent);parent.mkdir(parents=True,exist_ok=True)
    target=parent/(VERSION+'-'+key[:16])
    if target.exists():verify(target/'manifest.json');return target/'manifest.json'
    pending=parent/('.pending-'+uuid4().hex);pending.mkdir()
    for name,payload in sorted(files.items()):
        require((pending/name).resolve().is_relative_to(pending.resolve()),'Escaping publication member')
        write_file(pending/name,payload)
    for name,h in hashes.items():require(sha256((pending/name).read_bytes()).hexdigest()==h,'Pending file corrupted')
    meta={'status':'complete','version':VERSION,'build_key':key,'files':hashes,'bindings':bindings,
          'holdout_files_opened':False,'model_signals_used':False,'strategy_backtest_performed':False}
    write_file(pending/'manifest.json',stable_bytes(meta))
    require((pending/'manifest.json').read_bytes()==stable_bytes(meta),'Pending manifest corrupted')
    verify_sources();pending.rename(target)
    return target/'manifest.json'

"""Offline erratum and manual ledger audits; no model, threshold or strategy imports."""
import argparse
from dataclasses import asdict, replace
from hashlib import sha256
import json
from pathlib import Path
import sys
from src.backtest.contracts import Asset, OrderIntent, Policy, Quote, number, require, stable_bytes
from src.backtest.inputs import load_inputs, read_policy, verify_preserved_code
from src.backtest.pipeline import SCHEMAS, TABLE_FILES, csv_bytes
from src.backtest.phase6a5.engine import DelayedShareAction, run_research
from src.backtest.phase6a5.evidence import correct_event
from src.backtest.phase6a5.publication import publish, research_gates, verify, validate_research


def check_market_access(root: Path,path: Path,allowed: set[str],calendar: str) -> bool:
    """Check before open, including raw mixed CSV and sealed-file aliases."""
    path=path.resolve();root=root.resolve()
    if not path.is_relative_to(root):return False
    rel=path.relative_to(root).as_posix();low=rel.lower()
    if low.startswith(('data/raw/','data/processed/')):
        require('holdout' not in low and 'coverage_prices' not in low,'Sealed/mixed market access forbidden')
        require(not (low.startswith('data/raw/') and low.endswith('.csv') and rel!=calendar),'Mixed raw market access forbidden')
        require(rel in allowed,'Unlisted market access forbidden')
        return True
    return False


def install_guard(root: Path,config: dict,raw: dict) -> dict:
    """Process audit tracks direct opens in addition to FrozenReader checks."""
    lock=json.loads((root/raw['source_lock']).read_bytes())
    allowed=set(lock['files_sha256'])|set(config['evidence_files_sha256'])
    audit={'holdout_files_opened':False,'forbidden_access_attempts':[],'market_paths_opened':[]}
    def hook(event,args):
        if event!='open' or not args or not isinstance(args[0],(str,bytes)):return
        path=Path(args[0].decode() if isinstance(args[0],bytes) else args[0])
        try:
            if check_market_access(root,path,allowed,lock['calendar_raw']):
                audit['market_paths_opened'].append(path.resolve().relative_to(root).as_posix())
        except ValueError:
            audit['forbidden_access_attempts'].append(str(path));raise
    sys.addaudithook(hook)
    return audit


def snapshot(root: Path,config_path: Path,config: dict) -> dict:
    """Bind every consumed configuration, prior output and current implementation."""
    files={**config['evidence_files_sha256'],**config['prior_phase6a_files_sha256']}
    for rel,h in files.items():
        p=(root/rel).resolve()
        require(p.is_relative_to(root) and 'holdout' not in rel.lower() and 'coverage_prices' not in rel.lower(),'Unsafe binding')
        require(sha256(p.read_bytes()).hexdigest()==h,'Frozen evidence/prior artifact changed: '+rel)
    verify_preserved_code(root,config['prior_phase6a_code_sha256'])
    return {'config_sha256':sha256(config_path.read_bytes()).hexdigest(),'files_sha256':files,
        'code_sha256':{p.relative_to(root).as_posix():sha256(p.read_bytes().replace(b'\r\n',b'\n')).hexdigest()
                       for p in sorted((root/'src/backtest/phase6a5').glob('*.py'))},
        'prior_code_sha256':config['prior_phase6a_code_sha256']}


def intent(code,date,identifier,quantity='100',weight=None):
    """Fixed human fixture instruction, independent of prices and model predictions."""
    return OrderIntent(identifier,date,code,'buy',date+'T23:59:59+08:00','frozen_manual_fixture',
                       requested_shares=quantity,target_weight=weight)


def synthetic_late(policy: Policy,known: str,ratio: str | None):
    """Literal exact-conversion audit. The real 510500 event is never sent here."""
    dates=['2022-08-24','2022-08-25','2022-08-26','2022-08-29','2022-08-30','2022-08-31']
    asset=Asset('SYNTHETIC_LATE',dates[0])
    action=DelayedShareAction('synthetic_late','SYNTHETIC_LATE','split',known,dates[2],phase='after_close',
        record_date=dates[2],share_ratio=ratio,rounding='exact',evidence_sha256='a'*64,
        announced_at='2022-08-23T23:59:59+08:00')
    quotes=[Quote(d,asset.asset_code,'10' if i<3 else '5','10' if i<3 else '5') for i,d in enumerate(dates)]
    orders=[intent(asset.asset_code,dates[0],'establish'),intent(asset.asset_code,dates[2],'old_pending','10'),
            intent(asset.asset_code,dates[3],'new_after_known','10')]
    return run_research(replace(policy,fee_rate='0',slippage='0'),[asset],dates,quotes,orders,[action])


def build(root: Path,reverse: bool=False) -> tuple[dict,dict,object]:
    """Recheck all frozen sources around corrected-event and complete-calendar audits."""
    root=root.resolve();path=root/'config/execution_phase6a5.json';config=json.loads(path.read_bytes())
    require(config['version']=='phase6a5-v1','Unsupported correction version')
    require(all(config[k] is False for k in ['model_signals_used','strategy_backtest_performed','holdout_files_opened','live_execution_ready']),
            'Forbidden scope flag')
    raw,base_policy=read_policy(root/config['base_config']);before=snapshot(root,path,config)
    loaded=load_inputs(root,raw)
    def recheck():
        require(snapshot(root,path,config)==before,'Inputs/code changed during replay')
        loaded['reader'].verify_all()
        verify_preserved_code(root,loaded['source_lock']['preserved_code_sha256'])
    actions=[correct_event(a,config['correction']) if a.event_id==config['correction']['event_id'] else a for a in loaded['actions']]
    require(any(a.event_id==config['correction']['event_id'] for a in actions),'Correction event missing')
    policy=replace(base_policy,initial_cash=config['audit_initial_cash'])
    calendar=loaded['calendar'];assets=loaded['assets'];quotes=loaded['quotes']
    def replay(aa,dd,qq,oo,ee):
        parts=[aa,dd,qq,oo,ee]
        if reverse:parts=[list(reversed(x)) for x in parts]
        return run_research(policy,*parts)
    # Hold all assets through the full calendar; first-observation orders do not invent prelisting prices.
    manual=[intent(a.asset_code,a.first_tradable_date,'full_'+a.asset_code) for a in assets]
    full=replay(assets,calendar,quotes,manual,actions)
    require({f['asset_code'] for f in full.result.tables['fills']}=={a.asset_code for a in assets},'Full audit did not establish all assets')
    dates=[d for d in calendar if '2022-08-23'<=d<='2022-08-31']
    aa=[a for a in assets if a.asset_code=='510500'];qq=[q for q in quotes if q.asset_code=='510500' and q.date in dates]
    ee=[a for a in actions if a.asset_code=='510500']
    cases={'full_development_manual_accounting':full}
    for name,quantity in [('real_510500_exact','100000'),('real_510500_fractional','100')]:
        orders=[intent('510500',dates[0],'establish',quantity),intent('510500','2022-08-25','cross_event','10'),
                intent('510500','2022-08-26','old_basis_target',None,'0.9'),intent('510500','2022-08-29','new_order','10')]
        cases[name]=replay(aa,dates,qq,orders,ee)
    cases['synthetic_path_a']=synthetic_late(policy,'2022-08-29T08:00:00+08:00','2')
    cases['synthetic_path_b']=synthetic_late(policy,'2022-08-29T23:59:59+08:00','2')
    cases['synthetic_never_known']=synthetic_late(policy,'',None)
    for r in cases.values():validate_research(r.result,initial_cash=number(policy.initial_cash))
    gates=research_gates(actions,assets,full,calendar)
    registry=[{**asdict(a),'time_precision':'date','known_at_semantics':'conservative_publication_day_end',
        'correction_applied':a.event_id==config['correction']['event_id']} for a in actions]
    audit={'full_window_first_session':calendar[0],'full_window_last_session':calendar[-1],
        'calendar_sessions':len(calendar),'assets':sorted(a.asset_code for a in assets),
        'old_allowed_files_checked':len(loaded['reader'].files),'prior_phase6a_files_checked':len(config['prior_phase6a_files_sha256']),
        'sealed_payloads_rehashed':False,'holdout_files_opened':False,'predictions_parsed':False,'models_deserialized':False,
        'strategy_backtest_performed':False,'model_signals_used':False,
        'web_search_exposure':config['web_search_exposure'],
        'case_fingerprints':{n:r.result.fingerprint() for n,r in sorted(cases.items())},
        'case_sessions':{n:len(r.result.tables['daily']) for n,r in sorted(cases.items())}}
    files={'gates.json':stable_bytes(gates),'audit.json':stable_bytes(audit),'corporate_actions.json':stable_bytes(registry),
        'corporate_actions.csv':csv_bytes(registry),'correction.json':stable_bytes(config['correction']),
        'schemas.json':stable_bytes(SCHEMAS),'execution_policy.json':stable_bytes({**asdict(policy),'base_config':raw})}
    for name,r in sorted(cases.items()):
        for table,rows in r.result.tables.items():files[f'cases/{name}/{TABLE_FILES[table]}']=csv_bytes(rows,list(SCHEMAS[table]))
        if r.states:files[f'cases/{name}/corporate_action_states.csv']=csv_bytes(r.states)
    recheck()
    return files,{**before,'base_commit':config['base_commit'],'old_source_lock_sha256':raw['source_lock_sha256']},recheck


def main(argv=None) -> int:
    """Publish or verify without network, predictions or sealed data access."""
    root=Path(__file__).resolve().parents[3]
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['publish','verify'])
    parser.add_argument('--manifest',type=Path);args=parser.parse_args(argv)
    try:
        config=json.loads((root/'config/execution_phase6a5.json').read_bytes());raw,_=read_policy(root/config['base_config'])
        access=install_guard(root,config,raw);files,bindings,check=build(root)
        files['access_audit.json']=stable_bytes({**access,'market_paths_opened':sorted(set(access['market_paths_opened']))})
        require(not access['forbidden_access_attempts'],'Forbidden open attempted')
        if args.command=='publish':manifest=publish(root/'results/reports',files,bindings,check)
        else:
            require(args.manifest is not None,'Manifest required');manifest=args.manifest;meta=verify(manifest)
            require(meta['files']=={n:sha256(b).hexdigest() for n,b in files.items()} and meta['bindings']==bindings,'Replay differs')
        meta=verify(manifest)
        print(json.dumps({'manifest':str(manifest),'build_key':meta['build_key'],'gates':json.loads(files['gates.json'])},indent=2))
        return 0
    except (ValueError,OSError,KeyError,TypeError) as exc:
        print(json.dumps({'status':'failed','error':str(exc),'complete_result_published':False}));return 2


if __name__=='__main__':raise SystemExit(main())

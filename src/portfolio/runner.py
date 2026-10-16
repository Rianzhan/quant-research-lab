"""Offline protocol freeze: structural prediction audit, never historical orders."""
import argparse
from collections import Counter
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import yaml
from src.backtest.contracts import require,stable_bytes
from src.backtest.pipeline import csv_bytes
from src.portfolio.rules import Rules,FAMILIES,DecisionState,AssetState,project_predictions,schedule,rules_hash
from src.portfolio.orders import map_orders
from src.portfolio.inputs import SpecReader,read_spec,verify_code,csv_rows,install_guard
from src.portfolio.publication import publish,verify

PRED_COLUMNS=['date','exchange','asset_code','price_segment','fold_id','scope','model','task','prediction']

def load_registration(root: Path) -> tuple[dict,bytes,SpecReader,dict]:
    """Pin the source list before the reader can touch any model/data artifact."""
    payload=(root/'config/strategy_spec.json').read_bytes();spec=json.loads(payload);read_spec(spec)
    require(spec['source_lock']=='config/strategy_sources.json','Unexpected source lock')
    raw=(root/spec['source_lock']).read_bytes()
    require(sha256(raw).hexdigest()==spec['source_lock_sha256'],'Frozen source lock changed')
    lock=json.loads(raw)
    return spec,payload,SpecReader(root,lock['files_sha256'],lock['calendar_raw']),lock

def implementation_hashes(root: Path) -> dict[str,str]:
    """Bind this new module's code in each immutable publication."""
    return {p.relative_to(root).as_posix():sha256(p.read_bytes().replace(b'\r\n',b'\n')).hexdigest()
            for p in sorted((root/'src/portfolio').glob('*.py'))}

def synthetic_examples(rules: Rules,calendar: list[str]) -> dict:
    """Artificial constant balances/quotes illustrate sizing without a NAV series."""
    date=calendar[0]
    assets=tuple(AssetState('SYNTH_'+str(i),'SSE','2018-01-02','10') for i in range(5))
    state=DecisionState(date,'100000','100000',assets)
    rows=[]
    for strategy,values in [('A',['.60','.56','.55','.40','.50']),('B',['.002','.0015','.001','.0005','-.01'])]:
        scope,model,task=FAMILIES[strategy]
        rows.extend(dict(date=date,exchange=a.exchange,asset_code=a.asset_code,price_segment='0',
                         fold_id='test_'+date[:4],scope=scope,model=model,task=task,prediction=v)
                    for a,v in zip(assets,values))
    projected=project_predictions(rows)
    return {'synthetic_only':True,'not_real_market_orders':True,'decision_state':asdict(state),
            'cases':{s:asdict(map_orders(s,projected,state,rules,calendar)) for s in FAMILIES},
            'steps':['prediction_only_fixed_hurdle','capped_equal_weight','t_close_lot_quantity',
                     't_close_cash_budget','frozen_order_next_session_only'],
            'no_historical_strategy_nav_computed':True}

def build(root: Path,reverse: bool=False) -> tuple[dict,dict,object]:
    """Audit immutable coverage only; no real predictions reach map_orders."""
    root=root.resolve();spec,payload,reader,lock=load_registration(root);rules=read_spec(spec)
    require(spec['families']==json.loads(stable_bytes(FAMILIES)),'Strategy family changed')
    code=implementation_hashes(root);reader.verify_all();verify_code(root,lock['preserved_code_sha256'])
    def recheck() -> None:
        require((root/'config/strategy_spec.json').read_bytes()==payload,'Strategy spec changed during freeze')
        require(sha256((root/spec['source_lock']).read_bytes()).hexdigest()==spec['source_lock_sha256'],'Source lock changed')
        require(implementation_hashes(root)==code,'Strategy implementation changed during freeze')
        reader.verify_all();verify_code(root,lock['preserved_code_sha256'])
    calendar=[r['trade_date'] for r in csv_rows(reader.read(lock['calendar_raw']),['trade_date'])
              if '2018-01-01'<=r['trade_date']<='2025-12-31']
    outer=[d for d in calendar if '2022-01-01'<=d<=rules.end]
    require(outer and outer[0]==rules.anchor,'Anchor is not first outer-test domestic session')
    planned=schedule(calendar[::-1] if reverse else calendar,rules)
    index={d:i for i,d in enumerate(calendar)}
    dates=[{'decision_date':d,'execution_date':calendar[index[d]+1] if index[d]+1<len(calendar) else None,
            'global_session_offset':index[d]-index[rules.anchor]} for d in planned]
    rows=csv_rows(reader.read(lock['predictions']),PRED_COLUMNS)
    preds=project_predictions(rows[::-1] if reverse else rows)
    config=yaml.safe_load(reader.read('config/config.yaml'))
    universe={a['code']:a['exchange'] for a in config['universe']['assets']}
    require(all(p.asset_code in universe and universe[p.asset_code]==p.exchange and p.date in outer for p in preds),
            'Prediction identity outside frozen calendar/universe')
    require(not any(p.asset_code=='588000' and p.fold_id=='test_2022' for p in preds),'Unexpected 588000/2022 prediction')
    counts=Counter((p.scope,p.model,p.task,p.fold_id,p.asset_code) for p in preds)
    audit={'source_rows':len(rows),'selected_prediction_rows':len(preds),'columns_consumed':PRED_COLUMNS,
        'selected_coverage':[dict(zip(['scope','model','task','fold_id','asset_code'],k),rows=n) for k,n in sorted(counts.items())],
        '588000_2022_model_predictions':0,'benchmark_does_not_depend_on_prediction_coverage':True,
        'labels_parsed':False,'models_deserialized':False,'model_training_performed':False,
        'price_data_parsed':False,'historical_signals_generated':False,'strategy_backtest_performed':False,
        'holdout_files_opened':False,'limitation':'Phase5 coverage inherits complete features and label-valid sample selection; not deployment coverage.'}
    gate_path=str(Path(lock['accounting_manifest']).parent/'gates.json').replace('\\','/')
    accounting=json.loads(reader.read(gate_path))
    require(accounting['historical_research_execution_ready'] and accounting['historical_total_return_research_ready'],
            'Accepted research execution/accounting gates unavailable')
    gates={'strategy_spec_ready':True,'signal_to_order_mapping_ready':True,'strategy_rules_frozen':True,
           'strategy_backtest_performed':False,'holdout_files_opened':False,'historical_signals_generated':False,
           'model_training_performed':False,'broker_rules_ready':False,'broker_realism_ready':False,'live_execution_ready':False}
    spec_hash=sha256(stable_bytes(spec)).hexdigest()
    bindings={'base_commit':lock['base_commit'],'source_lock_sha256':spec['source_lock_sha256'],
              'strategy_spec_hash':spec_hash,'rules_hash':rules_hash(rules),'code_sha256':code}
    recheck()
    access={'holdout_files_opened':False,'sealed_payloads_rehashed':False,
            'opaque_verified_files':len(reader.files),'preserved_source_files':len(lock['preserved_code_sha256']),
            'opened_files':sorted(set(reader.opened)),'network_used':False}
    files={'strategy_spec.json':stable_bytes(spec),'strategy_spec.sha256':(spec_hash+'\n').encode(),
        'gates.json':stable_bytes(gates),'history_audit.json':reader.read(lock['history_audit']),
        'prediction_audit.json':stable_bytes(audit),'rebalance_calendar.csv':csv_bytes(dates),
        'examples.json':stable_bytes(synthetic_examples(rules,outer)),
        'access_audit.json':stable_bytes(access),'protocol.md':reader.read(spec['protocol']),
        'source_bindings.json':stable_bytes(bindings)}
    return files,bindings,recheck

def main(argv: list[str] | None=None) -> int:
    """Publish/verify the protocol, never a strategy backtest command."""
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['publish','verify'])
    parser.add_argument('--manifest',type=Path);parser.add_argument('--reverse',action='store_true')
    args=parser.parse_args(argv);root=Path(__file__).resolve().parents[2]
    try:
        _,_,reader,_=load_registration(root);access=install_guard(reader)
        files,bindings,recheck=build(root,args.reverse)
        require(not access['denied'],'Forbidden data open attempted')
        if args.command=='publish':manifest=publish(root/'results/reports',files,bindings,recheck)
        else:
            require(args.manifest is not None,'Manifest required');manifest=args.manifest;meta=verify(manifest)
            require(meta['bindings']==bindings and meta['files']=={n:sha256(b).hexdigest() for n,b in files.items()},'Protocol replay differs')
        meta=verify(manifest)
        print(json.dumps({'manifest':str(manifest),'build_key':meta['build_key'],**bindings,'gates':json.loads(files['gates.json'])},indent=2))
        return 0
    except (ValueError,OSError,KeyError,TypeError) as exc:
        print(json.dumps({'status':'failed','error':str(exc),'complete_result_published':False}));return 2

if __name__=='__main__':raise SystemExit(main())

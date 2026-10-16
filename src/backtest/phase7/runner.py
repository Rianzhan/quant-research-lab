"""Run the frozen twelve accounts once; separately replay identical inputs for audit."""
import argparse,json
from dataclasses import asdict,replace
from hashlib import sha256
from pathlib import Path
from src.backtest.contracts import require,stable_bytes
from src.backtest.pipeline import SCHEMAS,TABLE_FILES,csv_bytes
from src.backtest.phase7.inputs import contract,load
from src.backtest.phase7.engine import run_account
from src.backtest.phase7.audit import account_tables,coverage
from src.backtest.phase7.publication import account_id,publish,verify
from src.portfolio.inputs import install_guard
from src.portfolio.rules import Rules,rules_hash

EXTRA_NAMES={'costs':'costs.csv','turnover':'turnover.csv','corporate_actions':'corporate_action_journal.csv',
             'cancelled_unfilled':'cancelled_unfilled_orders.csv','holding_count_distribution':'holding_count_distribution.csv'}

def generate(data: dict,diagnostics: dict,rules: Rules,spec: dict,source_audit: dict,
             checkpoint=None,reverse: bool=False) -> dict[str,bytes]:
    """Only fixed A/B/C and four fees; diagnostics are joined after each replay."""
    files={};summaries=[];coverage_rows=[];coverage_summaries=[];hashes={};signals={};schema={}
    arguments={k:v for k,v in data.items() if k not in {'policy','predictions'}}
    predictions=data['predictions']
    if reverse:
        arguments={k:list(reversed(v)) if isinstance(v,list) else v for k,v in arguments.items()}
        predictions=predictions[::-1]
    for strategy in ['A','B','C']:
        for fee in rules.fee_scenarios:
            identity=account_id(strategy,fee)
            account=run_account(strategy,replace(data['policy'],fee_rate=fee),predictions,**arguments,rules=rules)
            signature=sha256(stable_bytes(account.signals)).hexdigest()
            require(signals.setdefault(strategy,signature)==signature,'Fee scenario changed strategy signals')
            tables,summary=account_tables(account);summaries.append({'account_id':identity,**summary})
            hashes[identity]={'account_hash':account.fingerprint(),'signal_hash':signature,'rules_hash':account.rule_hash}
            prefix='accounts/'+identity+'/'
            for table,values in tables.items():
                name={**TABLE_FILES,**EXTRA_NAMES}[table]
                base_table={'cancelled_unfilled':'orders','corporate_actions':'journal'}.get(table,table)
                columns=list(values[0]) if values else list(SCHEMAS.get(base_table,{}))
                schema[name]=columns;files[prefix+name]=csv_bytes(values,columns)
            files[prefix+'signals.csv']=csv_bytes(account.signals)
            files[prefix+'decisions.csv']=csv_bytes(account.decisions)
            files[prefix+'order_provenance.json']=stable_bytes(account.order_provenance)
            files[prefix+'summary.json']=stable_bytes(summary)
            if fee==rules.primary_fee_rate:
                rows,totals=coverage(account,diagnostics);coverage_rows.extend(rows);coverage_summaries.extend(totals)
            if checkpoint:
                for name,payload in sorted(files.items()):
                    if name.startswith(prefix):checkpoint(name,payload)
    gates={'strategy_spec_ready':True,'signal_to_order_mapping_ready':True,'strategy_rules_frozen':True,
        'strategy_backtest_performed':True,'historical_strategy_orders_generated':True,'strategy_pnl_computed':True,
        'phase7_reconciled':True,'holdout_files_opened':False,'live_execution_ready':False,
        'broker_rules_ready':False,'broker_realism_ready':False,'model_training_performed':False,
        'strategy_optimization_performed':False,'strategy_ranking_performed':False}
    files.update({'execution_summary.csv':csv_bytes(summaries),'coverage_audit.csv':csv_bytes(coverage_rows),
        'coverage_summary.csv':csv_bytes(coverage_summaries),'strategy_spec.json':stable_bytes(spec),'gates.json':stable_bytes(gates),
        'source_audit.json':stable_bytes({**source_audit,'fee_scenario_signals_identical':True,'rules_hash':rules_hash(rules),
            'prediction_coverage_interpretation':'development_research_on_frozen_OOS_coverage_not_point_in_time_live_deployment'}),
        'corporate_actions.json':stable_bytes([asdict(a) for a in sorted(data['actions'],key=lambda a:a.event_id)]),
        'schemas.json':stable_bytes({'tables':schema,'currency':'CNY','quantity':'ETF shares',
            'turnover':'effective executed notional / open pre-trade NAV; both sides summed, not halved',
            'gross_value':'net NAV + cumulative explicit fees + cumulative slippage on the SAME executed path',
            'cash_ratio':'cash including restricted sale proceeds / closing NAV',
            'coverage':'one row per strategy/asset/rebalance date; invariant across fee scenarios',
            'orders_source':'strategy_spec:A|B|C; quantities frozen at decision close'}),
        'account_hashes.json':stable_bytes(hashes)})
    return files

def code_hashes(root: Path) -> dict:
    return {p.relative_to(root).as_posix():sha256(p.read_bytes().replace(b'\r\n',b'\n')).hexdigest()
            for p in sorted((root/'src/backtest/phase7').glob('*.py'))}

def main(argv: list[str] | None=None) -> int:
    """Publish first official run or verify its hash without replacing any output."""
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['publish','verify'])
    parser.add_argument('--manifest',type=Path);parser.add_argument('--reverse',action='store_true')
    args=parser.parse_args(argv);root=Path(__file__).resolve().parents[3]
    try:
        config,reader,rules,source_check=contract(root);access=install_guard(reader)
        code=code_hashes(root);config_hash=sha256((root/'config/phase7.json').read_bytes()).hexdigest()
        def recheck():
            source_check();require(code_hashes(root)==code,'Phase7 implementation changed during replay')
            require(not access['denied'],'Forbidden data open attempted')
        data,diagnostics=load(root,config,reader)
        spec=json.loads(reader.read('config/strategy_spec.json'))
        bindings={'base_commit':config['base_commit'],'phase7_config_sha256':config_hash,'code_sha256':code,
                  'strategy_spec_hash':config['strategy_spec_hash'],'rules_hash':config['rules_hash']}
        source_audit={'holdout_files_opened':False,'sealed_payloads_rehashed':False,'network_used':False,
            'files_verified':len(reader.files),'preserved_source_files':len(config['preserved_code_sha256']),
            'opened_files':sorted(set(reader.opened)),'sessions':len(data['sessions']),
            'first_session':data['sessions'][0],'last_session':data['sessions'][-1],
            'price_rows':len(data['quotes']),'selected_predictions':len(data['predictions']),
            'diagnostic_flags_separate_from_execution':True,'model_objects_deserialized':False}
        def produce(checkpoint=None):
            return generate(data,diagnostics,rules,spec,source_audit,checkpoint,args.reverse)
        if args.command=='publish':
            manifest=publish(root/'results/reports',bindings,produce,recheck)
        else:
            require(args.manifest is not None,'Manifest required');manifest=args.manifest;meta=verify(manifest)
            require(meta['bindings']==bindings,'Replay inputs/code changed')
            files=produce();recheck()
            require(meta['files']=={n:sha256(b).hexdigest() for n,b in files.items()},'Replay output hash mismatch')
        meta=verify(manifest)
        print(json.dumps({'manifest':str(manifest),'build_key':meta['build_key'],'accounts':12,
                          'holdout_files_opened':False,'phase7_reconciled':True},indent=2));return 0
    except (ValueError,OSError,KeyError,TypeError) as exc:
        print(json.dumps({'status':'failed','error':str(exc),'complete_result_published':False}));return 2

if __name__=='__main__':raise SystemExit(main())

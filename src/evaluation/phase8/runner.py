"""Offline evaluation CLI. Re-evaluation verifies metrics; it never executes strategies."""
import argparse,json,platform,warnings
from importlib.metadata import version
from pathlib import Path
import pandas as pd
from .common import digest,stable_bytes,require
from .inputs import load,install_guard,BUILD_KEY,SPEC_HASH,RULES_HASH,MANIFEST_SHA
from .tables import evaluate
from .charts import render as charts
from .report import render as report
from .publication import write_complete,verify_saved

GATES={'performance_evaluation_completed':True,'phase8_reconciled':True,'strategy_rules_frozen':True,
       'strategy_optimization_performed':False,'holdout_files_opened':False,'live_execution_ready':False,
       'strategy_replayed':False,'model_retrained':False}


def source_hashes(root: Path) -> dict[str,str]:
    """Bind only this evaluator's code and contract, not unrelated historical source."""
    paths=sorted((root/'src/evaluation/phase8').glob('*.py'))+[root/'config/phase8.json']
    return {p.relative_to(root).as_posix():digest(p.read_bytes()) for p in paths}


def artifacts(bundle: dict,tables: dict[str,pd.DataFrame],warning_list: list[str]) -> dict[str,bytes]:
    """CSV formatting is fixed, preserving exact monetary strings where required."""
    files={name+'.csv':frame.to_csv(index=False,lineterminator='\n',float_format='%.15g',na_rep='').encode('utf-8')
           for name,frame in sorted(tables.items())}
    files.update(charts(tables));files['report.md']=report(tables).encode('utf-8')
    for name,payload in {'gates':GATES,'input_audit':bundle['input_audit'],
        'account_hashes':bundle['account_hashes'],
        'evaluation_checks':{'accounts':12,'annual_rows':48,'shared_sessions':969,'annualization_sessions':252,
            'risk_free_rate':0,'all_account_identities_reconciled':True,'annual_compounding_reconciled':True,
            'yearly_wealth_gap_reconciled':True,'strategy_paths_recomputed':False,'warnings':warning_list}}.items():
        files[name+'.json']=stable_bytes(payload)+b'\n'
    return files


def run(root: Path,mode: str,path: Path | None=None) -> dict:
    """Validate before and after evaluation; publish or compare exact output bytes."""
    require(mode in {'publish','verify'},'Unsupported mode');root=Path(root).resolve()
    audit=install_guard(root);code=source_hashes(root)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        bundle,recheck=load(root);tables=evaluate(bundle)
        files=artifacts(bundle,tables,[])
    observed=sorted(set(f'{w.category.__name__}: {w.message}' for w in caught))
    checks=json.loads(files['evaluation_checks.json']);checks['warnings']=observed
    files['evaluation_checks.json']=stable_bytes(checks)+b'\n'
    bindings={'phase7_build_key':BUILD_KEY,'phase7_manifest_sha256':MANIFEST_SHA,
        'strategy_spec_hash':SPEC_HASH,'rules_hash':RULES_HASH,'account_hashes':bundle['account_hashes'],
        'source_hashes':code,'runtime':{'python':platform.python_version(),
            **{p:version(p) for p in ['numpy','pandas','matplotlib']}}}
    def unchanged():
        recheck();require(source_hashes(root)==code,'Evaluation source changed during run')
        require(not audit['denied'],'A prohibited data access was attempted')
    unchanged()
    if mode=='publish':
        require(path is None,'Publish destination is fixed')
        destination=write_complete(root/'results/reports',files,bindings,unchanged)
    else:
        require(path is not None,'Specify --publication for verification')
        destination=Path(path).resolve()
        require(destination.parent==root/'results/reports' and destination.name.startswith('phase8-v1-'),'Invalid verification destination')
    meta=verify_saved(destination,files,bindings);unchanged()
    return {'mode':mode,'publication':str(destination),'build_key':meta['build_key'],
            'files_verified':len(files),'reproducible_evaluation':True,'gates':GATES,
            'forbidden_access_attempts':audit['denied'],'warnings':observed}


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=['publish','verify'])
    parser.add_argument('--publication',type=Path);args=parser.parse_args()
    root=Path(__file__).resolve().parents[3]
    print(json.dumps(run(root,args.mode,args.publication),ensure_ascii=False,sort_keys=True))


if __name__=='__main__':main()

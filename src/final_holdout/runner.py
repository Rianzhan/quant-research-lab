"""Phase9B commands: development proof, sole final fitting, immutable verification."""
import argparse
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import platform
import sys
import pandas as pd
from src.backtest.contracts import require,stable_bytes
from src.final_holdout.inputs import load
from src.final_holdout.safety import sealed_guard
from src.final_holdout.equivalence import prove
from src.final_holdout.training import fit_final,csv_bytes
from src.final_holdout.publication import begin,finish,verify,load_model
from src.governance.final_holdout import load_documents,build as protocol_build

ROOT=Path(__file__).resolve().parents[2]
FINAL=ROOT/'results/reports/phase9b-final-models-v1'
PROOF=ROOT/'results/reports/phase9b-development-equivalence.json'
GATES={'final_holdout_execution_adapter_ready':True,'final_models_fitted':True,'final_models_frozen':True,
    'final_prediction_interface_ready':True,'holdout_files_opened':False,'final_holdout_evaluation_performed':False,
    'strategy_optimization_performed':False,'live_execution_ready':False,'broker_rules_ready':False,
    'broker_realism_ready':False,'final_holdout_event_evidence_verified':False,
    'final_refit_authorized_in_phase9b':True,'real_2026_calendar_opened':False}


def runtime_hashes() -> dict:
    """Append-only implementation source identity; never walk market directories."""
    return {p.relative_to(ROOT).as_posix():sha256(p.read_bytes()).hexdigest()
        for p in sorted((ROOT/'src/final_holdout').glob('*.py'))}


def dependencies() -> dict:
    """Bind the exact runtime used for this pickle/scaler publication."""
    return {'python':platform.python_version(),'python_executable':sys.executable,
        **{name:version(name) for name in ['numpy','pandas','scikit-learn','scipy','threadpoolctl','joblib','xgboost']}}


def fit_and_freeze() -> Path:
    """Save first fit; two audit repeats may only confirm identical bytes."""
    require(not FINAL.exists(),'Final fitting attempt already exists; use verify, never refit')
    data=load(ROOT);spec=load_documents(ROOT)['config/final_holdout_spec.json']
    proof=json.loads(PROOF.read_bytes())
    require(proof['binding']==data['binding'] and proof['runtime_sha256']==runtime_hashes(),'Equivalence evidence not current')
    require(len(proof['proof']['accounts'])==12,'Incomplete development equivalence')
    binding={**data['binding'],'runtime_sha256':runtime_hashes(),'dependencies':dependencies(),
        'equivalence_sha256':sha256(PROOF.read_bytes()).hexdigest(),
        'requirements_lock_sha256':sha256((ROOT/'requirements.lock.txt').read_bytes()).hexdigest()}
    require(binding['dependencies']['python']=='3.12.14','Frozen Python changed')
    begin(FINAL,binding)
    first=fit_final(data['features'],data['labels'],data['config'],spec)
    for repeat,shuffled in [(1,False),(2,True)]:
        x,y=data['features'],data['labels']
        if shuffled:x=x.sample(frac=1,random_state=17);y=y.sample(frac=1,random_state=3)
        replay=fit_final(x,y,data['config'],spec)
        require(first['files']==replay['files'],'Final fitting determinism failed; preserve first attempt')
        print('Determinism audit passed: '+str(repeat),flush=True)
    require(runtime_hashes()==binding['runtime_sha256'],'Runtime changed during fitting')
    # All old sources and publications are checked again without a raw loader.
    require(load(ROOT)['binding']==data['binding'],'Frozen sources changed during fitting')
    audit=[]
    for row in first['audit']:
        audit.append({k:json.dumps(v,ensure_ascii=False,sort_keys=True) if isinstance(v,(list,dict)) else v for k,v in row.items()})
    files={**first['files'],'final_training_cohort_audit.csv':csv_bytes(pd.DataFrame(audit)),
        'development_equivalence.json':PROOF.read_bytes(),
        'training_validation.json':stable_bytes({'single_official_fit':True,'audit_repeats':2,
            'repeated_fit_bytes_identical':True,'shuffled_fit_bytes_identical':True,
            'all_labels_end_by':'2025-12-31','test_counts_used':False,
            'frozen_eligibility_equal':True,'holdout_files_opened':False,'real_future_inference_calls':0})}
    return finish(FINAL,files,binding,gates=GATES)


def main(argv=None) -> int:
    """Never accepts a holdout path, future calendar, or strategy override."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prove','fit','verify']);parser.add_argument('--manifest-sha256')
    args=parser.parse_args(argv)
    with sealed_guard(ROOT) as access:
        if args.command=='prove':
            data=load(ROOT);runtime=runtime_hashes()
            evidence=prove(data,lambda name:print('Equal frozen account: '+name,flush=True))
            require(runtime_hashes()==runtime,'Runtime changed during proof')
            payload=stable_bytes({'binding':data['binding'],'runtime_sha256':runtime,'proof':evidence})
            if PROOF.exists():require(PROOF.read_bytes()==payload,'Existing proof differs; do not overwrite')
            else:
                with PROOF.open('xb') as stream:stream.write(payload)
            print(json.dumps({'development_equivalence':True,'accounts':12,'proof_sha256':sha256(payload).hexdigest()}))
        elif args.command=='fit':
            manifest=fit_and_freeze()
            print(json.dumps({'manifest':str(manifest),'manifest_sha256':sha256(manifest.read_bytes()).hexdigest()}))
        else:
            require(args.manifest_sha256 is not None,'Specify externally frozen manifest SHA')
            manifest=FINAL/'final_model_manifest.json';meta=verify(manifest,expected_sha256=args.manifest_sha256)
            require(runtime_hashes()==meta['binding']['runtime_sha256'],'Frozen runtime changed')
            require(load(ROOT)['binding']=={k:meta['binding'][k] for k in load(ROOT)['binding']},'Frozen sources changed')
            require(dependencies()==meta['binding']['dependencies'],'Runtime dependency versions changed')
            for name in meta['files']:
                if name.startswith('models/'):
                    load_model(manifest,Path(name).stem,expected_manifest_sha256=args.manifest_sha256)
            print(json.dumps({'verified':True,'models':sum(n.startswith('models/') for n in meta['files']),'gates':meta['gates']}))
        require(not access['denied'],'Unexpected forbidden market access attempt')
    return 0


if __name__=='__main__':raise SystemExit(main())

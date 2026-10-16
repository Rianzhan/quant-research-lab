"""Immutable evaluation publication with frozen source and current evaluation code pins."""
from datetime import datetime,timezone
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4
import pandas as pd
from src.data.loader import digest,json_bytes,write_bytes
from src.data.pipeline import read_audit
from src.data.research_dataset import _inside
from src.data.research_policy import require
from src.evaluation.inputs import load_evaluation_inputs
from src.evaluation.tables import build_tables
from src.evaluation.plots import build_figures
from src.evaluation.report import build_report,scope_stability,concentration
from src.evaluation.cohort import EVAL_KEYS
from src.validation.pipeline import _csv

VERSION='phase55-v1'
TABLES={'classification_metrics.csv','regression_metrics.csv','baseline_metrics.csv','baseline_comparison.csv',
    'calibration_results.csv','per_asset_metrics.csv','fold_metrics.csv','paired_scope_comparison.csv','stability.csv',
    'loss_contributions.csv','baseline_parameters.csv','availability.csv','scope_stability.csv','concentration.csv'}
FIGURES={'roc_auc_by_fold.png','classification_vs_baselines.png','brier_vs_baselines.png','regression_errors.png',
    'regression_correlations.png','calibration_curves.png','paired_scope_by_year.png',
    'classification_etf_heatmap.png','regression_etf_heatmap.png'}
FILES=TABLES|{'figures/'+name for name in FIGURES}|{'audit.json','evaluation_policy.json','report.md'}


def binding(policy: dict,inputs: dict) -> dict:
    """Bind new code without changing the accepted historical Phase5 code binding."""
    code_root=Path(__file__).resolve().parents[2]
    code={p.relative_to(code_root).as_posix():digest(p.read_bytes().replace(b'\r\n',b'\n'))
          for p in sorted((code_root/'src').rglob('*.py'))}
    return {'input':inputs,'evaluation_policy_sha256':digest(json_bytes(policy)),
            'matplotlib_version':version('matplotlib'),'code_sha256':code,
            'code_hash_normalization':'CRLF to LF; other source bytes unchanged'}


def compute_artifacts(joined: pd.DataFrame,parameters: pd.DataFrame,availability: pd.DataFrame,policy: dict) -> dict[str,bytes]:
    """Pure frozen-prediction scoring; never fit, retune or generate trading signals."""
    tables=build_tables(joined,parameters,availability,policy)
    tables['scope_stability.csv']=scope_stability(tables['paired_scope_comparison.csv'])
    tables['concentration.csv']=concentration(tables['loss_contributions.csv'])
    artifacts={name:_csv(frame) for name,frame in tables.items()}
    artifacts.update(build_figures(tables));artifacts['report.md']=build_report(tables,joined,policy)
    artifacts['evaluation_policy.json']=json_bytes(policy)
    market=joined[EVAL_KEYS].drop(columns='task').drop_duplicates()
    audit={'status':'complete','prediction_rows':len(joined),'unique_market_sample_keys':len(market),
           'unique_evaluation_keys':len(joined[EVAL_KEYS].drop_duplicates()),
           'test_date_min':joined.date.min(),'test_date_max':joined.date.max(),
           'test_folds':sorted(joined.fold_id.unique()),'classification_threshold':.5,
           'all_comparisons_exact_keyed':True,'paired_status_counts':tables['paired_scope_comparison.csv'].status.value_counts().to_dict(),
           'model_training_performed':False,'new_model_predictions_performed':False,'models_deserialized':False,
           'model_selection_performed':False,'threshold_search_performed':False,'probability_calibration_fitted':False,
           'backtest_total_return_ready':False,'holdout_files_opened':False,
           'uncertainty':'descriptive_only; overlapping labels and correlated assets are not independent',
           'undefined_metrics':'NA with reason; empty calibration bins retained',
           'warnings':['Frozen probabilities are uncalibrated; evaluation does not calibrate them.',
                       '588000/2022 remains unavailable; no fabricated predictions.',
                       'No transaction-cost or total-return accounting; no profitability inference.']}
    artifacts['audit.json']=json_bytes(audit)
    require(set(artifacts)==FILES,'Unexpected evaluation artifact set')
    return artifacts


def publish(config: dict,policy: dict,root: Path) -> Path:
    """Publish an atomic new result directory; never silently overwrite a frozen result."""
    root=Path(root).resolve();joined,parameters,availability,inputs=load_evaluation_inputs(config,policy,root)
    pins=binding(policy,inputs);artifacts=compute_artifacts(joined,parameters,availability,policy)
    *_,after=load_evaluation_inputs(config,policy,root)
    require(pins==binding(policy,after),'Evaluation inputs/code changed during publication')
    hashes={n:digest(b) for n,b in artifacts.items()};key=digest(json_bytes({'version':VERSION,'bindings':pins,'files':hashes}))
    parent=root/'results/reports';parent.mkdir(parents=True,exist_ok=True);target=parent/(VERSION+'-'+key[:16])
    if target.exists():
        verify(target/'manifest.json',config,policy,root)
        return target/'manifest.json'
    pending=parent/('.pending-'+uuid4().hex);pending.mkdir()
    for name,payload in artifacts.items():
        path=pending/name;path.parent.mkdir(parents=True,exist_ok=True);write_bytes(path,payload)
    meta={'kind':'frozen_prediction_evaluation','status':'complete','version':VERSION,'build_key':key,
        'created_at':datetime.now(timezone.utc).isoformat(),'bindings':pins,'files':hashes,
        'scope':'development_only','model_training_performed':False,'performance_metrics_computed':True,
        'model_selection_performed':False,'backtest_total_return_ready':False,'holdout_files_opened':False}
    payload=json_bytes(meta);write_bytes(pending/'manifest.json',payload)
    write_bytes(pending/'manifest.sha256',(digest(payload)+'\n').encode('ascii'));pending.rename(target)
    return target/'manifest.json'


def verify(manifest: Path,config: dict,policy: dict,root: Path) -> dict:
    """Verify source lineage and artifacts without refitting or rerunning predictions."""
    root=Path(root).resolve();manifest=Path(manifest).resolve()
    require(manifest.is_relative_to(root/'results/reports') and manifest.name=='manifest.json'
            and not any(p.startswith('.pending-') for p in manifest.parts),'Invalid evaluation publication path')
    meta=read_audit(manifest)
    require(meta['kind']=='frozen_prediction_evaluation' and meta['status']=='complete' and meta['version']==VERSION
            and meta['scope']=='development_only' and meta['performance_metrics_computed'] is True,'Unsupported evaluation publication')
    require(all(meta[k] is False for k in ['model_training_performed','model_selection_performed',
            'backtest_total_return_ready','holdout_files_opened']),'Unapproved evaluation scope')
    *_,inputs=load_evaluation_inputs(config,policy,root)
    require(meta['bindings']==binding(policy,inputs),'Evaluation input/policy/code binding changed')
    require(set(meta['files'])==FILES,'Unexpected evaluation artifact set')
    for name,sha in meta['files'].items():
        require(digest(_inside(manifest.parent,name).read_bytes())==sha,'Evaluation artifact hash mismatch: '+name)
    key=digest(json_bytes({'version':VERSION,'bindings':meta['bindings'],'files':meta['files']}))
    require(meta['build_key']==key and manifest.parent.name==VERSION+'-'+key[:16],'Evaluation build key mismatch')
    return {'status':'verified','version':VERSION,'build_key':key,'files_sha256':meta['files'],
            'model_training_performed':False,'backtest_total_return_ready':False,'holdout_files_opened':False}

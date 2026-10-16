"""Immutable, source-bound local model artifacts and verified trusted deserialization."""
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
import pickle
import platform
import re
from uuid import uuid4
from src.data.loader import digest, json_bytes, write_bytes
from src.data.pipeline import read_audit
from src.data.research_dataset import _inside
from src.data.research_policy import require
from src.models.config import validate_experiment
from src.models.experiment import run_experiment
from src.models.inputs import load_model_data
from src.utils.config import config_digest, resolve_paths

VERSION='phase5-v1'
STATIC_FILES={'predictions.csv','fit_runs.csv','fit_details.json','audit.json','experiment_policy.json'}
MODEL_PATH=re.compile(r'models/[A-Za-z0-9_-]+/(?:per_asset/(?:SSE|SZSE)_[0-9]{6}|pooled/ALL)/(?:logistic_classification|random_forest_(?:classification|regression)|xgboost_(?:classification|regression))\.pkl')


def runtime_versions() -> dict:
    """Record the actual local numeric runtime, not only requested requirements."""
    return {'python':platform.python_version(),'platform':platform.platform(),
            **{name:version(name) for name in ['numpy','pandas','scikit-learn','scipy','xgboost','joblib','threadpoolctl']}}


def _binding(config: dict, experiment: dict, inputs: dict, root: Path) -> dict:
    code_root=Path(__file__).resolve().parents[2]
    code={p.relative_to(code_root).as_posix():digest(p.read_bytes().replace(b'\r\n',b'\n'))
          for p in sorted((code_root/'src').rglob('*.py'))}
    return {'input':inputs,'config_sha256':config_digest(config),'experiment_sha256':digest(json_bytes(experiment)),
            'requirements_lock_sha256':digest((root/'requirements.lock.txt').read_bytes()),
            'runtime':runtime_versions(),'code_sha256':code,'code_hash_normalization':'CRLF to LF; other source bytes unchanged'}


def publish(config: dict, experiment: dict, root: Path) -> Path:
    """Fit and publish completely, or reuse only an intact identical publication."""
    root=Path(root).resolve()
    frame,membership,calendar,validation,registration,inputs=load_model_data(config,experiment,root)
    binding=_binding(config,experiment,inputs,root)
    artifacts=run_experiment(frame,membership,calendar,validation,registration,config,experiment)
    *_,after=load_model_data(config,experiment,root)
    require(binding==_binding(config,experiment,after,root),'Model inputs/code/runtime changed during fitting')
    hashes={name:digest(payload) for name,payload in artifacts.items()}
    key=digest(json_bytes({'version':VERSION,'bindings':binding,'files':hashes}))
    parent=resolve_paths(config,root)['processed']/'models';parent.mkdir(parents=True,exist_ok=True)
    target=parent/(VERSION+'-'+key[:16])
    if target.exists():
        verify(target/'manifest.json',config,experiment,root)
        return target/'manifest.json'
    pending=parent/('.pending-'+uuid4().hex);pending.mkdir()
    for name,payload in artifacts.items():
        path=pending/name;path.parent.mkdir(parents=True,exist_ok=True);write_bytes(path,payload)
    meta={'kind':'development_model_experiment','status':'complete','version':VERSION,'build_key':key,
        'created_at':datetime.now(timezone.utc).isoformat(),'bindings':binding,'files':hashes,
        'scope':'development_only','predictors':registration['shared_whitelist'],'model_scopes':experiment['scopes'],
        'model_training_performed':True,'model_selection_performed':False,'performance_metrics_computed':False,
        'backtest_total_return_ready':False,'holdout_files_opened':False}
    payload=json_bytes(meta);write_bytes(pending/'manifest.json',payload)
    write_bytes(pending/'manifest.sha256',(digest(payload)+'\n').encode('ascii'))
    pending.rename(target)
    return target/'manifest.json'


def verify(manifest: Path, config: dict, experiment: dict, root: Path) -> dict:
    """Check all hashes and frozen sources without deserializing any model."""
    root=Path(root).resolve();manifest=Path(manifest).resolve()
    parent=(resolve_paths(config,root)['processed']/'models').resolve()
    require(manifest.is_relative_to(parent) and manifest.name=='manifest.json'
            and not any(p.startswith('.pending-') for p in manifest.parts),'Invalid/incomplete model publication path')
    meta=read_audit(manifest)
    require(meta['kind']=='development_model_experiment' and meta['status']=='complete' and meta['version']==VERSION
            and meta['scope']=='development_only' and meta['model_training_performed'] is True,'Unsupported model publication')
    require(all(meta[k] is False for k in ['model_selection_performed','performance_metrics_computed',
            'backtest_total_return_ready','holdout_files_opened']),'Unapproved model publication scope')
    *_,registration,inputs=load_model_data(config,experiment,root)
    validate_experiment(experiment,config,registration)
    require(meta['predictors']==registration['shared_whitelist'] and meta['model_scopes']==experiment['scopes'],
            'Predictor/scope registration changed')
    require(meta['bindings']==_binding(config,experiment,inputs,root),'Model input/config/code/runtime binding changed')
    require(STATIC_FILES<=set(meta['files']) and all(n in STATIC_FILES or MODEL_PATH.fullmatch(n) for n in meta['files']),
            'Unexpected model artifact set')
    for name,sha in meta['files'].items():
        path=_inside(manifest.parent,name)
        require(digest(path.read_bytes())==sha,'Model artifact hash mismatch: '+name)
    key=digest(json_bytes({'version':VERSION,'bindings':meta['bindings'],'files':meta['files']}))
    require(key==meta['build_key'] and manifest.parent.name==VERSION+'-'+key[:16],'Model build key mismatch')
    return {'status':'verified','version':VERSION,'scope':'development_only','build_key':key,
            'model_files':sum(n.endswith('.pkl') for n in meta['files']),'files_sha256':meta['files'],
            'model_training_performed':True,'backtest_total_return_ready':False,'holdout_files_opened':False}


def load_trained_model(manifest: Path, name: str, config: dict, experiment: dict, root: Path):
    """Load only a verified model in this trusted local publication.

    Hash integrity is not an author signature. Never use external/untrusted pickle
    files or external manifests, even if somebody supplies matching hashes.
    """
    result=verify(manifest,config,experiment,root)
    require(name in result['files_sha256'] and MODEL_PATH.fullmatch(name) is not None,'Unregistered model path')
    payload=_inside(Path(manifest).resolve().parent,name).read_bytes()
    require(digest(payload)==result['files_sha256'][name],'Model changed before deserialization')
    return pickle.loads(payload)

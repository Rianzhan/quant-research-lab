"""Offline, immutable fold audits; no model fit or new feature calculation."""
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
import pandas as pd
from src.data.loader import digest, json_bytes, write_bytes
from src.data.pipeline import read_audit
from src.data.research_policy import require
from src.utils.config import resolve_paths
from src.validation.audit import audit_folds
from src.validation.inputs import load_inputs
from src.validation.walk_forward import build_membership

VERSION='phase4-v1'
FILES={'membership.csv','folds.csv','asset_folds.csv','target_distributions.csv','purged_samples.csv',
       'audit.json','feature_registration.json','validation_policy.json'}


def _csv(frame: pd.DataFrame) -> bytes:
    """Stable floats, missing fields and LF line endings for reproducible output."""
    return frame.to_csv(index=False,float_format='%.17g',lineterminator='\n',na_rep='').encode('utf-8')


def compute_artifacts(frame: pd.DataFrame, calendar: list[str], config: dict, policy: dict) -> dict[str,bytes]:
    """Split first, then audit labels without feeding their statistics back."""
    membership=build_membership(frame,calendar,config,policy)
    audit=audit_folds(frame,membership,calendar,config,policy)
    return {'membership.csv':_csv(membership),'folds.csv':_csv(pd.DataFrame(audit['folds'])),
        'asset_folds.csv':_csv(pd.DataFrame(audit['assets'])),
        'target_distributions.csv':_csv(pd.DataFrame(audit['distributions'])),
        'purged_samples.csv':_csv(membership.loc[membership.partition=='purged']),
        'audit.json':json_bytes(audit),'feature_registration.json':json_bytes(policy['feature_registration']),
        'validation_policy.json':json_bytes(policy)}


def _binding(policy: dict, source: dict) -> dict:
    root=Path(__file__).resolve().parents[2]
    code={path.relative_to(root).as_posix():digest(path.read_bytes().replace(b'\r\n',b'\n'))
          for path in sorted((root/'src').rglob('*.py'))}
    return {'input':source,'validation_policy_sha256':digest(json_bytes(policy)),
            'code_sha256':code,'code_hash_normalization':'CRLF to LF; other source bytes unchanged'}


def publish(config: dict, policy: dict, root: Path) -> Path:
    """Atomically publish all folds, or verify/reuse an identical publication."""
    root=Path(root).resolve();frame,calendar,source=load_inputs(config,policy,root)
    binding=_binding(policy,source);artifacts=compute_artifacts(frame,calendar,config,policy)
    _,_,after=load_inputs(config,policy,root)
    require(binding==_binding(policy,after),'Inputs/code changed during fold construction')
    hashes={name:digest(value) for name,value in artifacts.items()}
    key=digest(json_bytes({'version':VERSION,'bindings':binding,'files':hashes}))
    parent=resolve_paths(config,root)['processed']/'validation';parent.mkdir(parents=True,exist_ok=True)
    target=parent/(VERSION+'-'+key[:16])
    if target.exists():
        verify(target/'manifest.json',config,policy,root)
        return target/'manifest.json'
    pending=parent/('.pending-'+uuid4().hex);pending.mkdir()
    for name,value in artifacts.items(): write_bytes(pending/name,value)
    meta={'kind':'development_walk_forward_audit','status':'complete','version':VERSION,'build_key':key,
        'created_at':datetime.now(timezone.utc).isoformat(),'bindings':binding,'files':hashes,
        'scope':'development_only','model_training_performed':False,'model_training_ready':False,
        'new_features_materialized':False,'backtest_total_return_ready':False,'holdout_files_opened':False}
    payload=json_bytes(meta);write_bytes(pending/'manifest.json',payload)
    write_bytes(pending/'manifest.sha256',(digest(payload)+'\n').encode('ascii'))
    pending.rename(target)
    return target/'manifest.json'


def verify(manifest: Path, config: dict, policy: dict, root: Path) -> dict:
    """Check bytes and current bindings without touching sealed/mixed price files."""
    root=Path(root).resolve();manifest=Path(manifest).resolve()
    parent=resolve_paths(config,root)['processed']/'validation'
    require(manifest.is_relative_to(parent) and not any(p.startswith('.pending-') for p in manifest.parts),'Invalid/incomplete fold publication path')
    meta=read_audit(manifest)
    require(meta['kind']=='development_walk_forward_audit' and meta['status']=='complete' and meta['version']==VERSION
            and meta['scope']=='development_only','Unsupported fold publication')
    require(all(meta[name] is False for name in ['model_training_performed','model_training_ready','new_features_materialized',
                                               'backtest_total_return_ready','holdout_files_opened']),'Unexpected research scope flags')
    _,_,source=load_inputs(config,policy,root)
    require(meta['bindings']==_binding(policy,source),'Fold input/config/code binding changed')
    require(set(meta['files'])==FILES,'Fold artifact set mismatch')
    for name,sha in meta['files'].items():
        require(digest((manifest.parent/name).read_bytes())==sha,'Fold artifact hash mismatch: '+name)
    key=digest(json_bytes({'version':VERSION,'bindings':meta['bindings'],'files':meta['files']}))
    require(key==meta['build_key'] and manifest.parent.name==VERSION+'-'+key[:16],'Fold build key mismatch')
    return {'status':'verified','scope':'development_only','files_sha256':meta['files'],
            'model_training_performed':False,'backtest_total_return_ready':False,'holdout_files_opened':False}

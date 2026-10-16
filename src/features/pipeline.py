"""Offline development-only features/targets and reproducible immutable publication."""
from __future__ import annotations
from datetime import datetime,timezone
import json
from pathlib import Path
from uuid import uuid4
import pandas as pd
from src.data.development_calendar import load_development_calendar
from src.data.loader import digest,json_bytes,write_bytes
from src.data.pipeline import read_audit
from src.data.research_dataset import check_research_gate,load_development,_inside
from src.data.research_policy import require
from src.features.audit import audit_samples
from src.features.builder import build_features,feature_specs
from src.labels.forward_return import build_labels
from src.utils.config import config_digest,resolve_paths
from src.utils.research_rows import KEYS

VERSION='phase3-v1'


def _csv(frame: pd.DataFrame) -> bytes:
    """Stable row/column order and full float precision; missing remains empty."""
    return frame.to_csv(index=False,float_format='%.17g',lineterminator='\n',na_rep='').encode('utf-8')


def compute_artifacts(panel: pd.DataFrame, config: dict, calendar: list[str]) -> dict[str,bytes]:
    """Pure calculation, retaining all observed rows and separating future targets."""
    features=build_features(panel,config,calendar)
    labels=build_labels(panel,config,calendar)
    audit=audit_samples(panel,features,labels,config,calendar)
    names=[s['name'] for s in feature_specs(config)]
    eligibility=features[KEYS].copy()
    eligibility['features_complete']=features[names].notna().all(axis=1)
    eligibility['label_valid']=labels.label_valid
    eligibility['supervised_eligible']=eligibility.features_complete&eligibility.label_valid
    return {'features.csv':_csv(features),'labels.csv':_csv(labels),'sample_eligibility.csv':_csv(eligibility),
            'features_head20.csv':_csv(features.head(20)),'audit.json':json_bytes(audit)}


def build_artifacts(config: dict, root: Path) -> dict[str,bytes]:
    """The sole default input path: approved development quotes and bounded dates."""
    development=load_development(config,root)
    calendar=load_development_calendar(config,root)
    return compute_artifacts(development,config,calendar)


def _bindings(config: dict, root: Path) -> dict:
    """Bind data policy, exact development input, calendar, configuration and code."""
    gate=check_research_gate(config,root)
    require(gate['research_ready'],'Research gate closed: '+str(gate['blockers']))
    source=_inside(root,config['governance']['dataset_manifest'])
    metadata=read_audit(source)
    phase2=read_audit(_inside(root,config['governance']['phase2_manifest']))
    code_root=Path(__file__).resolve().parents[2]
    # Git may convert text line endings on checkout; bind Python text with LF.
    # Market data and the accepted policy retain their exact-byte SHA contracts.
    code={p.relative_to(code_root).as_posix():digest(p.read_bytes().replace(b'\r\n',b'\n'))
          for p in sorted((code_root/'src').rglob('*.py'))}
    return {'config_sha256':config_digest(config),'research_manifest':config['governance']['dataset_manifest'],
            'research_manifest_sha256':digest(source.read_bytes()),'development_sha256':metadata['files']['development.csv'],
            'calendar_manifest_sha256':phase2['calendar_input']['manifest_sha256'],
            'policy_sha256':config['governance']['policy_sha256'],'code_sha256':code,
            'code_hash_normalization':'CRLF to LF; other source bytes unchanged'}


def publish(config: dict, root: Path) -> Path:
    """Publish atomically, or verify/reuse the same content; never overwrite."""
    root=Path(root).resolve(); bindings=_bindings(config,root)
    artifacts=build_artifacts(config,root)
    require(bindings==_bindings(config,root),'Inputs or code changed during construction')
    hashes={name:digest(value) for name,value in artifacts.items()}
    build_key=digest(json_bytes({'version':VERSION,'bindings':bindings,'files':hashes}))
    parent=resolve_paths(config,root)['processed']/'features';parent.mkdir(parents=True,exist_ok=True)
    target=parent/(VERSION+'-'+build_key[:16])
    if target.exists():
        verify(target/'manifest.json',config,root)
        return target/'manifest.json'
    pending=parent/('.pending-'+uuid4().hex);pending.mkdir()
    for name,payload in artifacts.items():
        write_bytes(pending/name,payload)
    metadata={'kind':'development_features_labels','status':'complete','formula_version':VERSION,
              'created_at':datetime.now(timezone.utc).isoformat(),'build_key':build_key,'bindings':bindings,
              'feature_parameters':config['features'],'prediction_horizon_days':config['timing']['prediction_horizon_days'],
              'classification_return_threshold':config['labels']['classification_return_threshold'],
              'predictors':[s['name'] for s in feature_specs(config)],'files':hashes,
              'scope':'development_only','backtest_total_return_ready':False,'model_training_performed':False}
    payload=json_bytes(metadata)
    write_bytes(pending/'manifest.json',payload)
    write_bytes(pending/'manifest.sha256',(digest(payload)+'\n').encode('ascii'))
    pending.rename(target)
    return target/'manifest.json'


def verify(manifest: Path, config: dict, root: Path) -> dict:
    """Verify current inputs/code and saved bytes; do not train or unlock holdout."""
    root=Path(root).resolve();manifest=Path(manifest).resolve()
    parent=resolve_paths(config,root)['processed']/'features'
    require(manifest.is_relative_to(parent) and not any(p.startswith('.pending-') for p in manifest.parts),'Invalid/incomplete feature path')
    metadata=read_audit(manifest)
    require(metadata['kind']=='development_features_labels' and metadata['status']=='complete'
            and metadata['formula_version']==VERSION,'Unsupported feature publication')
    require(metadata['bindings']==_bindings(config,root),'Feature input/code binding changed')
    expected={'features.csv','labels.csv','sample_eligibility.csv','features_head20.csv','audit.json'}
    require(set(metadata['files'])==expected,'Unexpected feature artifact set')
    for name,sha in metadata['files'].items():
        require(digest((manifest.parent/name).read_bytes())==sha,'Feature artifact hash mismatch: '+name)
    key=digest(json_bytes({'version':VERSION,'bindings':metadata['bindings'],'files':metadata['files']}))
    require(key==metadata['build_key'] and manifest.parent.name==VERSION+'-'+key[:16],'Feature build-key mismatch')
    return {'status':'verified','formula_version':VERSION,'files_sha256':metadata['files'],
            'scope':'development_only','backtest_total_return_ready':False}

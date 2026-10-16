"""Immutable Phase4.5 features sharing the accepted Phase4 sample contract."""
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
import pandas as pd
from src.data.loader import digest, json_bytes, write_bytes
from src.data.pipeline import read_audit
from src.data.research_dataset import _inside
from src.data.research_policy import require
from src.features.scale_free import build_scale_free, validate_registration
from src.utils.config import resolve_paths
from src.utils.research_rows import ASSET, KEYS
from src.validation.config import load_validation
from src.validation.inputs import _frozen_code_hashes, _read, load_inputs
from src.validation.pipeline import FILES as PHASE4_FILES, _csv
from src.validation.walk_forward import build_membership

VERSION='phase45-v1'
FILES={'features.csv','features_head20.csv','sample_eligibility.csv','feature_counts.csv',
       'feature_counts_by_asset.csv','feature_registration.json','fold_membership.csv','audit.json'}


def load_source(config: dict, registration: dict, root: Path) -> tuple[pd.DataFrame,list[str],dict,dict,bytes]:
    """Verify historical sources by pinned Git hashes, never old holdout gates."""
    root=Path(root).resolve();source=registration['source']
    vpath=_inside(root,source['validation_policy'])
    require(digest(vpath.read_bytes())==source['validation_policy_sha256'],'Frozen validation policy hash changed')
    validation=load_validation(vpath,config);validate_registration(registration,validation,config)
    path=_inside(root,source['validation_manifest'])
    require(digest(path.read_bytes())==source['validation_manifest_sha256'],'Frozen Phase4 manifest hash changed')
    meta=read_audit(path)
    require(meta['kind']=='development_walk_forward_audit' and meta['status']=='complete' and meta['version']=='phase4-v1'
            and meta['scope']=='development_only','Unapproved Phase4 publication')
    require(all(meta[name] is False for name in ['model_training_performed','model_training_ready','new_features_materialized',
                                               'backtest_total_return_ready','holdout_files_opened']),'Unapproved Phase4 scope')
    require(meta['bindings']['code_sha256']==_frozen_code_hashes(source['validation_code_commit']),
            'Historical Phase4 code mismatch')
    require(meta['bindings']['validation_policy_sha256']==digest(json_bytes(validation)),'Phase4 policy binding changed')
    require(set(meta['files'])==PHASE4_FILES,'Unexpected Phase4 files')
    for name,sha in meta['files'].items():
        require(digest((path.parent/name).read_bytes())==sha,'Frozen Phase4 artifact hash changed: '+name)
    key=digest(json_bytes({'version':'phase4-v1','bindings':meta['bindings'],'files':meta['files']}))
    require(key==meta['build_key'] and path.parent.name=='phase4-v1-'+key[:16],'Frozen Phase4 build key mismatch')
    frame,calendar,inputs=load_inputs(config,validation,root)
    require(inputs==meta['bindings']['input'],'Phase4 development input binding changed')
    binding={'accepted_phase3_input':inputs,'accepted_phase4_source':source,
             'membership_sha256':meta['files']['membership.csv'],'holdout_files_opened':False}
    return frame,calendar,validation,binding,(path.parent/'membership.csv').read_bytes()


def compute_artifacts(frame: pd.DataFrame, calendar: list[str], config: dict, validation: dict,
                      registration: dict, expected_membership: bytes) -> dict[str,bytes]:
    """Build 17 inputs and count availability; do not recalculate target statistics."""
    features=build_scale_free(frame,calendar,config,validation,registration)
    source=frame.sort_values(KEYS,kind='stable').reset_index(drop=True)
    names=registration['shared_whitelist'];raw=validation['feature_registration']['shared_raw_whitelist']
    complete=features[names].notna().all(axis=1)
    require(complete.equals(source[raw].notna().all(axis=1)),'Scale-free eligibility differs from frozen Phase4')
    membership=_csv(build_membership(source,calendar,config,validation))
    require(membership==expected_membership,'Frozen Phase4 membership changed')
    eligibility=features[KEYS].copy()
    eligibility['features_complete']=complete
    eligibility['label_valid']=source.label_valid
    eligibility['supervised_eligible']=complete&source.label_valid
    counts=[];by_asset=[]
    for name in names:
        counts.append({'feature':name,'rows':len(features),'valid_samples':int(features[name].notna().sum()),
                       'missing_samples':int(features[name].isna().sum())})
        for identity,group in features.groupby(ASSET,sort=True):
            by_asset.append({'feature':name,'exchange':identity[0],'asset_code':identity[1],
                'rows':len(group),'valid_samples':int(group[name].notna().sum()),'missing_samples':int(group[name].isna().sum())})
    audit={'version':VERSION,'scope':'development_only','rows':len(features),'feature_count':len(names),
           'features_complete':int(complete.sum()),'label_valid':int(source.label_valid.sum()),
           'supervised_eligible':int(eligibility.supervised_eligible.sum()),'new_ratio_features':8,
           'phase4_membership_unchanged':True,'source_phase3_eligibility_unchanged':True,
           'model_training_performed':False,'backtest_total_return_ready':False,'holdout_files_opened':False}
    registered={**registration,'feature_inputs_ready':True,
                'formulas':validation['feature_registration']['planned_replacements'],
                'label_source':validation['source']['feature_manifest']}
    return {'features.csv':_csv(features),'features_head20.csv':_csv(features.head(20)),
        'sample_eligibility.csv':_csv(eligibility),'feature_counts.csv':_csv(pd.DataFrame(counts)),
        'feature_counts_by_asset.csv':_csv(pd.DataFrame(by_asset)),
        'feature_registration.json':json_bytes(registered),'fold_membership.csv':membership,'audit.json':json_bytes(audit)}


def _binding(registration: dict, inputs: dict) -> dict:
    code_root=Path(__file__).resolve().parents[2]
    code={p.relative_to(code_root).as_posix():digest(p.read_bytes().replace(b'\r\n',b'\n'))
          for p in sorted((code_root/'src').rglob('*.py'))}
    return {'input':inputs,'registration_sha256':digest(json_bytes(registration)),
            'code_sha256':code,'code_hash_normalization':'CRLF to LF; other source bytes unchanged'}


def publish(config: dict, registration: dict, root: Path) -> Path:
    """Write a new immutable version or verify/reuse identical bytes."""
    root=Path(root).resolve();frame,calendar,validation,inputs,membership=load_source(config,registration,root)
    binding=_binding(registration,inputs)
    artifacts=compute_artifacts(frame,calendar,config,validation,registration,membership)
    *_,after,_members=load_source(config,registration,root)
    require(binding==_binding(registration,after),'Inputs/code changed during scale-free construction')
    hashes={name:digest(payload) for name,payload in artifacts.items()}
    key=digest(json_bytes({'version':VERSION,'bindings':binding,'files':hashes}))
    parent=resolve_paths(config,root)['processed']/'features';parent.mkdir(parents=True,exist_ok=True)
    target=parent/(VERSION+'-'+key[:16])
    if target.exists():
        verify(target/'manifest.json',config,registration,root)
        return target/'manifest.json'
    pending=parent/('.pending-'+uuid4().hex);pending.mkdir()
    for name,payload in artifacts.items(): write_bytes(pending/name,payload)
    meta={'kind':'development_scale_free_features','status':'complete','version':VERSION,'build_key':key,
        'created_at':datetime.now(timezone.utc).isoformat(),'bindings':binding,'files':hashes,
        'predictors':registration['shared_whitelist'],'model_scopes':registration['model_scopes'],
        'scope':'development_only','feature_inputs_ready':True,'model_training_performed':False,
        'backtest_total_return_ready':False,'holdout_files_opened':False}
    payload=json_bytes(meta);write_bytes(pending/'manifest.json',payload)
    write_bytes(pending/'manifest.sha256',(digest(payload)+'\n').encode('ascii'))
    pending.rename(target)
    return target/'manifest.json'


def verify(manifest: Path, config: dict, registration: dict, root: Path) -> dict:
    """Verify all new bytes and frozen source bindings without opening holdout."""
    root=Path(root).resolve();manifest=Path(manifest).resolve()
    parent=resolve_paths(config,root)['processed']/'features'
    require(manifest.is_relative_to(parent) and not any(p.startswith('.pending-') for p in manifest.parts),'Invalid/incomplete feature path')
    meta=read_audit(manifest)
    require(meta['kind']=='development_scale_free_features' and meta['status']=='complete' and meta['version']==VERSION
            and meta['scope']=='development_only' and meta['feature_inputs_ready'] is True,'Unsupported scale-free publication')
    require(all(meta[name] is False for name in ['model_training_performed','backtest_total_return_ready','holdout_files_opened']),
            'Unexpected scope flags')
    require(meta['predictors']==registration['shared_whitelist'] and meta['model_scopes']==registration['model_scopes'],
            'Shared feature whitelist changed')
    *_,inputs,_members=load_source(config,registration,root)
    require(meta['bindings']==_binding(registration,inputs),'Scale-free input/config/code binding changed')
    require(set(meta['files'])==FILES,'Unexpected scale-free artifact set')
    for name,sha in meta['files'].items():
        require(digest((manifest.parent/name).read_bytes())==sha,'Scale-free artifact hash mismatch: '+name)
    key=digest(json_bytes({'version':VERSION,'bindings':meta['bindings'],'files':meta['files']}))
    require(key==meta['build_key'] and manifest.parent.name==VERSION+'-'+key[:16],'Scale-free build key mismatch')
    return {'status':'verified','version':VERSION,'scope':'development_only','files_sha256':meta['files'],
            'feature_inputs_ready':True,'model_training_performed':False,'backtest_total_return_ready':False,'holdout_files_opened':False}


def load_scale_free(manifest: Path, config: dict, registration: dict, root: Path) -> pd.DataFrame:
    """One common predictor table for both scopes; no holdout or scope override."""
    verify(manifest,config,registration,root)
    features=_read(Path(manifest).parent/'features.csv')
    require(list(features)==KEYS+registration['shared_whitelist'],'Unexpected scale-free columns')
    return features

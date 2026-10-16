"""Model-only adapter for historical pinned Phase4.5 development artifacts."""
from io import BytesIO
from pathlib import Path
import pandas as pd
from src.data.loader import digest, json_bytes
from src.data.pipeline import read_audit
from src.data.research_dataset import _inside
from src.data.research_policy import require
from src.features.scale_free import read_registration
from src.features.scale_free_pipeline import FILES, load_source
from src.models.config import validate_experiment
from src.models.experiment import audit_membership
from src.utils.research_rows import KEYS
from src.validation.inputs import _frozen_code_hashes, _read


def load_model_data(config: dict, experiment: dict, root: Path) -> tuple:
    """Return approved inputs plus their identity; never open holdout or raw quotes."""
    root=Path(root).resolve();source=experiment['source']
    rpath=_inside(root,source['feature_registration'])
    require(digest(rpath.read_bytes())==source['feature_registration_sha256'],'Frozen scale-free registration hash changed')
    registration=read_registration(rpath);validate_experiment(experiment,config,registration)
    frame,calendar,validation,input_binding,members=load_source(config,registration,root)
    path=_inside(root,source['feature_manifest'])
    require(digest(path.read_bytes())==source['feature_manifest_sha256'],'Frozen Phase4.5 manifest hash changed')
    meta=read_audit(path)
    require(meta['kind']=='development_scale_free_features' and meta['status']=='complete'
            and meta['version']=='phase45-v1' and meta['scope']=='development_only'
            and meta['feature_inputs_ready'] is True,'Unapproved Phase4.5 publication')
    require(all(meta[k] is False for k in ['model_training_performed','backtest_total_return_ready','holdout_files_opened']),
            'Unapproved feature scope')
    require(meta['predictors']==registration['shared_whitelist'] and meta['model_scopes']==registration['model_scopes'],
            'Changed common predictor whitelist')
    require(meta['bindings']['input']==input_binding
            and meta['bindings']['registration_sha256']==digest(json_bytes(registration)),'Changed Phase4.5 input binding')
    require(meta['bindings']['code_sha256']==_frozen_code_hashes(source['feature_code_commit']),
            'Historical Phase4.5 code mismatch')
    require(set(meta['files'])==FILES,'Unexpected Phase4.5 artifact set')
    for name,sha in meta['files'].items():
        require(digest((path.parent/name).read_bytes())==sha,'Frozen Phase4.5 artifact hash changed: '+name)
    key=digest(json_bytes({'version':'phase45-v1','bindings':meta['bindings'],'files':meta['files']}))
    require(meta['build_key']==key and path.parent.name=='phase45-v1-'+key[:16],'Phase4.5 build key mismatch')
    require((path.parent/'fold_membership.csv').read_bytes()==members,'Frozen membership bytes changed')
    features=_read(path.parent/'features.csv');names=registration['shared_whitelist']
    require(list(features)==KEYS+names,'Unexpected feature columns')
    original=frame.sort_values(KEYS,kind='stable').reset_index(drop=True)
    require(features[KEYS].equals(original[KEYS]),'Feature and label keys do not align')
    labels=['label_valid','label_end_date','classification_target','regression_target']
    features=features.merge(original[KEYS+labels],on=KEYS,validate='one_to_one')
    membership=pd.read_csv(BytesIO(members),dtype={'date':str,'exchange':str,'asset_code':str,
                            'label_end_date':str},keep_default_na=False)
    audit_membership(features,membership,calendar,validation,registration,config,experiment)
    binding={'accepted_feature_source':source,'accepted_phase45_bindings':meta['bindings'],
             'phase45_files_sha256':meta['files'],'holdout_files_opened':False}
    return features,membership,calendar,validation,registration,binding

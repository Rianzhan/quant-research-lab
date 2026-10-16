"""Historical Phase 5 verification adapter; hash model bytes without deserialization."""
from io import BytesIO
from pathlib import Path
import pandas as pd
from src.data.loader import digest,json_bytes
from src.data.pipeline import read_audit
from src.data.research_dataset import _inside
from src.data.research_policy import require
from src.evaluation.config import validate_policy
from src.evaluation.cohort import prepare_cohort
from src.models.config import read_experiment
from src.models.inputs import load_model_data
from src.models.pipeline import STATIC_FILES,MODEL_PATH,runtime_versions
from src.utils.config import config_digest
from src.validation.inputs import _frozen_code_hashes


def _accepted_manifest(config: dict,policy: dict,root: Path,experiment: dict,registration: dict,inputs: dict) -> tuple:
    source=policy['source'];path=_inside(root,source['model_manifest'])
    require(path.name=='manifest.json' and not any(p.startswith('.pending-') for p in path.parts),'Incomplete model source')
    require(digest(path.read_bytes())==source['model_manifest_sha256'],'Frozen Phase5 manifest hash changed')
    meta=read_audit(path)
    require(meta['kind']=='development_model_experiment' and meta['status']=='complete' and meta['version']=='phase5-v1'
            and meta['scope']=='development_only' and meta['model_training_performed'] is True,'Unapproved model source')
    require(all(meta[k] is False for k in ['model_selection_performed','performance_metrics_computed',
                'backtest_total_return_ready','holdout_files_opened']),'Unexpected Phase5 scope')
    binding={'input':inputs,'config_sha256':config_digest(config),'experiment_sha256':digest(json_bytes(experiment)),
        'requirements_lock_sha256':digest((root/'requirements.lock.txt').read_bytes()),'runtime':runtime_versions(),
        'code_sha256':_frozen_code_hashes(source['model_code_commit']),
        'code_hash_normalization':'CRLF to LF; other source bytes unchanged'}
    require(meta['bindings']==binding,'Frozen model historical input/code/runtime binding changed')
    require(meta['predictors']==registration['shared_whitelist'] and meta['model_scopes']==experiment['scopes'],
            'Frozen predictor/scope registration changed')
    require(STATIC_FILES<=set(meta['files']) and all(n in STATIC_FILES or MODEL_PATH.fullmatch(n) for n in meta['files']),
            'Unexpected model artifact set')
    for name,sha in meta['files'].items():
        require(digest(_inside(path.parent,name).read_bytes())==sha,'Frozen model artifact hash mismatch: '+name)
    key=digest(json_bytes({'version':'phase5-v1','bindings':meta['bindings'],'files':meta['files']}))
    require(key==meta['build_key'] and path.parent.name=='phase5-v1-'+key[:16],'Frozen model build key mismatch')
    require((path.parent/'experiment_policy.json').read_bytes()==json_bytes(experiment),'Frozen model policy mismatch')
    return path,meta


def load_evaluation_inputs(config: dict,policy: dict,root: Path) -> tuple:
    """Reuse exact frozen labels/membership/predictions; no raw market or holdout reads."""
    validate_policy(policy);root=Path(root).resolve();source=policy['source']
    ep=_inside(root,source['model_experiment'])
    require(digest(ep.read_bytes())==source['model_experiment_sha256'],'Frozen experiment file hash changed')
    experiment=read_experiment(ep)
    frame,members,calendar,validation,registration,inputs=load_model_data(config,experiment,root)
    require(frame.date.between(policy['development']['start'],policy['development']['end']).all(),'Non-development rows forbidden')
    path,meta=_accepted_manifest(config,policy,root,experiment,registration,inputs)
    def read(name: str) -> pd.DataFrame:
        payload=(path.parent/name).read_bytes()
        require(digest(payload)==meta['files'][name],'Frozen input changed during read')
        return pd.read_csv(BytesIO(payload),dtype={'asset_code':str,'exchange':str},keep_default_na=False,float_precision='round_trip')
    predictions=read('predictions.csv');runs=read('fit_runs.csv')
    records=runs.loc[runs.status.str.startswith('trained')]
    require(set(records.model_file)=={n for n in meta['files'] if n.endswith('.pkl')},'Model file/fit registry mismatch')
    require(all(meta['files'][row.model_file]==row.model_sha256 for row in records.itertuples()),'Fitted model digest mismatch')
    joined,parameters,availability=prepare_cohort(frame,members,calendar,validation,registration,config,experiment,predictions,runs)
    binding={'source':source,'accepted_model_bindings':meta['bindings'],'phase5_files_sha256':meta['files'],
             'holdout_files_opened':False}
    return joined,parameters,availability,binding

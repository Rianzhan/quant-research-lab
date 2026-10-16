"""Fixed Phase5 experiment registration, separate from frozen data configuration."""
from pathlib import Path
import re
import yaml
from src.data.research_policy import relative_path, require
from src.models.factory import MODEL_TASKS
from src.utils.config import _keys, _UniqueSafeLoader, validate_config

SCHEMA={
    'schema_version':None,'version':None,'development':dict.fromkeys(['start','end']),
    'source':dict.fromkeys(['feature_registration','feature_registration_sha256','feature_manifest',
                          'feature_manifest_sha256','feature_code_commit']),
    'scopes':None,'model_tasks':None,'preprocessing':dict.fromkeys(['logistic','random_forest','xgboost']),
    'hyperparameters_source':None,'missing_inputs':None,'classification_single_class':None,
    'fit_policy':None,'sample_weight':None,'early_stopping':None,'hyperparameter_search':None,
    'probability_calibration':None,'prediction_output':None,
}


def validate_experiment(policy: dict, config: dict, registration: dict) -> dict:
    """Reject unregistered behavior rather than silently ignoring configuration."""
    validate_config(config);_keys(policy,SCHEMA,'model_experiment')
    require(policy['schema_version']==1 and policy['version']=='phase5-v1','Unsupported model experiment')
    require(policy['development']==registration['development']=={'start':config['data']['start_date'],
            'end':config['data']['end_date']},'Models require the frozen development window')
    require(policy['scopes']==registration['model_scopes']==['per_asset','pooled'],'Both model scopes required')
    require(policy['model_tasks']==MODEL_TASKS,'Unregistered model/task combination')
    require(policy['preprocessing']=={'logistic':'train_only_standard_scaler','random_forest':'none','xgboost':'none'},
            'Unregistered preprocessing')
    expected={'hyperparameters_source':'config.yaml.models','missing_inputs':'reject',
        'classification_single_class':'unavailable_all_classifiers','fit_policy':'fresh_model_per_fold_and_scope',
        'sample_weight':'uniform_per_training_row','prediction_output':'continuous_probability_or_return_only'}
    require(all(policy[k]==v for k,v in expected.items()),'Unregistered experiment behavior')
    require(all(policy[k] is False for k in ['early_stopping','hyperparameter_search','probability_calibration']),
            'Tuning/calibration/early stopping are not registered')
    require(config['models']['n_jobs']==1,'One model worker required')
    for key in ['feature_registration','feature_manifest']: relative_path(policy['source'][key])
    for key,length in [('feature_registration_sha256',64),('feature_manifest_sha256',64),('feature_code_commit',40)]:
        value=policy['source'][key]
        require(isinstance(value,str) and re.fullmatch('[0-9a-f]{'+str(length)+'}',value) is not None,'Invalid source pin')
    return policy


def read_experiment(path: Path) -> dict:
    """Read YAML with duplicate-key rejection; validate with the pinned registration."""
    with Path(path).open(encoding='utf-8-sig') as stream:
        policy=yaml.load(stream,Loader=_UniqueSafeLoader)
    _keys(policy,SCHEMA,'model_experiment')
    return policy

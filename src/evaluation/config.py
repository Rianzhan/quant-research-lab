"""Preregistered evaluation policy; no fitted-model or trading settings."""
from pathlib import Path
import re
import yaml
from src.data.research_policy import require, relative_path
from src.utils.config import _UniqueSafeLoader, _keys

SCHEMA={'schema_version':None,'version':None,'source':dict.fromkeys(['model_experiment','model_experiment_sha256',
        'model_manifest','model_manifest_sha256','model_code_commit']),'development':dict.fromkeys(['start','end']),
        **dict.fromkeys(['classification_threshold','positive_rule','majority_tie_class','calibration_edges',
        'classification_baselines','regression_baselines','levels','pairing','undefined_metrics','uncertainty',
        'numerical_tie_tolerance','retraining','model_selection','threshold_search','trading_backtest','holdout_access'])}


def validate_policy(policy: dict) -> dict:
    """Fail closed on any unsupported or outcome-driven policy change."""
    _keys(policy,SCHEMA,'evaluation')
    fixed={'schema_version':1,'version':'phase55-v1','development':{'start':'2018-01-01','end':'2025-12-31'},
        'classification_threshold':.5,'positive_rule':'greater_or_equal','majority_tie_class':1,
        'calibration_edges':[i/10 for i in range(11)],
        'classification_baselines':['majority_class','always_up','training_prior'],
        'regression_baselines':['zero_return','training_mean'],'levels':['all','fold','asset','fold_asset'],
        'pairing':'exact_keys_or_unavailable','undefined_metrics':'missing_with_reason',
        'uncertainty':'descriptive_only_no_independence_tests','numerical_tie_tolerance':1e-12}
    require(all(policy[k]==v for k,v in fixed.items()),'Unregistered evaluation policy')
    require(all(policy[k] is False for k in ['retraining','model_selection','threshold_search','trading_backtest','holdout_access']),
            'Training/selection/trading/holdout forbidden')
    for key in ['model_experiment','model_manifest']: relative_path(policy['source'][key])
    for key,length in [('model_experiment_sha256',64),('model_manifest_sha256',64),('model_code_commit',40)]:
        require(isinstance(policy['source'][key],str) and re.fullmatch('[0-9a-f]{'+str(length)+'}',policy['source'][key]) is not None,
                'Invalid frozen evaluation source pin')
    return policy


def read_policy(path: Path) -> dict:
    with Path(path).open(encoding='utf-8-sig') as stream:
        return validate_policy(yaml.load(stream,Loader=_UniqueSafeLoader))

"""Pre-registered validation settings, separate from the frozen data configuration."""
from datetime import date, datetime
import math
from pathlib import Path
import re
import yaml
from src.data.research_policy import require, relative_path
from src.features.builder import feature_specs
from src.utils.config import _UniqueSafeLoader, _keys, validate_config

SCHEMA={
    'schema_version':None,'version':None,'registration_status':None,'registered_at':None,'purpose':None,
    'source':dict.fromkeys(['feature_manifest','feature_manifest_sha256','feature_code_commit']),
    'development':dict.fromkeys(['start','end']),
    'split':dict.fromkeys(['method','train_start','test_windows','purge_rule','embargo_days','shuffle']),
    'eligibility':dict.fromkeys(['min_train_samples_per_asset','min_test_samples_per_asset','paired_asset_universe','sample_rule']),
    'feature_registration':dict.fromkeys(['preserve_phase3_raw_features','duplicates_removed','shared_raw_whitelist',
        'per_asset_main','pooled_status','pooled_main_whitelist','planned_replacements']),
    'audit':dict.fromkeys(['quantiles','regression_std_ddof','test_statistics_use']),
}


def iso_date(value: str) -> str:
    """Require an unambiguous calendar date, never a partial timestamp."""
    require(isinstance(value,str) and date.fromisoformat(value).isoformat()==value,'Expected ISO date')
    return value


def validate_validation(policy: dict, config: dict) -> dict:
    """Reject unknown settings, future windows and unregistered predictor columns."""
    validate_config(config);_keys(policy,SCHEMA,'validation_policy')
    require(policy['schema_version']==1 and isinstance(policy['version'],str) and bool(policy['version']),'Invalid validation version')
    require(policy['registration_status']=='frozen','Validation registration is not frozen')
    require(datetime.fromisoformat(policy['registered_at']).tzinfo is not None,'Registration time needs timezone')
    require(isinstance(policy['purpose'],str) and bool(policy['purpose']),'Missing purpose')
    source=policy['source'];relative_path(source['feature_manifest'])
    for key,length in [('feature_manifest_sha256',64),('feature_code_commit',40)]:
        require(isinstance(source[key],str) and re.fullmatch('[0-9a-f]{'+str(length)+'}',source[key]) is not None,'Invalid source identity')
    development=policy['development']
    require(development=={'start':'2018-01-01','end':'2025-12-31'},'Only frozen development window is allowed')
    require(development=={'start':config['data']['start_date'],'end':config['data']['end_date']},'Development config mismatch')
    split=policy['split']
    require(split['method']=='expanding_walk_forward' and split['shuffle'] is False,'Only chronological expanding validation')
    require(split['purge_rule']=='label_end_date_gte_test_start' and type(split['embargo_days']) is int
            and split['embargo_days']==0,'Endpoint purge required; embargo is not implemented')
    require(iso_date(split['train_start'])==development['start'],'Training must retain expanding development history')
    windows=split['test_windows'];require(isinstance(windows,list) and bool(windows),'Missing test windows')
    previous=split['train_start'];ids=set()
    for window in windows:
        _keys(window,dict.fromkeys(['id','start','end']),'test_window')
        require(isinstance(window['id'],str) and window['id'] and window['id'] not in ids,'Invalid fold ID')
        ids.add(window['id'])
        start,end=iso_date(window['start']),iso_date(window['end'])
        require(previous<start<=end<=development['end'],'Overlapping or out-of-development test windows')
        previous=end
    eligibility=policy['eligibility']
    for key in ['min_train_samples_per_asset','min_test_samples_per_asset']:
        require(type(eligibility[key]) is int and eligibility[key]>0,'Sample minima must be positive integers')
    require(eligibility['paired_asset_universe'] is True and eligibility['sample_rule']=='shared_raw_whitelist_complete_and_label_valid',
            'Both scopes must share sample/asset eligibility')
    feature=policy['feature_registration'];raw=[s['name'] for s in feature_specs(config)]
    removed=feature['duplicates_removed'];shared=feature['shared_raw_whitelist']
    require(isinstance(removed,list) and len(removed)==2 and len(set(removed))==2,'Remove exactly two duplicate columns')
    require(sum(x in removed for x in ['return_5d','momentum_5d'])==1
            and sum(x in removed for x in ['return_20d','momentum_20d'])==1,'Keep one column from each exact duplicate pair')
    require(shared==[name for name in raw if name not in removed] and len(shared)==17,'Invalid shared predictor whitelist')
    require(feature['preserve_phase3_raw_features'] is True and feature['per_asset_main']=='shared_raw_whitelist'
            and feature['pooled_status']=='preregistered_not_materialized','Feature registration scope changed')
    replacements=feature['planned_replacements']
    require(set(replacements)=={'ma_5d','ma_20d','ma_60d','ma_diff_5_20','ma_diff_20_60','macd_line','macd_signal','macd_hist'},
            'Expected eight scale-free replacements')
    for item in replacements.values():
        _keys(item,dict.fromkeys(['name','numerator','denominator','subtract']),'planned_replacement')
        require(isinstance(item['name'],str) and re.fullmatch('[a-z][a-z0-9_]*',item['name']) is not None,'Invalid planned name')
        require(item['numerator'] in raw+['research_close'] and item['denominator'] in ['ma_5d','ma_20d','ma_60d','research_close'],
                'Unregistered future/unknown feature dependency')
        require(type(item['subtract']) is int and item['subtract'] in (0,1),'Invalid ratio offset')
    pooled=[replacements[name]['name'] if name in replacements else name for name in shared]
    require(feature['pooled_main_whitelist']==pooled and len(set(pooled))==17 and not set(pooled)&{'date','asset_code','label_end_date','classification_target','regression_target'},
            'Invalid pooled whitelist')
    audit=policy['audit'];q=audit['quantiles']
    require(isinstance(q,list) and bool(q) and all(type(v) in (int,float) and math.isfinite(v) and 0<=v<=1 for v in q)
            and q==sorted(set(q)),'Invalid audit quantiles')
    require(len({f'q{round(v*100):02d}' for v in q})==len(q),'Audit quantile column names collide')
    require(type(audit['regression_std_ddof']) is int and audit['regression_std_ddof']==1
            and audit['test_statistics_use']=='quality_and_distribution_audit_only','Audit cannot tune models')
    return policy


def load_validation(path: Path, config: dict) -> dict:
    """Read safe, duplicate-key-rejecting YAML and validate the registered contract."""
    with Path(path).open(encoding='utf-8-sig') as stream:
        policy=yaml.load(stream,Loader=_UniqueSafeLoader)
    return validate_validation(policy,config)

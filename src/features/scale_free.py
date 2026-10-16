"""Pointwise scale-free transforms of approved, causally segmented Phase3 inputs."""
from pathlib import Path
import re
import numpy as np
import pandas as pd
import yaml
from src.data.cleaner import date_strings
from src.data.research_policy import require, relative_path
from src.utils.config import _keys, _UniqueSafeLoader
from src.utils.research_rows import ASSET, KEYS, validate_calendar
from src.validation.config import validate_validation

SCHEMA={
    'schema_version':None,'version':None,'development':dict.fromkeys(['start','end']),
    'source':dict.fromkeys(['validation_policy','validation_policy_sha256','validation_manifest',
                          'validation_manifest_sha256','validation_code_commit']),
    'model_scopes':None,'sample_contract':None,'shared_whitelist':None,
}


def validate_registration(registration: dict, validation: dict, config: dict) -> dict:
    """Require one shared whitelist and the exact existing formula registration."""
    validate_validation(validation,config);_keys(registration,SCHEMA,'scale_free_registration')
    require(registration['schema_version']==1 and registration['version']=='phase45-v1','Unsupported scale-free version')
    require(registration['development']==validation['development'],'Only frozen development dates are allowed')
    require(registration['model_scopes']==['per_asset','pooled']
            and registration['sample_contract']=='unchanged_phase4_membership','Both model scopes must share the sample contract')
    require(registration['shared_whitelist']==validation['feature_registration']['pooled_main_whitelist'],
            'Shared whitelist must equal the 17 preregistered scale-free predictors')
    source=registration['source']
    for key in ['validation_policy','validation_manifest']: relative_path(source[key])
    for key,length in [('validation_policy_sha256',64),('validation_manifest_sha256',64),('validation_code_commit',40)]:
        require(isinstance(source[key],str) and re.fullmatch('[0-9a-f]{'+str(length)+'}',source[key]) is not None,'Invalid frozen source identity')
    return registration


def read_registration(path: Path) -> dict:
    """Read YAML without duplicate keys; validation happens with the frozen source."""
    with Path(path).open(encoding='utf-8-sig') as stream:
        value=yaml.load(stream,Loader=_UniqueSafeLoader)
    _keys(value,SCHEMA,'scale_free_registration')
    return value


def build_scale_free(frame: pd.DataFrame, calendar: list[str], config: dict,
                     validation: dict, registration: dict) -> pd.DataFrame:
    """Transform same-row dependencies, preserving all keys and dependency NaNs.

    Production callers must use the pinned-input adapter. Rolling/recursive
    warmups live in the accepted Phase3 columns; no rolling state, fill or
    target is used here. Synthetic tests rebuild those columns before perturbing.
    """
    validate_registration(registration,validation,config);calendar=validate_calendar(calendar,config)
    spec=validation['feature_registration'];raw=spec['shared_raw_whitelist']
    require(not frame.empty and frame.columns.is_unique and set(KEYS+raw+['research_close'])<=set(frame),
            'Missing scale-free dependency rows/columns')
    rows=frame[KEYS+raw+['research_close']].copy(deep=True)
    require(rows[ASSET].notna().all().all(),'Missing asset identity')
    rows['date']=date_strings(rows.date)
    require(rows.date.isin(calendar).all(),'Holdout/out-of-development observations are forbidden')
    require(not rows.duplicated(['date',*ASSET]).any(),'Duplicate asset date')
    for name in [*raw,'research_close','price_segment']:
        rows[name]=pd.to_numeric(rows[name],errors='raise')
    require(not np.isinf(rows[[*raw,'research_close']].to_numpy(dtype=float,na_value=np.nan)).any(),'Nonfinite feature dependency')
    require(np.isfinite(rows.price_segment).all() and rows.price_segment.ge(0).all()
            and rows.price_segment.eq(np.floor(rows.price_segment)).all(),'Invalid segment identity')
    rows=rows.sort_values(KEYS,kind='stable').reset_index(drop=True)
    assets={(a['exchange'],a['code']):a for a in config['universe']['assets']}
    for identity,group in rows.groupby(ASSET,sort=False):
        require(identity in assets,'Unknown research asset')
        require(group.date.ge(assets[identity]['data_available_from']).all(),'Pre-observation values')
        require(group.price_segment.is_monotonic_increasing,'Corporate-action segments must not reverse or recur')
    result=rows[KEYS].copy()
    replacements=spec['planned_replacements']
    for old in raw:
        if old not in replacements:
            result[old]=rows[old]
            continue
        item=replacements[old];numerator=rows[item['numerator']];denominator=rows[item['denominator']]
        if item['numerator'] in ['research_close','ma_5d','ma_20d','ma_60d']:
            numerator=numerator.where(numerator.gt(0))
        denominator=denominator.where(denominator.gt(0))
        with np.errstate(over='ignore',divide='ignore',invalid='ignore'):
            result[item['name']]=numerator/denominator-item['subtract']
    names=registration['shared_whitelist']
    require(not np.isinf(result[names].to_numpy(dtype=float,na_value=np.nan)).any(),'Scale-free ratio overflow')
    return result[KEYS+names]

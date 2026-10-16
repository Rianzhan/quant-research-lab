"""Single preregistered final fit on frozen development rows; no test input."""
from hashlib import sha256
import pickle
import numpy as np
import pandas as pd
from src.backtest.contracts import require, stable_bytes
from src.models.factory import create_model
from src.utils.research_rows import KEYS

FEATURES = ('return_1d','return_5d','return_20d','momentum_60d','close_over_ma_5d',
    'close_over_ma_20d','close_over_ma_60d','ma_ratio_5_20','ma_ratio_20_60',
    'volatility_5d','volatility_20d','volume_change_1d','volume_ratio_20d','rsi_14',
    'macd_line_over_close','macd_signal_over_close','macd_hist_over_close')


def csv_bytes(frame: pd.DataFrame) -> bytes:
    """The accepted stable CSV serialization, including full float precision."""
    return frame.to_csv(index=False,float_format='%.17g',lineterminator='\n',na_rep='').encode()


def validate_contract(config: dict, spec: dict) -> None:
    """Pin parameters and order independently of caller-supplied declarations."""
    p=spec['training_protocol']
    require(tuple(spec['feature_whitelist'])==FEATURES,'Changed ordered features')
    require(config['experiment']['seed']==p['random_seed']==42,'Changed seed')
    require(config['models']['n_jobs']==p['n_jobs']==1,'Changed worker count')
    require(config['models']['logistic']==p['logistic_parameters']==
        {'C':1.,'max_iter':1000,'solver':'lbfgs'},'Changed A parameters')
    require(config['models']['random_forest']==p['random_forest_parameters']==
        {'n_estimators':200,'max_depth':5,'min_samples_leaf':10},'Changed B parameters')
    require(p['start']=='2018-01-01' and p['end']==p['label_end_date_max']=='2025-12-31'
        and p['min_train_samples_per_asset']==252,'Changed final training window/eligibility')
    require(spec['families']['A']=={'scope':'per_asset','model':'logistic','task':'classification','eligibility':'prediction > 0.55'}
        and spec['families']['B']=={'scope':'pooled','model':'random_forest','task':'regression','eligibility':'prediction > 0.001'},
        'Changed model families')


def cohort(features: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Use true label end dates, never positional tail deletion or test counts."""
    from src.final_holdout.inputs import validate_development
    require(list(features)==KEYS+list(FEATURES),'Changed feature columns/order')
    required=KEYS+['label_valid','label_end_date','classification_target','regression_target']
    require(set(required)<=set(labels),'Missing label contract')
    for frame in (features,labels):
        validate_development(frame)
        require(not frame.duplicated(KEYS).any() and frame[KEYS].notna().all().all(),'Invalid training keys')
    x=features.sort_values(KEYS,kind='stable').reset_index(drop=True)
    y=labels.sort_values(KEYS,kind='stable').reset_index(drop=True)
    require(x[KEYS].equals(y[KEYS]),'Feature/label key mismatch')
    require(y.label_valid.map(lambda v:isinstance(v,(bool,np.bool_))).all(),'Invalid label_valid')
    values=x[list(FEATURES)].to_numpy(dtype=float,na_value=np.nan)
    require(not np.isinf(values).any(),'Infinite training feature')
    valid=y.label_valid
    require(y.loc[valid,'label_end_date'].notna().all(),'Valid label missing end date')
    require((y.loc[valid,'label_end_date']>y.loc[valid,'date']).all(),'Invalid label interval')
    require(np.isfinite(y.loc[valid,['classification_target','regression_target']].to_numpy(dtype=float)).all(),
        'Nonfinite valid target')
    require(y.loc[valid,'classification_target'].eq((y.loc[valid,'regression_target']>0).astype(int)).all(),
        'Classification/regression disagreement')
    for col in required[len(KEYS):]:x[col]=y[col]
    return x.loc[np.isfinite(values).all(axis=1)&valid].reset_index(drop=True)


def fit_final(features: pd.DataFrame, labels: pd.DataFrame, config: dict, spec: dict) -> dict:
    """Fit each eligible A and the one B; deterministic repeats are audit-only."""
    validate_contract(config,spec);rows=cohort(features,labels)
    registry={(a['exchange'],a['code']) for a in config['universe']['assets']}
    require(set(map(tuple,rows[['exchange','asset_code']].drop_duplicates().to_numpy()))<=registry,'Unknown training asset')
    models={};files={};audit=[];eligible=[]
    def fit_one(identity: str, data: pd.DataFrame, family: str, task: str, members: list) -> None:
        data=data.sort_values(KEYS,kind='stable').reset_index(drop=True)
        model=create_model(family,task,config,list(FEATURES))
        model.fit(data[list(FEATURES)],data[task+'_target'])
        binary=pickle.dumps(model,protocol=5);path='models/'+identity+'.pkl'
        files[path]=binary;models[identity]=model
        keys=csv_bytes(data[KEYS]);files['cohorts/'+identity+'.csv']=keys
        scaler_hash=None
        if family=='logistic':
            scaler=pickle.dumps(model.estimator.named_steps['scaler'],protocol=5)
            files['scalers/'+identity+'.pkl']=scaler;scaler_hash=sha256(scaler).hexdigest()
        audit.append({'model_id':identity,'status':'fitted','training_rows':len(data),
            'first_training_date':data.date.min(),'last_training_date':data.date.max(),
            'max_label_end_date':data.label_end_date.max(),'membership':members,
            'class_counts':{str(int(k)):int(v) for k,v in data.classification_target.value_counts().sort_index().items()} if task=='classification' else None,
            'feature_names':list(FEATURES),'seed':42,'n_jobs':1,**model.metadata(),
            'training_row_key_sha256':sha256(keys).hexdigest(),'training_values_sha256':sha256(csv_bytes(data)).hexdigest(),
            'model_binary_sha256':sha256(binary).hexdigest(),'scaler_sha256':scaler_hash})
    for exchange,code in sorted(registry):
        group=rows.loc[rows.exchange.eq(exchange)&rows.asset_code.eq(code)]
        identity='A_'+exchange+'_'+code
        if len(group)<252:
            audit.append({'model_id':identity,'status':'insufficient_training_history','training_rows':len(group)})
            continue
        eligible.append(group)
        require(set(group.classification_target.unique())=={0,1},'A lacks both classes; final fit must stop')
        fit_one(identity,group,'logistic','classification',[[exchange,code]])
    require(bool(eligible),'No eligible pooled training assets')
    pooled=pd.concat(eligible,ignore_index=True)
    members=pooled[['exchange','asset_code']].drop_duplicates().sort_values(['exchange','asset_code']).values.tolist()
    fit_one('B_pooled',pooled,'random_forest','regression',members)
    files['training_details.json']=stable_bytes(audit)
    return {'models':models,'files':files,'audit':audit}

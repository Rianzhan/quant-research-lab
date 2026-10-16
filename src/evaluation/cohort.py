"""Bind predictions to exact frozen test keys and compute strictly training-only priors."""
import re
import numpy as np
import pandas as pd
from src.data.loader import digest
from src.data.research_policy import require
from src.models.experiment import PREDICTION_COLUMNS,audit_membership
from src.utils.research_rows import KEYS
from src.validation.pipeline import _csv
from src.validation.walk_forward import fold_windows,select_samples

EVAL_KEYS=['fold_id',*KEYS,'task']
RUN_KEYS=['fold_id','scope','exchange','asset_code','model','task']


def key_hash(rows: pd.DataFrame) -> str:
    """Canonical semantic keys; never rely on CSV physical row order."""
    return digest(_csv(rows[EVAL_KEYS].sort_values(EVAL_KEYS,kind='stable').reset_index(drop=True)))


def _train_parameters(train: pd.DataFrame, identity: dict) -> dict:
    prior=float(train.classification_target.mean()) if len(train) else np.nan
    return {**identity,'status':'available' if len(train) else 'unavailable','train_n':len(train),
            'train_start':train.date.min() if len(train) else '', 'train_end':train.date.max() if len(train) else '',
            'train_label_end_max':train.label_end_date.max() if len(train) else '',
            'training_prior':prior,'majority_class':int(prior>=.5) if len(train) else np.nan,
            'training_mean':float(train.regression_target.mean()) if len(train) else np.nan,
            'train_keys_sha256':digest(_csv(train[KEYS]))}


def _validate_fit(row: pd.Series, train: pd.DataFrame, test: pd.DataFrame, names: list[str], window: dict) -> str:
    """Reconcile a saved fit record without fitting or making a new prediction."""
    target='classification_target' if row.task=='classification' else 'regression_target'
    status='unavailable_no_training' if train.empty else (
        'unavailable_single_class' if row.task=='classification' and train[target].nunique()!=2 else (
            'trained' if len(test) else 'trained_no_test'))
    expected={'status':status,'train_samples':len(train),'test_samples':len(test),
        'train_start':train.date.min() if len(train) else '', 'train_end':train.date.max() if len(train) else '',
        'train_label_end_max':train.label_end_date.max() if len(train) else '',
        'train_keys_sha256':digest(_csv(train[KEYS])),'test_keys_sha256':digest(_csv(test[KEYS])),
        'train_X_sha256':digest(_csv(train[names])) if len(train) else '',
        'train_y_sha256':digest(_csv(train[[target]])) if len(train) else '',
        'test_start_date':window['test_start_date'],'test_end_date':window['test_end_date']}
    require(all(row[k]==v for k,v in expected.items()),'Frozen fit record does not match training/test contract')
    if status.startswith('trained'):
        asset=row.exchange+'_'+row.asset_code if row.scope=='per_asset' else 'ALL'
        path=f'models/{row.fold_id}/{row.scope}/{asset}/{row.model}_{row.task}.pkl'
        require(row.model_file==path and re.fullmatch('[0-9a-f]{64}',row.model_sha256) is not None,'Invalid fitted model identity')
    else:
        require(row.model_file==row.model_sha256=='','Unavailable run cannot contain a model')
    return status


def prepare_cohort(frame: pd.DataFrame, members: pd.DataFrame, calendar: list[str], validation: dict,
                   registration: dict, config: dict, experiment: dict, predictions: pd.DataFrame,
                   runs: pd.DataFrame) -> tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    """Return exact labelled predictions, baseline parameters and full asset availability."""
    audit_membership(frame,members,calendar,validation,registration,config,experiment)
    require(list(predictions)==PREDICTION_COLUMNS,'Unexpected saved prediction schema')
    require(predictions[KEYS+['fold_id','scope','model','task']].notna().all().all(),'Missing prediction identity')
    require(not predictions.duplicated(EVAL_KEYS+['scope','model']).any(),'Duplicate prediction key')
    require(np.isfinite(pd.to_numeric(predictions.prediction,errors='raise')).all(),'Nonfinite prediction')
    prob=predictions.loc[predictions.task=='classification','prediction']
    require(prob.between(0,1).all(),'Probability outside [0,1]')
    require(not runs.duplicated(RUN_KEYS).any(),'Duplicate frozen fit record')
    windows=fold_windows(calendar,validation);folds=[w['fold_id'] for w in windows]
    assets=[(a['exchange'],a['code']) for a in config['universe']['assets']]
    joined=[];parameters=[];availability=[];seen_runs=[];seen_prediction_rows=0
    for window in windows:
        fold=window['fold_id'];train=select_samples(frame,members,fold,'train',registered_folds=folds)
        test=select_samples(frame,members,fold,'test',registered_folds=folds)
        require(train.date.lt(window['test_start_date']).all() and train.label_end_date.lt(window['test_start_date']).all(),
                'Baseline training crosses test start')
        for scope in experiment['scopes']:
            for exchange,code in assets if scope=='per_asset' else [('ALL','ALL')]:
                tr=train if scope=='pooled' else train.loc[(train.exchange==exchange)&(train.asset_code==code)].reset_index(drop=True)
                te=test if scope=='pooled' else test.loc[(test.exchange==exchange)&(test.asset_code==code)].reset_index(drop=True)
                identity={'fold_id':fold,'scope':scope,'exchange':exchange,'asset_code':code}
                parameter=_train_parameters(tr,identity);parameters.append(parameter)
                for task in experiment['model_tasks']:
                    run_identity={**identity,**task};seen_runs.append(run_identity)
                    mask=pd.Series(True,index=runs.index)
                    for k,value in run_identity.items(): mask&=runs[k].eq(value)
                    require(int(mask.sum())==1,'Missing/extra registered fit record')
                    status=_validate_fit(runs.loc[mask].iloc[0],tr,te,registration['shared_whitelist'],window)
                    subset=predictions.loc[(predictions.fold_id==fold)&(predictions.scope==scope)&
                        (predictions.model==task['model'])&(predictions.task==task['task'])]
                    if scope=='per_asset': subset=subset.loc[(subset.exchange==exchange)&(subset.asset_code==code)]
                    expected=te if status=='trained' else te.iloc[:0]
                    actual_keys=subset[KEYS].sort_values(KEYS).reset_index(drop=True)
                    require(actual_keys.equals(expected[KEYS].reset_index(drop=True)),
                            'Frozen prediction keys differ from expected test membership')
                    seen_prediction_rows+=len(subset)
                    if len(subset):
                        target='classification_target' if task['task']=='classification' else 'regression_target'
                        labelled=subset.merge(te[KEYS+[target]],on=KEYS,validate='one_to_one').rename(columns={target:'actual'})
                        for name in ['majority_class','training_prior','training_mean']: labelled[name]=parameter[name]
                        joined.append(labelled)
                    for a_exchange,a_code in assets if scope=='pooled' else [(exchange,code)]:
                        n=int(((subset.exchange==a_exchange)&(subset.asset_code==a_code)).sum())
                        candidate=members.loc[(members.fold_id==fold)&(members.exchange==a_exchange)&(members.asset_code==a_code)]
                        reason=';'.join(sorted(set(candidate.exclusion_reason)-{''})) if len(candidate) else 'no_observations'
                        availability.append({'fold_id':fold,'scope':scope,'exchange':a_exchange,'asset_code':a_code,**task,
                            'n':n,'status':'available' if n else 'unavailable','reason':'' if n else (reason or status)})
    require(len(runs)==len(seen_runs) and seen_prediction_rows==len(predictions),'Unregistered run/prediction identities')
    columns=[*PREDICTION_COLUMNS,'actual','majority_class','training_prior','training_mean']
    result=pd.concat(joined,ignore_index=True) if joined else pd.DataFrame(columns=columns)
    result=result.sort_values(EVAL_KEYS+['scope','model'],kind='stable').reset_index(drop=True)
    return (result,pd.DataFrame(parameters).sort_values(['fold_id','scope','exchange','asset_code']).reset_index(drop=True),
            pd.DataFrame(availability).sort_values(RUN_KEYS).reset_index(drop=True))

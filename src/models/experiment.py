"""Frozen-fold research fitting: no test labels, tuning, signals or performance metrics."""
import logging
import pickle
import re
import pandas as pd
from src.data.loader import digest, json_bytes
from src.data.research_policy import require
from src.features.scale_free import validate_registration
from src.models.config import validate_experiment
from src.models.factory import create_model
from src.utils.research_rows import KEYS, validate_calendar
from src.validation.pipeline import _csv
from src.validation.walk_forward import MEMBERSHIP_COLUMNS, build_membership, fold_windows, select_samples

PREDICTION_COLUMNS=[*KEYS,'fold_id','scope','model','task','prediction']


def audit_membership(frame: pd.DataFrame, membership: pd.DataFrame, calendar: list[str],
                     validation: dict, registration: dict, config: dict, experiment: dict) -> None:
    """Recheck complete-row eligibility and endpoint purge, without computing features.

    The positional rename reuses Phase4's validation of presence, targets and dates:
    it never evaluates a raw-feature formula or exposes raw scales to the models.
    Phase4.5 already proved these 17 columns have the same completeness mask.
    """
    validate_registration(registration,validation,config);validate_experiment(experiment,config,registration)
    validate_calendar(calendar,config)
    names=registration['shared_whitelist'];raw=validation['feature_registration']['shared_raw_whitelist']
    require(frame.columns.is_unique and set(KEYS+names)<=set(frame),'Missing model inputs')
    require(list(membership)==MEMBERSHIP_COLUMNS,'Unexpected membership schema')
    for fold in validation['split']['test_windows']:
        require(re.fullmatch('[A-Za-z0-9_-]+',fold['id']) is not None,'Unsafe fold ID')
    neutral=frame.rename(columns=dict(zip(names,raw)))
    expected=build_membership(neutral,calendar,config,validation)
    actual=membership.sort_values(['fold_id',*KEYS],kind='stable').reset_index(drop=True)
    require(_csv(actual)==_csv(expected),'Frozen fold membership/endpoints/eligibility mismatch')


def _fit_one(train: pd.DataFrame, test: pd.DataFrame, identity: dict, names: list[str],
             config: dict) -> tuple[dict,dict,pd.DataFrame,bytes | None]:
    """Fit only supplied historical rows; predictions consume no target column."""
    row={**identity,'train_samples':len(train),'test_samples':len(test),
         'train_start':train.date.min() if len(train) else '',
         'train_end':train.date.max() if len(train) else '',
         'train_label_end_max':train.label_end_date.max() if len(train) else '',
         'train_keys_sha256':digest(_csv(train[KEYS])),
         'test_keys_sha256':digest(_csv(test[KEYS])),'model_file':'','model_sha256':''}
    row['status']='unavailable_no_training';row['reason']='No retained training rows under frozen eligibility'
    detail={};pred=pd.DataFrame(columns=PREDICTION_COLUMNS);payload=None
    if train.empty: return row,detail,pred,payload
    target='classification_target' if identity['task']=='classification' else 'regression_target'
    row['train_X_sha256']=digest(_csv(train[names]));row['train_y_sha256']=digest(_csv(train[[target]]))
    if identity['task']=='classification' and train[target].nunique()!=2:
        row.update(status='unavailable_single_class',reason='Training labels contain only one class; all classifiers unavailable')
        return row,detail,pred,payload
    model=create_model(identity['model'],identity['task'],config,names)
    model.fit(train[names],train[target])
    values=model.predict(test[names])
    pred=test[KEYS].copy()
    for key in ['fold_id','scope','model','task']: pred[key]=identity[key]
    pred['prediction']=values;pred=pred[PREDICTION_COLUMNS]
    payload=pickle.dumps(model,protocol=5)
    asset=identity['exchange']+'_'+identity['asset_code'] if identity['scope']=='per_asset' else 'ALL'
    filename=f"models/{identity['fold_id']}/{identity['scope']}/{asset}/{identity['model']}_{identity['task']}.pkl"
    row.update(status='trained' if len(test) else 'trained_no_test',reason='' if len(test) else 'No eligible test rows',
               model_file=filename,model_sha256=digest(payload))
    detail=model.metadata()
    return row,detail,pred,payload


def run_experiment(frame: pd.DataFrame, membership: pd.DataFrame, calendar: list[str], validation: dict,
                   registration: dict, config: dict, experiment: dict) -> dict[str,bytes]:
    """Generate deterministic per-asset and pooled continuous OOS predictions.

    Production callers must use the pinned input adapter. Test-label statistics
    are deliberately absent; Phase8 will join targets for registered evaluation.
    """
    audit_membership(frame,membership,calendar,validation,registration,config,experiment)
    names=registration['shared_whitelist'];windows=fold_windows(calendar,validation)
    fold_ids=[w['fold_id'] for w in windows];rows=[];details=[];predictions=[];artifacts={}
    identities=[(a['exchange'],a['code']) for a in config['universe']['assets']]
    for window in windows:
        train=select_samples(frame,membership,window['fold_id'],'train',registered_folds=fold_ids)
        test=select_samples(frame,membership,window['fold_id'],'test',registered_folds=fold_ids)
        require(train.date.lt(window['test_start_date']).all() and train.label_end_date.lt(window['test_start_date']).all(),
                'Training crosses the global test boundary')
        for scope in experiment['scopes']:
            for exchange,code in identities if scope=='per_asset' else [('ALL','ALL')]:
                tr=train if scope=='pooled' else train.loc[(train.exchange==exchange)&(train.asset_code==code)].reset_index(drop=True)
                te=test if scope=='pooled' else test.loc[(test.exchange==exchange)&(test.asset_code==code)].reset_index(drop=True)
                for task in experiment['model_tasks']:
                    identity={'fold_id':window['fold_id'],'scope':scope,'exchange':exchange,'asset_code':code,**task}
                    row,detail,pred,payload=_fit_one(tr,te,identity,names,config)
                    row.update(test_start_date=window['test_start_date'],test_end_date=window['test_end_date'])
                    rows.append(row);details.append({**identity,**detail,'status':row['status']})
                    if len(pred): predictions.append(pred)
                    if payload is not None: artifacts[row['model_file']]=payload
                    logging.getLogger('quant_lab').info('%s %s %s %s/%s: %s (%d train, %d test)',
                        window['fold_id'],scope,code,task['model'],task['task'],row['status'],len(tr),len(te))
    predictions=pd.concat(predictions,ignore_index=True) if predictions else pd.DataFrame(columns=PREDICTION_COLUMNS)
    predictions=predictions.sort_values(['fold_id',*KEYS,'scope','model','task'],kind='stable').reset_index(drop=True)
    runs=pd.DataFrame(rows).fillna('')
    audit={'version':'phase5-v1','scope':'development_only','requested_runs':len(runs),
           'trained_models':len(artifacts),'status_counts':runs.status.value_counts().sort_index().to_dict(),
           'prediction_rows':len(predictions),'feature_count':len(names),'shared_whitelist':names,
           'fold_ids':fold_ids,'frozen_membership_unchanged':True,'model_training_performed':True,
           'model_selection_performed':False,'performance_metrics_computed':False,
           'backtest_total_return_ready':False,'holdout_files_opened':False}
    artifacts.update({'predictions.csv':_csv(predictions),'fit_runs.csv':_csv(runs),
                      'fit_details.json':json_bytes(details),'audit.json':json_bytes(audit),
                      'experiment_policy.json':json_bytes(experiment)})
    return artifacts

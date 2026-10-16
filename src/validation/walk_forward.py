"""Date-based expanding folds with endpoint purge, shared across model scopes."""
from datetime import date, timedelta
import numpy as np
import pandas as pd
from src.data.cleaner import date_strings
from src.data.research_policy import require
from src.utils.research_rows import ASSET, KEYS, validate_calendar
from src.validation.config import validate_validation

MEMBERSHIP_COLUMNS=['fold_id',*KEYS,'label_end_date','partition','included',
                    'training_eligible','evaluation_eligible','asset_status','exclusion_reason']


def prepare_samples(frame: pd.DataFrame, calendar: list[str], config: dict, policy: dict) -> pd.DataFrame:
    """Validate observed rows, then select complete registered features/valid labels.

    This operates on an audited label table; the input adapter verifies its exact
    five-session price definition. No target value influences split membership.
    """
    validate_validation(policy,config);calendar=validate_calendar(calendar,config)
    names=policy['feature_registration']['shared_raw_whitelist']
    required=set(KEYS+names+['label_valid','label_end_date','classification_target','regression_target'])
    require(not frame.empty and frame.columns.is_unique and required<=set(frame),'Missing validation input columns/rows')
    rows=frame.copy(deep=True)
    require(rows[ASSET].notna().all().all(),'Missing asset identity')
    expected={(a['exchange'],a['code']) for a in config['universe']['assets']}
    require(set(zip(rows.exchange,rows.asset_code))<=expected,'Unknown validation asset')
    rows['date']=date_strings(rows.date)
    require(rows.date.between(policy['development']['start'],policy['development']['end']).all()
            and rows.date.isin(calendar).all(),'Holdout/out-of-development quote dates')
    require(not rows.duplicated(['date',*ASSET]).any(),'Duplicate asset date')
    present=rows.label_end_date.notna()
    rows.loc[present,'label_end_date']=date_strings(rows.loc[present,'label_end_date'])
    require(rows.loc[present,'label_end_date'].between(policy['development']['start'],policy['development']['end']).all()
            and rows.loc[present,'label_end_date'].isin(calendar).all(),'Holdout/out-of-development label endpoints')
    require(rows.label_valid.map(lambda v:isinstance(v,(bool,np.bool_))).all(),'label_valid must be Boolean')
    for name in [*names,'regression_target','classification_target','price_segment']:
        rows[name]=pd.to_numeric(rows[name],errors='raise')
    require(not np.isinf(rows[names].to_numpy(dtype=float)).any(),'Nonfinite feature values')
    require(np.isfinite(rows.price_segment).all() and rows.price_segment.ge(0).all()
            and rows.price_segment.eq(np.floor(rows.price_segment)).all(),'Invalid segment identity')
    valid=rows.label_valid
    require(np.isfinite(rows.loc[valid,'regression_target']).all() and rows.loc[valid,'classification_target'].isin([0,1]).all(),
            'Missing/nonfinite valid label')
    require(rows.loc[~valid,['regression_target','classification_target']].isna().all().all(),'Invalid labels must remain missing')
    require(rows.loc[valid,'label_end_date'].notna().all()
            and rows.loc[valid,'label_end_date'].gt(rows.loc[valid,'date']).all(),'Label endpoint must be strictly after feature date')
    expected_class=(rows.loc[valid,'regression_target']>config['labels']['classification_return_threshold']).astype(int)
    require(rows.loc[valid,'classification_target'].eq(expected_class).all(),'Classification/regression label mismatch')
    complete=rows[names].notna().all(axis=1)
    return rows.loc[complete&valid].sort_values(KEYS,kind='stable').reset_index(drop=True)


def fold_windows(calendar: list[str], policy: dict) -> list[dict]:
    """Resolve one global session axis, not each asset's first observed test quote."""
    result=[]
    for window in policy['split']['test_windows']:
        days=[d for d in calendar if window['start']<=d<=window['end']]
        require(bool(days),'Test window has no development calendar sessions: '+window['id'])
        result.append({'fold_id':window['id'],'train_window_start':policy['split']['train_start'],
            'train_window_end':(date.fromisoformat(window['start'])-timedelta(days=1)).isoformat(),
            'test_window_start':window['start'],'test_window_end':window['end'],
            'test_start_date':days[0],'test_end_date':days[-1]})
    return result


def asset_availability(rows: pd.DataFrame, policy: dict) -> dict:
    """Past-only training eligibility; future counts govern evaluation only."""
    train=int(rows.partition.eq('train').sum());purged=int(rows.partition.eq('purged').sum())
    test=int(rows.partition.eq('test').sum());limits=policy['eligibility'];reasons=[]
    if train<limits['min_train_samples_per_asset']:
        reasons.append(f'train_samples_{train}_below_{limits["min_train_samples_per_asset"]}')
    if test<limits['min_test_samples_per_asset']:
        reasons.append(f'test_samples_{test}_below_{limits["min_test_samples_per_asset"]}')
    status='available' if not reasons else ('unavailable' if train+purged==0 or test==0 else 'insufficient')
    training_eligible=train>=limits['min_train_samples_per_asset']
    evaluation_eligible=training_eligible and test>=limits['min_test_samples_per_asset']
    return {'status':status,'reason':';'.join(reasons),
            'training_eligible':training_eligible,'evaluation_eligible':evaluation_eligible,'train_before_purge':train+purged,
            'train_after_purge':train,'purged_samples':purged,'test_samples':test}


def build_membership(frame: pd.DataFrame, calendar: list[str], config: dict, policy: dict) -> pd.DataFrame:
    """Preserve candidate membership and removal reasons without fitting anything."""
    rows=prepare_samples(frame,calendar,config,policy);pieces=[]
    for window in fold_windows(calendar,policy):
        start,end=window['test_start_date'],window['test_end_date']
        train=rows.date.ge(window['train_window_start'])&rows.date.lt(start)
        test=rows.date.between(start,end)
        candidate=rows.loc[train|test,KEYS+['label_end_date']].copy()
        candidate['partition']=np.where(candidate.date.ge(start),'test',
                                         np.where(candidate.label_end_date.ge(start),'purged','train'))
        candidate['fold_id']=window['fold_id'];candidate['included']=False
        candidate['asset_status']='';candidate['exclusion_reason']=''
        candidate['training_eligible']=False;candidate['evaluation_eligible']=False
        for _,group in candidate.groupby(ASSET,sort=False):
            state=asset_availability(group,policy)
            candidate.loc[group.index,'asset_status']=state['status']
            candidate.loc[group.index,'training_eligible']=state['training_eligible']
            candidate.loc[group.index,'evaluation_eligible']=state['evaluation_eligible']
            candidate.loc[group.index,'included']=(group.partition.eq('train')&state['training_eligible']) | (
                group.partition.eq('test')&state['evaluation_eligible'])
            candidate.loc[group.index,'exclusion_reason']=state['reason']
        purged=candidate.partition.eq('purged')
        candidate.loc[purged,'exclusion_reason']=candidate.loc[purged,'exclusion_reason'].map(
            lambda reason: (reason+';' if reason else '')+'label_end_date_gte_test_start')
        retained=candidate.loc[candidate.partition=='train']
        require(retained.date.lt(start).all() and retained.label_end_date.lt(start).all(),'Train/test leakage')
        pieces.append(candidate[MEMBERSHIP_COLUMNS])
    result=pd.concat(pieces,ignore_index=True).sort_values(['fold_id',*KEYS],kind='stable').reset_index(drop=True)
    result.attrs['fold_ids']=[window['id'] for window in policy['split']['test_windows']]
    return result


def select_samples(frame: pd.DataFrame, membership: pd.DataFrame, fold_id: str, partition: str,
                   asset: tuple[str,str] | None=None, *, registered_folds: list[str] | None=None) -> pd.DataFrame:
    """Common sample accessor for pooled or a participating single-asset model.

    Returns full rows for a future caller to select its registered predictor
    whitelist. Audit/target columns returned here must never be passed as X.
    After CSV reload, supply registered_folds to distinguish empty from unknown
    folds; in-memory membership carries this registry in DataFrame.attrs.
    """
    require(partition in ('train','test'),'Select train or test only')
    fold_ids=registered_folds if registered_folds is not None else membership.attrs.get(
        'fold_ids',membership.fold_id.unique().tolist())
    require(fold_id in fold_ids,'Unknown fold: '+fold_id)
    selected=membership.loc[(membership.fold_id==fold_id)&membership.included&(membership.partition==partition)]
    if asset is not None:
        selected=selected.loc[(selected.exchange==asset[0])&(selected.asset_code==asset[1])]
        require(not selected.empty,'Asset unavailable for this fold/partition')
    result=selected[KEYS].merge(frame,on=KEYS,how='left',validate='one_to_one',indicator=True)
    require(result._merge.eq('both').all(),'Membership does not align with observations')
    return result.drop(columns='_merge').sort_values(KEYS,kind='stable').reset_index(drop=True)

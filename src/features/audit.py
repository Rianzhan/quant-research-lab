"""Disjoint missingness attribution; not feature selection or strategy evaluation."""
import numpy as np
import pandas as pd
from src.data.research_policy import require
from src.features.builder import feature_specs
from src.utils.research_rows import ASSET, KEYS, prepare_rows

LOSS_KEYS=['initial_warmup','event_reset','calendar_gap_reset','volume_anomaly','other_missing']


def _dependency_missing(rows: pd.DataFrame, lookback: int) -> pd.Series:
    """Flag invalid-volume dependencies, without using future or another run."""
    invalid=pd.Series(False,index=rows.index)
    for _,group in rows.groupby([*ASSET,'price_segment','_run_id'],sort=False):
        invalid.loc[group.index]=group._effective_volume.isna().rolling(lookback+1,min_periods=1).max().eq(1)
    return invalid


def _loss_masks(rows: pd.DataFrame, values: pd.Series, lookback: int, volume_missing: pd.Series) -> dict[str,pd.Series]:
    """Initial warmup, action, gap, then volume: explicit non-overlapping precedence."""
    initial=rows._asset_position.lt(lookback)
    event=~initial & rows._segment_position.lt(lookback)
    gap=~initial & ~event & rows._run_position.lt(lookback)
    volume=~initial & ~event & ~gap & volume_missing
    expected=initial|event|gap|volume
    require(not (expected & values.notna()).any(),'Feature has values before complete dependency window')
    other=values.isna() & ~expected
    require(not other.any(),'Unexplained feature missingness')
    return dict(zip(LOSS_KEYS,[initial,event,gap,volume,other]))


def _counts(values: pd.Series, masks: dict[str,pd.Series], index: pd.Index) -> dict:
    result={'rows':len(index),'valid':int(values.loc[index].notna().sum())}
    result.update({key:int(mask.loc[index].sum()) for key,mask in masks.items()})
    require(result['rows']==result['valid']+sum(result[k] for k in LOSS_KEYS),'Loss counts do not reconcile')
    return result


def audit_samples(panel: pd.DataFrame, features: pd.DataFrame, labels: pd.DataFrame,
                  config: dict, calendar: list[str]) -> dict:
    """Count each predictor, event/gap warmup and valid targets by asset."""
    rows=prepare_rows(panel,config,calendar)
    require(features[KEYS].equals(rows[KEYS]) and labels[KEYS].equals(rows[KEYS]),'Feature/label alignment mismatch')
    specs=feature_specs(config); names=[s['name'] for s in specs]
    require(list(features.columns)==KEYS+names,'Predictor whitelist mismatch')
    feature_counts=[];by_asset=[];volume_any=pd.Series(False,index=rows.index)
    for spec in specs:
        dependency=_dependency_missing(rows,spec['lookback']) if spec['uses_volume'] else pd.Series(False,index=rows.index)
        volume_any|=dependency
        masks=_loss_masks(rows,features[spec['name']],spec['lookback'],dependency)
        feature_counts.append({**spec,**_counts(features[spec['name']],masks,rows.index),
                               'volume_dependency_affected':int(dependency.sum())})
        for identity,group in rows.groupby(ASSET,sort=False):
            by_asset.append({'exchange':identity[0],'asset_code':identity[1],'name':spec['name'],
                **_counts(features[spec['name']],masks,group.index),
                'volume_dependency_affected':int(dependency.loc[group.index].sum())})
    complete=features[names].notna().all(axis=1)
    pseudo=pd.Series(np.where(complete,1.,np.nan),index=rows.index)
    masks=_loss_masks(rows,pseudo,max(s['lookback'] for s in specs),volume_any)
    per_asset=[]
    for identity,group in rows.groupby(ASSET,sort=False):
        target=labels.loc[group.index]; valid=target.label_valid
        per_asset.append({'exchange':identity[0],'asset_code':identity[1],'asset_name':group.asset_name.iloc[0],
            'first_date':group.date.min(),'last_date':group.date.max(),
            **_counts(pseudo,masks,group.index),'valid_labels':int(valid.sum()),
            'supervised_eligible':int((complete.loc[group.index]&valid).sum()),
            'label_invalid_reasons':{str(k):int(v) for k,v in target.loc[~valid,'label_invalid_reason'].value_counts().sort_index().items()}})
    require(sum(item['rows'] for item in per_asset)==len(rows),'Per-asset row counts do not reconcile')
    return {'scope':'development_only','formula_version':'phase3-v1','rows':len(rows),'feature_count':len(names),
            'features':feature_counts,'features_by_asset':by_asset,'per_asset':per_asset,
            'all_features_complete':_counts(pseudo,masks,rows.index),
            'valid_labels':int(labels.label_valid.sum()),'supervised_eligible':int((complete&labels.label_valid).sum()),
            'invalid_volume_dates':rows.loc[rows._effective_volume.isna(),['date',*ASSET]].to_dict('records'),
            'backtest_total_return_ready':False,
            'notes':['Loss categories are mutually exclusive within a feature; do not sum across feature columns.',
                     'Volume dependency coverage may overlap warmup and is not an additional additive loss.',
                     '5/20-day momentum duplicates matching historical return columns; no feature selection performed.',
                     'Only original observed development rows are emitted; absent sessions reset feature state.',
                     'Future endpoint and supervised eligibility are never predictors or realtime signal filters.']}

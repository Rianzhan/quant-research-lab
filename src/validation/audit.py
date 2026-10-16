"""Fold distribution summaries for quality checks only, never model selection."""
import numpy as np
import pandas as pd
from src.data.research_policy import require
from src.utils.research_rows import ASSET
from src.validation.walk_forward import asset_availability, fold_windows, select_samples


def target_statistics(rows: pd.DataFrame, policy: dict) -> dict:
    """Report class proportions and price-target moments; undefined stays None."""
    n=len(rows);positive=int(rows.classification_target.eq(1).sum());negative=int(rows.classification_target.eq(0).sum())
    require(positive+negative==n and np.isfinite(rows.regression_target).all(),'Statistics require valid targets')
    values=rows.regression_target
    result={'samples':n,'positive_count':positive,'negative_count':negative,
        'positive_ratio':positive/n if n else None,'negative_ratio':negative/n if n else None,
        'both_classes_present':positive>0 and negative>0,
        'regression_mean':float(values.mean()) if n else None,
        'regression_median':float(values.median()) if n else None,
        'regression_std':float(values.std(ddof=policy['audit']['regression_std_ddof'])) if n>1 else None}
    result.update({f'q{round(q*100):02d}':float(values.quantile(q,interpolation='linear')) if n else None
                   for q in policy['audit']['quantiles']})
    return result


def _bounds(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    def bound(rows,column,method):
        return str(getattr(rows[column],method)()) if len(rows) else None
    return {'train_actual_start':bound(train,'date','min'),'train_actual_end':bound(train,'date','max'),
        'train_label_end_max':bound(train,'label_end_date','max'),
        'test_actual_start':bound(test,'date','min'),'test_actual_end':bound(test,'date','max'),
        'test_label_end_max':bound(test,'label_end_date','max')}


def audit_folds(frame: pd.DataFrame, membership: pd.DataFrame, calendar: list[str], config: dict, policy: dict) -> dict:
    """Audit causal training membership separately from evaluation eligibility."""
    folds=[];assets=[];distributions=[]
    for window in fold_windows(calendar,policy):
        fold_id=window['fold_id'];members=membership.loc[membership.fold_id==fold_id]
        participated=[];training_assets=[];excluded=[];asset_rows=[]
        for asset in config['universe']['assets']:
            identity=(asset['exchange'],asset['code'])
            group=members.loc[(members.exchange==identity[0])&(members.asset_code==identity[1])]
            state=asset_availability(group,policy)
            candidate=group.copy();candidate['included']=candidate.partition.ne('purged')
            train=select_samples(frame,candidate,fold_id,'train');test=select_samples(frame,candidate,fold_id,'test')
            row={**window,'exchange':identity[0],'asset_code':identity[1],'asset_name':asset['name'],**state,**_bounds(train,test)}
            assets.append(row);asset_rows.append(row)
            (participated if state['evaluation_eligible'] else excluded).append(identity[0]+':'+identity[1])
            if state['training_eligible']:
                training_assets.append(identity[0]+':'+identity[1])
            for name,subset in [('train',train),('test',test)]:
                distributions.append({'fold_id':fold_id,'scope':'per_asset','exchange':identity[0],'asset_code':identity[1],
                    'partition':name,'asset_status':state['status'],'use':'candidate_quality_audit_only',**target_statistics(subset,policy)})
        train=select_samples(frame,membership,fold_id,'train');test=select_samples(frame,membership,fold_id,'test')
        training=[row for row in asset_rows if row['training_eligible']]
        active=[row for row in asset_rows if row['evaluation_eligible']]
        counts={key:sum(row[key] for row in training) for key in ['train_before_purge','train_after_purge','purged_samples']}
        counts['test_samples']=sum(row['test_samples'] for row in active)
        require(counts['train_before_purge']==counts['train_after_purge']+counts['purged_samples'],'Purge counts do not reconcile')
        require(counts['train_after_purge']==len(train) and counts['test_samples']==len(test),'Scope sample counts differ')
        require(train.date.lt(window['test_start_date']).all() and train.label_end_date.lt(window['test_start_date']).all(),'Train/test endpoint leakage')
        folds.append({**window,'status':'available' if active else 'unavailable',**counts,**_bounds(train,test),
            **{'candidate_'+key:sum(row[key] for row in asset_rows) for key in counts},
            'training_assets':training_assets,'evaluation_assets':participated,
            'participating_assets':participated,'excluded_assets':excluded})
        for name,subset in [('train',train),('test',test)]:
            distributions.append({'fold_id':fold_id,'scope':'pooled','exchange':'ALL','asset_code':'ALL','partition':name,
                'asset_status':'available' if active else 'unavailable','use':'paired_participants_quality_audit_only',**target_statistics(subset,policy)})
    return {'version':policy['version'],'scope':'development_only','folds':folds,'assets':assets,'distributions':distributions,
        'features_materialized_in_phase4':False,'model_training_performed':False,'backtest_total_return_ready':False,
        'notes':['Test target summaries must not select thresholds, predictors, models or hyperparameters.',
                 'Per-asset candidate statistics include unavailable assets for transparent auditing only.',
                 'Both scopes share training eligibility based only on past retained rows; future counts affect evaluation only.',
                 'participating_assets aliases evaluation_assets, not training_assets; source history is never deleted.',
                 'Five-session targets overlap and are not independent observations.',
                 'Planned pooled scale-free predictors are registered but not materialized or model-ready.']}

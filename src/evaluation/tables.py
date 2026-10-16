"""Stratified frozen-prediction scores, exact scope pairs and additive loss diagnostics."""
import numpy as np
import pandas as pd
from src.evaluation.cohort import EVAL_KEYS,key_hash
from src.evaluation.config import validate_policy
from src.evaluation.metrics import CLASS_METRICS,REG_METRICS,LOWER_IS_BETTER,classification,regression,calibration

GROUP_KEYS=['level','fold_id','exchange','asset_code']


def groups(availability: pd.DataFrame) -> list[dict]:
    """Include registered unavailable cells, not only observed prediction groups."""
    rows=[{'level':'all','fold_id':'ALL','exchange':'ALL','asset_code':'ALL'}]
    for fold in sorted(availability.fold_id.unique()):
        rows.append({'level':'fold','fold_id':fold,'exchange':'ALL','asset_code':'ALL'})
    for exchange,code in sorted(set(zip(availability.exchange,availability.asset_code))):
        rows.append({'level':'asset','fold_id':'ALL','exchange':exchange,'asset_code':code})
    for fold,exchange,code in sorted(set(zip(availability.fold_id,availability.exchange,availability.asset_code))):
        rows.append({'level':'fold_asset','fold_id':fold,'exchange':exchange,'asset_code':code})
    return rows


def select_group(rows: pd.DataFrame, group: dict) -> pd.DataFrame:
    result=rows
    for name in ['fold_id','exchange','asset_code']:
        if group[name]!='ALL': result=result.loc[result[name]==group[name]]
    return result


def _methods(rows: pd.DataFrame, task: str) -> dict[str,np.ndarray]:
    values={'model':rows.prediction.to_numpy(dtype=float)}
    if task=='classification':
        values.update(majority_class=rows.majority_class.to_numpy(dtype=float),always_up=np.ones(len(rows)),
                      training_prior=rows.training_prior.to_numpy(dtype=float))
    else:
        values.update(zero_return=np.zeros(len(rows)),training_mean=rows.training_mean.to_numpy(dtype=float))
    return values


def _group_scores(rows: pd.DataFrame, identity: dict, task: str, policy: dict) -> tuple[list,list,list,list]:
    y=rows.actual.to_numpy(dtype=float);methods=_methods(rows,task);scorer=classification if task=='classification' else regression
    names=CLASS_METRICS if task=='classification' else REG_METRICS
    sha=key_hash(rows);scores=[];calibrations=[];comparisons=[];contributions=[];scored={}
    for method,pred in methods.items():
        metric=scorer(y,pred);scored[method]=metric
        scores.append({**identity,'method':method,'key_sha256':sha,'status':'available' if len(rows) else 'unavailable',**metric})
        if task=='classification':
            for item in calibration(y,pred,policy['calibration_edges']).to_dict('records'):
                calibrations.append({**identity,'method':method,'group_n':len(rows),'key_sha256':sha,**item})
    for baseline,pred in methods.items():
        if baseline=='model': continue
        for name in names:
            a=scored['model'][name];b=scored[baseline][name]
            improvement=b-a if name in LOWER_IS_BETTER else a-b
            comparisons.append({**identity,'baseline':baseline,'metric':name,'n':len(rows),
                'model_value':a,'baseline_value':b,'improvement':improvement,
                'status':'comparable' if np.isfinite(improvement) else 'undefined',
                'model_key_sha256':sha,'baseline_key_sha256':sha})
        if identity['level'] not in ['fold','asset','fold_asset']: continue
        diagnostics=['brier'] if task=='classification' else ['mae','squared_error']
        for metric in diagnostics:
            model_error=methods['model']-y;base_error=pred-y
            delta=np.abs(base_error)-np.abs(model_error) if metric=='mae' else base_error**2-model_error**2
            contributions.append({**identity,'baseline':baseline,'metric':metric,'n':len(rows),
                'loss_improvement_sum':float(delta.sum()),'mean_improvement':float(delta.mean()) if len(rows) else np.nan})
    return scores,calibrations,comparisons,contributions


def paired_scopes(joined: pd.DataFrame, availability: pd.DataFrame) -> pd.DataFrame:
    """No silent intersection: different keys produce explicit unavailable differences."""
    records=[];combinations=availability[['model','task']].drop_duplicates().sort_values(['model','task'])
    for model,task in combinations.itertuples(index=False,name=None):
        subset=joined.loc[(joined.model==model)&(joined.task==task)]
        for group in groups(availability):
            rows=select_group(subset,group)
            left=rows.loc[rows.scope=='per_asset'];right=rows.loc[rows.scope=='pooled']
            a_keys=set(map(tuple,left[EVAL_KEYS].itertuples(index=False,name=None)))
            b_keys=set(map(tuple,right[EVAL_KEYS].itertuples(index=False,name=None)))
            status='unavailable_no_predictions' if not a_keys and not b_keys else (
                'paired' if a_keys==b_keys else 'unavailable_mismatched_keys')
            scorer=classification if task=='classification' else regression
            a=scorer(left.actual,left.prediction);b=scorer(right.actual,right.prediction)
            for metric in CLASS_METRICS if task=='classification' else REG_METRICS:
                improvement=(a[metric]-b[metric] if metric in LOWER_IS_BETTER else b[metric]-a[metric]) if status=='paired' else np.nan
                records.append({**group,'model':model,'task':task,'metric':metric,'status':status,
                    'n_per_asset':len(left),'n_pooled':len(right),'n_common':len(a_keys&b_keys),
                    'per_asset_key_sha256':key_hash(left),'pooled_key_sha256':key_hash(right),
                    'per_asset_value':a[metric] if status=='paired' else np.nan,
                    'pooled_value':b[metric] if status=='paired' else np.nan,'improvement':improvement})
    return pd.DataFrame(records)


def stability_summary(comparison: pd.DataFrame, policy: dict) -> pd.DataFrame:
    """Counts and macro means describe consistency, not statistical significance."""
    records=[];keys=['model','scope','task','baseline','metric','level'];tol=policy['numerical_tie_tolerance']
    for identity,group in comparison.loc[comparison.level!='all'].groupby(keys,sort=True):
        valid=group.loc[np.isfinite(group.improvement)];delta=valid.improvement
        records.append({**dict(zip(keys,identity)),'groups_total':len(group),'groups_valid':len(valid),
            'wins':int((delta>tol).sum()),'ties':int((delta.abs()<=tol).sum()),'losses':int((delta<-tol).sum()),
            'macro_model_value':float(valid.model_value.mean()) if len(valid) else np.nan,
            'macro_baseline_value':float(valid.baseline_value.mean()) if len(valid) else np.nan,
            'macro_improvement':float(delta.mean()) if len(valid) else np.nan,
            'worst_improvement':float(delta.min()) if len(valid) else np.nan,
            'best_improvement':float(delta.max()) if len(valid) else np.nan})
    return pd.DataFrame(records)


def build_tables(joined: pd.DataFrame, parameters: pd.DataFrame, availability: pd.DataFrame, policy: dict) -> dict[str,pd.DataFrame]:
    """All methods score the same keys as their reference model, at four resolutions."""
    validate_policy(policy);joined=joined.sort_values(EVAL_KEYS+['model','scope']).reset_index(drop=True)
    scores=[];calibrations=[];comparisons=[];contributions=[]
    combinations=availability[['model','task','scope']].drop_duplicates().sort_values(['model','task','scope'])
    for model,task,scope in combinations.itertuples(index=False,name=None):
        subset=joined.loc[(joined.model==model)&(joined.task==task)&(joined.scope==scope)]
        for group in groups(availability):
            parts=_group_scores(select_group(subset,group),{**group,'model':model,'task':task,'scope':scope},task,policy)
            for target,part in zip([scores,calibrations,comparisons,contributions],parts): target.extend(part)
    metrics=pd.DataFrame(scores);base=pd.DataFrame(comparisons)
    return {'classification_metrics.csv':metrics.loc[(metrics.task=='classification')&(metrics.method=='model')].reset_index(drop=True),
            'regression_metrics.csv':metrics.loc[(metrics.task=='regression')&(metrics.method=='model')].reset_index(drop=True),
            'baseline_metrics.csv':metrics.loc[metrics.method!='model'].reset_index(drop=True),
            'baseline_comparison.csv':base,'calibration_results.csv':pd.DataFrame(calibrations),
            'per_asset_metrics.csv':metrics.loc[metrics.level.isin(['asset','fold_asset'])].reset_index(drop=True),
            'fold_metrics.csv':metrics.loc[metrics.level=='fold'].reset_index(drop=True),
            'paired_scope_comparison.csv':paired_scopes(joined,availability),'stability.csv':stability_summary(base,policy),
            'loss_contributions.csv':pd.DataFrame(contributions),'baseline_parameters.csv':parameters,'availability.csv':availability}

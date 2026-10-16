"""Descriptive scores only: no estimator fitting, p-values or decision optimization."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from src.data.research_policy import require

CLASS_METRICS=['roc_auc','brier','accuracy','balanced_accuracy','precision','recall','f1']
REG_METRICS=['mae','rmse','pearson','spearman']
LOWER_IS_BETTER={'brier','mae','rmse'}


def _arrays(y, p, *, binary: bool=False) -> tuple[np.ndarray,np.ndarray]:
    y=np.asarray(y,dtype=float);p=np.asarray(p,dtype=float)
    require(y.ndim==p.ndim==1 and len(y)==len(p),'Aligned one-dimensional arrays required')
    require(np.isfinite(y).all() and np.isfinite(p).all(),'Nonfinite evaluation values')
    if binary:
        require(np.isin(y,[0,1]).all() and ((p>=0)&(p<=1)).all(),'Invalid binary target/probability')
    return y,p


def classification(y, p) -> dict:
    """Binary Brier in [0,1], fixed p>=0.5 classification, explicit undefined reasons."""
    y,p=_arrays(y,p,binary=True);n=len(y);pred=p>=.5;up=y==1
    tn=int((~up&~pred).sum());fp=int((~up&pred).sum());fn=int((up&~pred).sum());tp=int((up&pred).sum())
    missing=[]
    def ratio(name,numerator,denominator):
        if denominator: return numerator/denominator
        missing.append(name+':zero_denominator');return float('nan')
    result={'n':n,'positive_count':int(up.sum()),'negative_count':n-int(up.sum()),'tn':tn,'fp':fp,'fn':fn,'tp':tp}
    result['accuracy']=ratio('accuracy',tp+tn,n)
    result['precision']=ratio('precision',tp,tp+fp);result['recall']=ratio('recall',tp,tp+fn)
    result['f1']=ratio('f1',2*tp,2*tp+fp+fn)
    if n:
        result['brier']=float(np.mean((p-y)**2))
    else:
        result['brier']=float('nan');missing.append('brier:no_samples')
    if tp+fn and tn+fp:
        result['roc_auc']=float(roc_auc_score(y,p));result['balanced_accuracy']=.5*(tp/(tp+fn)+tn/(tn+fp))
    else:
        result.update(roc_auc=float('nan'),balanced_accuracy=float('nan'))
        missing.extend(['roc_auc:requires_two_classes','balanced_accuracy:requires_two_classes'])
    result['undefined_reason']=';'.join(missing)
    return result


def regression(y, p) -> dict:
    """Price-return error and correlation; constants are not zero correlation."""
    y,p=_arrays(y,p);result={'n':len(y)};missing=[]
    if len(y):
        with np.errstate(over='raise',invalid='raise'):
            result['mae']=float(np.mean(np.abs(p-y)));result['rmse']=float(np.sqrt(np.mean((p-y)**2)))
    else:
        result.update(mae=float('nan'),rmse=float('nan'));missing.extend(['mae:no_samples','rmse:no_samples'])
    if len(y)<2 or np.unique(y).size<2 or np.unique(p).size<2:
        result.update(pearson=float('nan'),spearman=float('nan'))
        missing.extend(['pearson:constant_or_insufficient','spearman:constant_or_insufficient'])
    else:
        result['pearson']=float(np.corrcoef(y,p)[0,1])
        result['spearman']=float(np.corrcoef(pd.Series(y).rank(method='average'),pd.Series(p).rank(method='average'))[0,1])
    result['undefined_reason']=';'.join(missing)
    return result


def calibration(y, p, edges: list[float]) -> pd.DataFrame:
    """Fixed left-closed equal-width bins, including probability 1 in the last bin."""
    y,p=_arrays(y,p,binary=True);edges=np.asarray(edges,dtype=float)
    require(len(edges)==11 and np.allclose(edges,np.arange(11)/10,atol=1e-15,rtol=0),'Ten registered equal-width bins required')
    # Decimal cutpoints are canonicalized so binary linspace representation cannot move an exact edge.
    edges=np.arange(11)/10;bins=np.minimum(np.searchsorted(edges,p,side='right')-1,9);rows=[]
    for i in range(10):
        mask=bins==i;count=int(mask.sum())
        mean=float(p[mask].mean()) if count else float('nan');rate=float(y[mask].mean()) if count else float('nan')
        rows.append({'bin':i,'bin_lower':edges[i],'bin_upper':edges[i+1],'upper_inclusive':i==9,
                     'count':count,'positive_count':int(y[mask].sum()),'mean_probability':mean,'observed_rate':rate,'gap':rate-mean})
    return pd.DataFrame(rows)

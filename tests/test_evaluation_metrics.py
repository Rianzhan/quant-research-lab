"""Hand-checked scores, honest undefined values and fixed calibration boundaries."""
import importlib
from pathlib import Path
import math
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]


def api():
    assert (ROOT/'src/evaluation/metrics.py').exists(), 'Evaluation metrics missing'
    return importlib.import_module('src.evaluation.metrics')


def test_hand_classification_metrics():
    m=api().classification([0,0,1,1],[.1,.8,.4,.9])
    assert m['n']==4 and m['positive_count']==2
    for name in ['accuracy','balanced_accuracy','precision','recall','f1']: assert m[name]==.5
    assert [m[k] for k in ['tn','fp','fn','tp']]==[1,1,1,1]
    assert m['roc_auc']==.75 and m['brier']==pytest.approx(.255)
    assert m['undefined_reason']==''


def test_fixed_half_tie_is_positive_and_auc_handles_ties():
    m=api().classification([0,0,1,1],[.5,.49,.5,.51])
    assert m['roc_auc']==.875 and m['accuracy']==.75 and m['balanced_accuracy']==.75
    assert m['precision']==pytest.approx(2/3) and m['recall']==1 and m['f1']==.8


@pytest.mark.parametrize('y,p,missing',[([1,1],[.2,.8],['roc_auc','balanced_accuracy']),
                                      ([0,1],[.1,.1],['precision']),
                                      ([0,0],[0,0],['roc_auc','balanced_accuracy','precision','recall','f1']),
                                      ([],[],['roc_auc','brier','accuracy','balanced_accuracy','precision','recall','f1'])])
def test_undefined_classification_is_not_fabricated(y,p,missing):
    m=api().classification(y,p)
    for name in missing:
        assert math.isnan(m[name]) and name in m['undefined_reason']
    if y==[0,1]: assert m['f1']==0 and m['recall']==0


def test_regression_scores_and_tied_ranks_hand_values():
    m=api().regression([1,2,3],[1,3,2])
    assert m['mae']==pytest.approx(2/3) and m['rmse']==pytest.approx(math.sqrt(2/3))
    assert m['pearson']==pytest.approx(.5) and m['spearman']==pytest.approx(.5)
    m=api().regression([1,1,2,2],[1,2,2,3])
    assert m['spearman']==pytest.approx(1/math.sqrt(2))


@pytest.mark.parametrize('y,p',[([.1,.2,.3],[0,0,0]),([1,1],[1,2]),([1],[2]),([],[])])
def test_constant_or_short_regression_correlation_is_missing(y,p):
    m=api().regression(y,p)
    assert math.isnan(m['pearson']) and math.isnan(m['spearman'])
    assert 'pearson' in m['undefined_reason'] and 'spearman' in m['undefined_reason']
    if y: assert math.isfinite(m['mae']) and math.isfinite(m['rmse'])


def test_calibration_exact_edges_empty_bins_and_endpoint_one():
    table=api().calibration([0,1,0,1,1],[0,.1,.2,.9,1],np.linspace(0,1,11).tolist())
    assert len(table)==10 and table['count'].sum()==5
    assert table['count'].tolist()==[1,1,1,0,0,0,0,0,0,2]
    assert table.iloc[9].mean_probability==pytest.approx(.95) and table.iloc[9].observed_rate==1
    assert table.loc[table['count']==0,['mean_probability','observed_rate','gap']].isna().all().all()


@pytest.mark.parametrize('y,p',[([0,1],[.1]),([0,2],[.1,.3]),([0,1],[-.1,.3]),
                              ([0,1],[.1,1.1]),([0,1],[.1,np.nan]),([0,np.inf],[.1,.3])])
def test_invalid_classification_rejected(y,p):
    with pytest.raises(ValueError): api().classification(y,p)


@pytest.mark.parametrize('y,p',[([1],[2,3]),([np.nan],[2]),([1],[np.inf]),([[1,2]],[[2,3]])])
def test_invalid_regression_rejected(y,p):
    with pytest.raises(ValueError): api().regression(y,p)


def test_metric_order_invariance_and_baseline_auc():
    y=[0,0,1,1];p=[.1,.8,.4,.9]
    assert api().classification(y,p)==api().classification(y[::-1],p[::-1])
    assert api().classification(y,[.6]*4)['roc_auc']==.5

"""Model behavior, train-only transforms and deterministic trusted round trips."""
import copy
import importlib
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import pytest
from test_scale_free import scale_registration

ROOT=Path(__file__).resolve().parents[1]
PAIRS=[('logistic','classification'),('random_forest','classification'),('random_forest','regression'),
       ('xgboost','classification'),('xgboost','regression')]


def api():
    assert (ROOT/'src/models/factory.py').exists(), 'Phase5 model factory missing'
    return importlib.import_module('src.models.factory')


@pytest.fixture
def model_case(phase3_config,scale_registration):
    config=copy.deepcopy(phase3_config)
    config['models']['random_forest'].update(n_estimators=8,min_samples_leaf=2)
    config['models']['xgboost']['n_estimators']=8
    names=scale_registration['shared_whitelist']
    rng=np.random.default_rng(202)
    X=pd.DataFrame(rng.normal(size=(80,17)),columns=names,index=range(100,180))
    y=pd.Series((X.iloc[:,0]+X.iloc[:,1]>.0).astype(int),index=X.index)
    returns=pd.Series(X.iloc[:,0]*.01+X.iloc[:,1]*.003,index=X.index)
    return config,names,X,y,returns


@pytest.mark.parametrize('family,task',PAIRS)
def test_unified_fit_predict_repeated_fit_and_saved_state(model_case,family,task):
    c,names,X,classes,returns=model_case;y=classes if task=='classification' else returns
    before=X.copy(deep=True);model=api().create_model(family,task,c,names)
    with pytest.raises(ValueError,match='fitted'): model.predict(X.iloc[60:])
    assert model.fit(X.iloc[:60],y.iloc[:60]) is model
    values=model.predict(X.iloc[60:])
    assert values.shape==(20,) and np.isfinite(values).all()
    if task=='classification': assert ((values>=0)&(values<=1)).all()
    other=api().create_model(family,task,c,names).fit(X.iloc[:60],y.iloc[:60])
    np.testing.assert_array_equal(values,other.predict(X.iloc[60:]))
    payload=pickle.dumps(model,protocol=5)
    assert payload==pickle.dumps(other,protocol=5)
    restored=pickle.loads(payload)
    np.testing.assert_array_equal(values,restored.predict(X.iloc[60:]))
    assert model.predict(X.iloc[:0]).shape==(0,)
    with pytest.raises(ValueError,match='fresh'): model.fit(X.iloc[:60],y.iloc[:60])
    pd.testing.assert_frame_equal(X,before)
    assert model.metadata()['parameters']['random_state']==42


def test_scaler_learns_literal_training_values_only(model_case):
    c,names,X,y,_=model_case
    X.loc[:,:]=0.
    X.loc[X.index[:4],names[0]]=[0.,2.,4.,6.]
    y.iloc[:4]=[0,0,1,1]
    X.loc[X.index[4:],names[0]]=1000000.
    model=api().create_model('logistic','classification',c,names).fit(X.iloc[:4],y.iloc[:4])
    scaler=model.estimator.named_steps['scaler']
    assert scaler.mean_[0]==3. and scaler.var_[0]==5. and scaler.n_samples_seen_==4
    before=pickle.dumps(model,protocol=5)
    model.predict(X.iloc[4:])
    assert pickle.dumps(model,protocol=5)==before


@pytest.mark.parametrize('fault',['extra_target','missing','order','nan','inf','string','label_index','label_nan','one_class'])
def test_invalid_model_input_is_rejected(model_case,fault):
    c,names,X,y,_=model_case
    if fault=='extra_target': X['classification_target']=y
    if fault=='missing': X=X.drop(columns=names[0])
    if fault=='order': X=X[X.columns[::-1]]
    if fault=='nan': X.iloc[0,0]=np.nan
    if fault=='inf': X.iloc[0,0]=np.inf
    if fault=='string': X[names[0]]=X[names[0]].astype(str)
    if fault=='label_index': y=y.iloc[::-1]
    if fault=='label_nan': y=y.astype(float);y.iloc[0]=np.nan
    if fault=='one_class': y[:]=1
    with pytest.raises(ValueError): api().create_model('logistic','classification',c,names).fit(X,y)


@pytest.mark.parametrize('family,task',PAIRS)
def test_predicting_other_rows_does_not_refit_or_change_prior_predictions(model_case,family,task):
    c,names,X,classes,returns=model_case;y=classes if task=='classification' else returns
    model=api().create_model(family,task,c,names).fit(X.iloc[:60],y.iloc[:60])
    before=model.predict(X.iloc[60:65]);state=pickle.dumps(model,protocol=5)
    changed=X.iloc[65:].copy()*100
    model.predict(changed)
    np.testing.assert_array_equal(before,model.predict(X.iloc[60:65]))
    assert state==pickle.dumps(model,protocol=5)


def test_unsupported_task_and_target_whitelist_cannot_create_model(model_case):
    c,names,*_=model_case
    with pytest.raises(ValueError): api().create_model('logistic','regression',c,names)
    with pytest.raises(ValueError): api().create_model('unknown','classification',c,names)
    with pytest.raises(ValueError): api().create_model('logistic','classification',c,['label_end_date']+names[1:])

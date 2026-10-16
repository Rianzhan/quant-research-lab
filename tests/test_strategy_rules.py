"""No labels, future opens or strategy returns in the decision contract."""
from dataclasses import replace
from decimal import Decimal as D
import pytest
from src.portfolio.rules import Rules, AssetState, DecisionState, project_predictions, schedule, targets, rules_hash

DAYS=['2022-01-04','2022-01-05','2022-01-06','2022-01-07','2022-01-10','2022-01-11','2022-01-12']

def state(n=2, date=DAYS[0]):
    return DecisionState(date,'100000','100000',tuple(AssetState(str(i),'SSE','2018-01-02','10') for i in range(n)))

def pred(code='0',value='.6',strategy='A',date=DAYS[0],**extra):
    scope,model,task=('per_asset','logistic','classification') if strategy=='A' else ('pooled','random_forest','regression')
    return dict(date=date,asset_code=code,exchange='SSE',price_segment='0',fold_id='test_'+date[:4],scope=scope,model=model,task=task,prediction=value,**extra)

def test_calendar_anchor_order_and_no_annual_reset():
    assert schedule(DAYS,Rules())==[DAYS[0],DAYS[5]]
    assert schedule(list(reversed(DAYS)),Rules())==schedule(DAYS,Rules())
    cross=['2022-12-29','2022-12-30','2023-01-03','2023-01-04','2023-01-05','2023-01-06']
    assert schedule(cross,replace(Rules(),anchor='2022-12-29'))==[cross[0],cross[5]]
    with pytest.raises(ValueError):schedule(DAYS+['2026-01-05'],Rules())
    with pytest.raises(ValueError):schedule(DAYS+[DAYS[0]],Rules())

@pytest.mark.parametrize('strategy,value,eligible',[('A','.55',False),('A','.55001',True),('B','.001',False),('B','.001001',True)])
def test_strict_fixed_hurdles(strategy,value,eligible):
    r=targets(strategy,project_predictions([pred(value=value,strategy=strategy)]),state(),Rules())
    assert r[0]['weight']==(D('.25') if eligible else 0)
    assert r[1]['reason']=='missing_prediction'

@pytest.mark.parametrize('n,weight',[(0,0),(1,'.25'),(2,'.25'),(3,'.25'),(4,'.25'),(5,'.2'),(8,'.125')])
def test_capped_equal_weights_cash_and_long_only(n,weight):
    out=targets('C',[],state(n),Rules())
    assert all(x['weight']==D(str(weight)) for x in out)
    assert all(0<=x['weight']<=D('.25') for x in out)
    assert sum(x['weight'] for x in out)<=1

def test_labels_and_future_open_are_never_read():
    class Guard(dict):
        def __getitem__(self,key):
            assert key not in {'label','future_return_5d','classification_target','regression_target','next_open'}
            return super().__getitem__(key)
    a=project_predictions([Guard(pred(label=1,next_open=999))])
    b=project_predictions([pred(label=-999,next_open=0)])
    assert a==b and targets('A',a,state(),Rules())==targets('A',b,state(),Rules())

def test_repeat_order_and_other_model_cannot_change_signals():
    rows=[pred('0'),pred('1',value='.7'),pred('0',value='.0001',strategy='B')]
    a=project_predictions(rows);b=project_predictions(rows[::-1])
    assert a==b
    assert targets('A',a,state(),Rules())==targets('A',b,state(),Rules())
    assert rules_hash(Rules())==rules_hash(Rules())

def test_588000_model_missing_does_not_remove_benchmark():
    a=AssetState('588000','SSE','2020-11-16','1')
    s=replace(state(),assets=(a,))
    assert targets('A',[],s,Rules())[0]['weight']==0
    assert targets('C',[],s,Rules())[0]['weight']==D('.25')
    before=replace(s,date='2020-11-13')
    assert targets('C',[],before,Rules())[0]['weight']==0

def test_stale_wrong_basis_and_pending_are_not_eligible():
    s=state(3)
    s=replace(s,assets=(replace(s.assets[0],close=None),replace(s.assets[1],basis_valid=False),replace(s.assets[2],pending_action=True)))
    assert all(x['weight']==0 for x in targets('C',[],s,Rules()))

@pytest.mark.parametrize('change',[{'date':'2026-01-05'},{'prediction':'NaN'},{'prediction':'1.1'},{'fold_id':'test_2023'},{'price_segment':'-1'}])
def test_invalid_prediction_contract_fails(change):
    p=pred();p.update(change)
    with pytest.raises(ValueError):project_predictions([p])

def test_duplicate_prediction_and_wrong_segment_refused_or_missing():
    with pytest.raises(ValueError):project_predictions([pred(),pred()])
    p=pred();p['price_segment']='1'
    assert targets('A',project_predictions([p]),state(),Rules())[0]['weight']==0

def test_future_predictions_cannot_change_today():
    a=project_predictions([pred(),pred(value='.99',date=DAYS[1])])
    b=project_predictions([pred(),pred(value='.01',date=DAYS[1])])
    assert targets('A',a,state(),Rules())==targets('A',b,state(),Rules())

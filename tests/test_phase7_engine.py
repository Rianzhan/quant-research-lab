"""Frozen strategies connect to the accepted ledger using synthetic markets only."""
from dataclasses import replace
from decimal import Decimal as D
import importlib
from pathlib import Path
import pytest
from src.backtest.contracts import Asset,Quote,Action,Policy,stable_bytes
from src.portfolio.rules import Rules,project_predictions,rules_hash

DAYS=['2022-01-04','2022-01-05','2022-01-06','2022-01-07','2022-01-10','2022-01-11',
      '2022-01-12','2022-01-13','2022-01-14','2022-01-17','2022-01-18','2022-01-19']

def api():
    assert (Path(__file__).parents[1]/'src/backtest/phase7/engine.py').exists(),'Phase7 account connector missing'
    return importlib.import_module('src.backtest.phase7.engine')

def predictions(strategy='A',value='.6',dates=None,code='X',**extra):
    scope,model,task=('per_asset','logistic','classification') if strategy=='A' else ('pooled','random_forest','regression')
    return project_predictions([dict(date=d,asset_code=code,exchange='SSE',price_segment='0',fold_id='test_'+d[:4],
        scope=scope,model=model,task=task,prediction=value,**extra) for d in (dates or DAYS)])

def inputs(days=None):
    days=days or DAYS
    return dict(assets=[Asset('X','2018-01-02')],sessions=days,
        quotes=[Quote(d,'X','10','10') for d in days],exchanges={'X':'SSE'},
        segments={(d,'X'):0 for d in days},actions=[])

def run(strategy='A',preds=None,fee='0.0003',**overrides):
    kwargs=inputs();kwargs.update(overrides)
    return api().run_account(strategy,replace(Policy(),fee_rate=fee),predictions() if preds is None else preds,**kwargs)

@pytest.mark.parametrize('strategy,value,fills',[('A','.55',False),('A','.55001',True),('B','.001',False),('B','.001001',True)])
def test_strict_frozen_thresholds(strategy,value,fills):
    r=run(strategy,predictions(strategy,value))
    assert bool(r.tables['fills'])==fills
    assert r.rule_hash==rules_hash(Rules())

def test_next_open_never_sizes_previous_order_and_no_chasing():
    a=run(sessions=DAYS[:3],quotes=[Quote(d,'X','10','10') for d in DAYS[:3]])
    b=run(sessions=DAYS[:3],quotes=[Quote(DAYS[0],'X','10','10'),Quote(DAYS[1],'X','100','100'),Quote(DAYS[2],'X','10','10')])
    assert a.tables['orders'][0]['frozen_shares']==b.tables['orders'][0]['frozen_shares']==2500
    c=run(sessions=DAYS[:3],quotes=[Quote(DAYS[0],'X','10','10'),Quote(DAYS[2],'X','10','10')])
    assert not c.tables['fills'] and c.tables['orders'][0]['status']=='not_filled_missing_open'
    assert all(f['date']>f['decision_date'] for f in a.tables['fills'])

def test_no_predictions_model_zero_benchmark_invests_and_588000_not_filled_in():
    kwargs=inputs();kwargs['assets']=[Asset('588000','2020-11-16')]
    kwargs['quotes']=[replace(q,asset_code='588000') for q in kwargs['quotes']]
    kwargs['exchanges']={'588000':'SSE'};kwargs['segments']={(d,'588000'):0 for d in DAYS}
    a=api().run_account('A',Policy(),[],**kwargs);c=api().run_account('C',Policy(),[],**kwargs)
    assert not a.tables['fills'] and c.tables['fills']
    assert all(x['reason']=='missing_prediction' and x['weight']==0 for x in a.signals)

def test_cost_scenarios_share_signals_not_quantities_or_account_balances():
    out=[run('B',predictions('B','.002'),fee=f) for f in Rules().fee_scenarios]
    assert len({stable_bytes(r.signals) for r in out})==1
    for r in out:
        assert r.policy.slippage=='0.0002'
        for d in r.tables['daily']:
            assert 0<=d['gross_asset_value']<=d['nav'] and d['cash']>=0 and abs(d['unexplained_pnl'])<=D('1e-8')
        assert all(x['shares']>=0 for x in r.tables['holdings'])

def test_year_boundary_schedule_never_resets_and_end_never_liquidates():
    days=[*DAYS[:3],'2022-12-30','2023-01-03','2023-01-04','2023-01-05','2025-12-31']
    r=run('C',[],**inputs(days))
    assert sorted({x['date'] for x in r.signals})==[days[0],days[5]]
    assert r.tables['holdings'][-1]['shares']>0
    assert all(o['decision_date']!='2025-12-31' for o in r.tables['orders'])

def test_dividends_accrue_then_payment_only_moves_assets():
    action=Action('div','X','cash_dividend','2022-01-03T23:59:59+08:00',DAYS[2],record_date=DAYS[1],
        payment_date=DAYS[4],cash_per_share='1',evidence_sha256='a'*64)
    r=run('C',[],actions=[action],quotes=[Quote(d,'X','10' if i<2 else '9','10' if i<2 else '9') for i,d in enumerate(DAYS)])
    accrual=[j for j in r.tables['journal'] if j['kind']=='dividend_accrual'][0]
    payment=[j for j in r.tables['journal'] if j['kind']=='dividend_payment'][0]
    assert accrual['receivable_delta']==2500
    assert payment['cash_delta']==2500 and payment['receivable_delta']==-2500 and payment['expected_nav_change']==0

def test_after_close_split_does_not_size_with_old_price_basis():
    action=Action('split','X','split','2022-01-03T23:59:59+08:00',DAYS[5],phase='after_close',record_date=DAYS[5],
        share_ratio='2',rounding='exact',evidence_sha256='a'*64)
    r=run('C',[],actions=[action],quotes=[Quote(d,'X','10' if i<=5 else '5','10' if i<=5 else '5') for i,d in enumerate(DAYS)])
    event=[j for j in r.tables['journal'] if j['kind']=='share_adjustment'][0]
    assert event['expected_nav_change']==event['unexplained_pnl']==0
    assert not [o for o in r.tables['orders'] if o['decision_date']==DAYS[5]]
    assert [x for x in r.signals if x['date']==DAYS[5]][0]['reason']=='invalid_price_basis'

def test_replay_reordering_and_extra_future_labels_do_not_change_results():
    a=run();kwargs=inputs()
    for name in ['assets','sessions','quotes','actions']:kwargs[name]=kwargs[name][::-1]
    b=run(preds=predictions(label=999,regression_target=-999)[::-1],**kwargs)
    assert a.fingerprint()==b.fingerprint()

def test_unfrozen_rules_costs_or_2026_rejected():
    with pytest.raises(ValueError):run('D')
    with pytest.raises(ValueError):run(fee='.0007')
    with pytest.raises(ValueError):run(sessions=DAYS+['2026-01-05'])
    with pytest.raises(ValueError):run(rules=replace(Rules(),classification_threshold='.5'))


@pytest.mark.parametrize('n',[1,2,3,4,8])
def test_integrated_capped_weights_and_gap_cannot_create_leverage(n):
    codes=[f'X{i}' for i in range(n)];days=DAYS[:2]
    kwargs=dict(assets=[Asset(c,'2018-01-02') for c in codes],sessions=days,
        quotes=[Quote(d,c,'10' if d==days[0] else '12','10' if d==days[0] else '12') for c in codes for d in days],
        exchanges={c:'SSE' for c in codes},segments={(d,c):0 for c in codes for d in days},actions=[])
    r=api().run_account('C',Policy(),[],**kwargs)
    assert all(0<=x['weight']<=D('.25') for x in r.signals)
    assert sum(x['weight'] for x in r.signals)<=1
    assert all(x['cash']>=0 and x['gross_asset_value']<=x['nav'] for x in r.tables['daily'])
    assert all(x['shares']>=0 for x in r.tables['holdings'])

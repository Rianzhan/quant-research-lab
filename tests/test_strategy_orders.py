"""Freeze quantities at decision time; execution remains the accepted engine."""
from dataclasses import replace,asdict
from decimal import Decimal as D,localcontext
import pytest
from src.portfolio.rules import Rules,AssetState,DecisionState,project_predictions
from src.portfolio.orders import map_orders,prepare_execution
from src.backtest.contracts import Policy,Asset,Quote,stable_bytes
from test_strategy_rules import DAYS,state,pred

def test_prediction_t_order_t_plus_one_and_fixed_quantity():
    r=map_orders('A',project_predictions([pred()]),state(),Rules(),DAYS)
    assert len(r.orders)==1
    o=r.orders[0]
    assert o.execution_date==DAYS[1] and o.intent.decision_date==DAYS[0]
    assert o.intent.requested_shares=='2500' and o.intent.source=='strategy_spec:A'
    for next_open in ['10','100']:
        sim=prepare_execution(Policy(),[Asset('0','2018-01-02'),Asset('1','2018-01-02')],DAYS[:2],
            [Quote(DAYS[0],'0','10','10'),Quote(DAYS[1],'0',next_open,next_open)],r.orders,[])
        sim.session(DAYS[0],DAYS[1],D('100000'))
        assert not sim.tables['fills']
        assert sim.tables['orders'][0]['frozen_shares']==2500
        sim.session(DAYS[1],None,sim.tables['daily'][-1]['nav'])
        assert sim.tables['orders'][0]['source']=='strategy_spec:A'
        assert sim.book.cash>=0

def test_missing_next_open_does_not_chase():
    r=map_orders('A',project_predictions([pred()]),state(),Rules(),DAYS)
    sim=prepare_execution(Policy(),[Asset('0','2018-01-02'),Asset('1','2018-01-02')],DAYS[:3],
        [Quote(DAYS[0],'0','10','10'),Quote(DAYS[2],'0','10','10')],r.orders,[])
    nav=D('100000')
    for i,d in enumerate(DAYS[:3]):nav=sim.session(d,DAYS[i+1] if i<2 else None,nav)
    assert not sim.tables['fills'] and sim.tables['orders'][0]['status']=='not_filled_missing_open'

def test_missing_prediction_closes_existing_target_without_fabricated_cash():
    s=state();s=replace(s,cash_available='99000',assets=(replace(s.assets[0],shares='100'),s.assets[1]))
    r=map_orders('A',[],s,Rules(),DAYS)
    assert r.targets[0]['reason']=='missing_prediction'
    assert r.orders[0].intent.side=='sell' and r.orders[0].intent.requested_shares=='100'
    assert s.assets[0].shares=='100' and s.cash_available=='99000'

def test_no_rebalance_or_no_next_session_no_order():
    assert map_orders('C',[],state(date=DAYS[1]),Rules(),DAYS).status=='not_rebalance'
    assert not map_orders('C',[],state(),Rules(),DAYS[:1]).orders

def test_buy_budget_lots_and_no_sale_proceeds_advance():
    s=state(3);s=replace(s,cash_available='11',assets=tuple(replace(a,shares='0',lot_size='1') for a in s.assets))
    r=map_orders('C',[],s,Rules(),DAYS)
    assert len(r.orders)==1 and r.orders[0].intent.requested_shares=='1'
    s=replace(s,cash_available='0',assets=(replace(s.assets[0],shares='10000'),*s.assets[1:]))
    r=map_orders('A',[],s,Rules(),DAYS)
    assert all(o.intent.side=='sell' for o in r.orders)

def test_pending_portfolio_blocks_other_target_orders():
    s=state();s=replace(s,assets=(replace(s.assets[0],shares='1',pending_action=True),s.assets[1]))
    r=map_orders('C',[],s,Rules(),DAYS)
    assert not r.orders and r.status=='blocked_pending_portfolio_valuation'

def test_invalid_close_does_not_silently_liquidate():
    s=state();s=replace(s,assets=(replace(s.assets[0],close=None,shares='100'),s.assets[1]))
    r=map_orders('A',[],s,Rules(),DAYS)
    assert not r.orders and r.blocked[0]['reason']=='missing_legal_decision_close'

def test_repeat_reorder_labels_and_regression_only_prediction():
    rows=[pred('0',strategy='B',value='.01'),pred('1',strategy='B',value='.02')]
    s=state();a=map_orders('B',project_predictions(rows),s,Rules(),DAYS)
    rows[0]['regression_target']=999;rows[1]['label']=-999
    b=map_orders('B',project_predictions(rows[::-1]),replace(s,assets=s.assets[::-1]),Rules(),DAYS[::-1])
    assert stable_bytes(asdict(a))==stable_bytes(asdict(b))

def test_prelisting_negative_shares_and_fake_future_state_fail():
    s=state()
    for a in [replace(s.assets[0],shares='-1'),replace(s.assets[0],shares='1',first_tradable_date='2023-01-01')]:
        with pytest.raises(ValueError):map_orders('C',[],replace(s,assets=(a,)),Rules(),DAYS)
    with pytest.raises(ValueError):map_orders('C',[],replace(s,date='2026-01-05'),Rules(),DAYS)

def test_execution_adapter_preserves_40_digit_accounting_precision():
    r=map_orders('A',project_predictions([pred()]),state(),Rules(),DAYS)
    price='7.2132999999999999999999999'
    sim=prepare_execution(Policy(),[Asset('0','2018-01-02'),Asset('1','2018-01-02')],DAYS[:2],
        [Quote(DAYS[0],'0','10','10'),Quote(DAYS[1],'0',price,price)],r.orders,[])
    sim.session(DAYS[0],DAYS[1],D('100000'));sim.session(DAYS[1],None,D('100000'))
    with localcontext() as ctx:
        ctx.prec=40
        expected=D('100000')-D('2500')*D(price)*D('1.0002')*D('1.0003')
    assert sim.book.cash==expected

def test_modified_frozen_order_or_rule_identity_refused():
    r=map_orders('A',project_predictions([pred()]),state(),Rules(),DAYS)
    o=r.orders[0]
    for changed in [replace(o,intent=replace(o.intent,requested_shares='3000')),replace(o,rule_hash='0'*64)]:
        with pytest.raises(ValueError):prepare_execution(Policy(),[Asset('0','2018-01-02')],DAYS[:2],[],[changed],[])


@pytest.mark.parametrize('mix',['strategy','targets','snapshot','duplicate'])
def test_execution_refuses_mixed_decision_batches(mix):
    s=state()
    a=map_orders('A',project_predictions([pred('0')]),s,Rules(),DAYS)
    if mix=='strategy':
        b=map_orders('C',[],s,Rules(),DAYS)
    elif mix=='targets':
        b=map_orders('A',project_predictions([pred('1')]),s,Rules(),DAYS)
    elif mix=='snapshot':
        b=map_orders('A',project_predictions([pred('1')]),replace(s,nav='90000',cash_available='90000'),Rules(),DAYS)
    else:b=a
    with pytest.raises(ValueError):
        prepare_execution(Policy(),[Asset('0','2018-01-02'),Asset('1','2018-01-02')],DAYS[:2],[],a.orders+b.orders,[])


def test_execution_rejects_calendar_override_before_any_mutation():
    r=map_orders('A',project_predictions([pred()]),state(),Rules(),DAYS)
    sim=prepare_execution(Policy(),[Asset('0','2018-01-02')],DAYS[:3],
        [Quote(DAYS[0],'0','10','10'),Quote(DAYS[2],'0','10','10')],r.orders,[])
    for date,next_date in [(DAYS[0],DAYS[2]),(DAYS[1],DAYS[2]),(DAYS[0],None)]:
        with pytest.raises(ValueError):sim.session(date,next_date,D('100000'))
    assert not sim.tables['orders'] and not sim.tables['daily']
    nav=sim.session(DAYS[0],DAYS[1],D('100000'))
    with pytest.raises(ValueError):sim.session(DAYS[0],DAYS[1],nav)
    nav=sim.session(DAYS[1],DAYS[2],nav)
    sim.session(DAYS[2],None,nav)
    assert not sim.tables['fills']
    assert sim.tables['orders'][0]['status']=='not_filled_missing_open'

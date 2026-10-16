"""Literal economic examples, never model predictions or real holdout files."""
from dataclasses import replace
from decimal import Decimal as D
import pytest

from src.backtest.contracts import Action, Asset, OrderIntent, Policy, Quote
from src.backtest.engine import run
from src.backtest.accounting import reconcile_day

DAYS = ['2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05', '2024-01-08', '2024-01-09']


def intent(day=0, side='buy', qty='10', asset='A', oid='o1', weight=None):
    return OrderIntent(oid, DAYS[day], asset, side, DAYS[day]+'T23:59:59+08:00',
                       'manual_test', requested_shares=qty, target_weight=weight)


def market(prices=None, assets=('A',)):
    prices = prices or ['10']*len(DAYS)
    return [Quote(day, code, price, price) for day, price in zip(DAYS, prices) for code in assets]


def execute(orders, quotes=None, actions=(), policy=None, assets=None):
    return run(policy or Policy(initial_cash='1000', fee_rate='0', slippage='0'),
               assets or [Asset('A', DAYS[0], lot_size='1')], DAYS,
               quotes if quotes is not None else market(), orders, actions)


def dividend(**changes):
    return replace(Action('div1', 'A', 'cash_dividend', '2024-01-01T23:59:59+08:00',
                          DAYS[2], record_date=DAYS[1], payment_date=DAYS[4],
                          cash_per_share='1', evidence_sha256='a'*64), **changes)


def split(**changes):
    return replace(Action('split1', 'A', 'split', '2024-01-01T23:59:59+08:00',
                          DAYS[2], phase='after_close', record_date=DAYS[1],
                          share_ratio='2', rounding='exact', evidence_sha256='a'*64,
                          suspended_dates=(DAYS[2],)), **changes)


def test_next_session_and_no_same_day_fill():
    r=execute([intent()]); f=r.tables['fills'][0]
    assert f['date']==DAYS[1] and f['decision_date']==DAYS[0]
    assert r.tables['daily'][0]['cash']==D('1000')


@pytest.mark.parametrize('bad', [None, '0', '-1', 'NaN'])
def test_missing_illegal_next_open_never_chases_future_quote_or_charges(bad):
    q=market();q[1]=replace(q[1], open=bad)
    r=execute([intent()],q)
    assert not r.tables['fills']
    assert r.tables['orders'][0]['status']=='not_filled_missing_open'
    assert all(d['fees_paid']==0 and d['cash']==1000 for d in r.tables['daily'])


def test_slippage_directions_and_fees_each_side_actual_notional():
    r=execute([intent(),intent(1,'sell','10',oid='sell')],
              policy=Policy(initial_cash='1000',fee_rate='0.001',slippage='0.0002'))
    a,b=r.tables['fills']
    assert (a['effective_price'],b['effective_price'])==(D('10.002'),D('9.998'))
    assert (a['fee'],b['fee'])==(D('0.10002'),D('0.09998'))
    assert a['slippage_cost']==b['slippage_cost']==D('0.02')
    assert r.tables['daily'][-1]['cash']==D('999.76')
    assert r.tables['daily'][1]['turnover']==D('0.10002')


def test_cash_gap_downsizes_to_lot_without_negative_cash():
    q=market(['10','20','20','20','20','20'])
    r=execute([intent(qty='100')], q,
              assets=[Asset('A',DAYS[0],lot_size='10')],
              policy=Policy(initial_cash='1000',fee_rate='0.001',slippage='0.0002'))
    assert r.tables['orders'][0]['frozen_shares']==D('100')
    assert r.tables['fills'][0]['shares']==D('40')
    assert r.tables['daily'][1]['cash']==D('199.03984')


def test_cannot_afford_one_lot_no_fill():
    r=execute([intent(qty='100')],policy=Policy(initial_cash='9',fee_rate='0',slippage='0'))
    assert not r.tables['fills'] and r.tables['orders'][0]['status']=='not_filled_cash_or_shares'


def test_sell_without_shares_never_shorts():
    r=execute([intent(side='sell')])
    assert not r.tables['fills'] and all(h['shares']==0 for h in r.tables['holdings'])


def test_oversell_is_capped_to_owned_shares():
    r=execute([intent(),intent(1,'sell','999',oid='s')])
    assert [f['shares'] for f in r.tables['fills']]==[D('10'),D('10')]


def test_negative_requested_quantity_rejected():
    with pytest.raises(ValueError,match='shares'): execute([intent(qty='-1')])


def test_target_weight_cannot_request_leverage():
    with pytest.raises(ValueError,match='weight'): execute([intent(qty=None,weight='1.01')])


def test_target_sizing_uses_decision_close_not_tomorrow_open():
    p=Policy(initial_cash='1000',fee_rate='0',slippage='0')
    o=intent(qty=None,weight='0.5')
    a=execute([o],policy=p);b=execute([o],market(['10','15','15','15','15','15']),policy=p)
    assert a.tables['orders'][0]['frozen_shares']==b.tables['orders'][0]['frozen_shares']==D('50')
    assert b.tables['fills'][0]['shares']==D('50')


def test_entitlement_ex_accrual_payment_are_three_distinct_events():
    r=execute([intent()],market(['10','10','9','9','9','9']),[dividend()])
    d=r.tables['daily']
    assert d[1]['dividend_receivable']==0 and d[1]['nav']==1000
    assert d[2]['dividend_receivable']==10 and d[2]['cash']==900 and d[2]['nav']==1000
    assert d[4]['dividend_receivable']==0 and d[4]['cash']==910 and d[4]['nav']==1000
    assert [e['kind'] for e in r.tables['receivables']]==['entitlement','accrual','payment']


def test_selling_after_record_does_not_destroy_dividend_right():
    # Ex-date order crossing is cancelled; sell the following session.
    r=execute([intent(),intent(2,'sell','10',oid='sell')],
              market(['10','10','9','9','9','9']),[dividend()])
    assert r.tables['daily'][3]['dividend_receivable']==10
    assert r.tables['daily'][4]['cash']==1000


def test_buying_after_record_gets_no_past_dividend():
    r=execute([intent(2)],market(['10','10','9','9','9','9']),[dividend()])
    assert all(d['dividend_receivable']==0 for d in r.tables['daily'])


def test_receivable_and_payment_day_open_cannot_fund_purchase():
    r=execute([intent(qty='100'),intent(2,qty='1',oid='before'),intent(3,qty='1',oid='payment'),
               intent(4,qty='1',oid='after')],market(['10','10','9','9','9','9']),[dividend()])
    byid={o['order_id']:o for o in r.tables['orders']}
    assert byid['before']['status']==byid['payment']['status']=='not_filled_cash_or_shares'
    assert byid['after']['status']=='filled'


def test_pure_split_changes_shares_not_nav_even_while_halted():
    q=market(['10','10',None,'5','5','5'])
    r=execute([intent()],q,[split()])
    h=[h for h in r.tables['holdings'] if h['date']==DAYS[2]][0]
    assert h['shares']==20 and h['valuation_price']==5 and h['valuation_stale']
    assert h['price_basis_adjustment_event']=='split1'
    assert all(d['nav']==1000 for d in r.tables['daily'])


def test_corporate_action_crossing_cancels_old_order():
    r=execute([intent(),intent(1,qty='1',oid='cross')],market(['10','10',None,'5','5','5']),[split()])
    assert r.tables['orders'][1]['status']=='cancelled_due_to_corporate_action'
    assert len(r.tables['fills'])==1


def test_halt_blocks_supplier_open_stale_valuation_never_becomes_execution_price():
    q=market();q[1]=replace(q[1],suspended=True,close=None)
    r=execute([intent()],q)
    assert not r.tables['fills'] and r.tables['orders'][0]['status']=='not_filled_suspended'


def test_stale_holdings_do_not_mutate_market_or_fill_orders():
    q=market();q[2]=replace(q[2],open=None,close=None);before=tuple(q)
    r=execute([intent(),intent(1,qty='5',oid='missing')],q)
    h=r.tables['holdings'][2]
    assert h['valuation_price']==10 and h['valuation_stale']
    assert len(r.tables['fills'])==1 and tuple(q)==before and q[2].open is None


def test_assets_cannot_leak_prices_or_holdings():
    assets=[Asset('A',DAYS[0]),Asset('B',DAYS[0])];q=market(assets=('A','B'))
    r=execute([intent()],q,assets=assets)
    altered=[replace(x,open='99',close='99') if x.asset_code=='B' else x for x in q]
    r2=execute([intent()],altered,assets=assets)
    assert r.tables['daily']==r2.tables['daily']
    assert all(x['shares']==0 for x in r2.tables['holdings'] if x['asset_code']=='B')


def test_input_order_and_repeated_hashes_equal():
    q=market();o=[intent(),intent(2,'sell','5',oid='sell')];a=execute(o,q)
    b=execute(list(reversed(o)),list(reversed(q)))
    assert a.fingerprint()==b.fingerprint()==execute(o,q).fingerprint()


@pytest.mark.parametrize('fee,slip,cash', [('0','0','1000'),('0','0.0002','999.96'),('0.001','0','999.8')])
def test_zero_cost_degeneracies(fee,slip,cash):
    r=execute([intent(),intent(1,'sell','10',oid='sell')],
              policy=Policy(initial_cash='1000',fee_rate=fee,slippage=slip))
    assert r.tables['daily'][-1]['cash']==D(cash)


def test_sale_proceeds_reuse_policy_is_explicit():
    q=market(assets=('A','B'));assets=[Asset('A',DAYS[0]),Asset('B',DAYS[0])]
    orders=[intent(qty='100'),intent(1,'sell','100',oid='sell'),intent(1,qty='100',asset='B',oid='buy')]
    a=execute(orders,q,assets=assets)
    b=execute(orders,q,assets=assets,policy=Policy(initial_cash='1000',fee_rate='0',slippage='0',sale_proceeds_reuse_policy='same_day_reuse'))
    assert len(a.tables['fills'])==2 and len(b.tables['fills'])==3
    assert a.tables['daily'][2]['cash']==1000 and a.tables['daily'][2]['cash_available']==0
    assert a.tables['daily'][3]['cash_available']==1000


def test_identity_and_unexplained_pnl_corruption_fail():
    d=execute([intent()]).tables['daily'][1]
    with pytest.raises(ValueError,match='identity'): reconcile_day({**d,'nav':D('1001')},D('0.00000001'))
    with pytest.raises(ValueError,match='unexplained'): reconcile_day({**d,'market_revaluation':D('2')},D('0.00000001'))


def test_2026_market_rows_and_decisions_rejected():
    q=market()+[Quote('2026-01-05','A','10','10')]
    with pytest.raises(ValueError,match='development'): execute([intent()],q)


def test_prelisting_intent_does_not_create_position():
    r=execute([intent()],assets=[Asset('A',DAYS[3])])
    assert not r.tables['fills'] and r.tables['orders'][0]['status']=='not_filled_not_tradable'
    assert all(h['valuation_price'] is None for h in r.tables['holdings'][:3])


def test_missing_action_fields_and_late_known_ratio_reject_window():
    for a in [dividend(payment_date=None),split(known_at='2024-01-08T23:59:59+08:00')]:
        with pytest.raises(ValueError,match='incomplete event'): execute([intent()],actions=[a])


def test_terminal_order_is_not_executed_past_development_window():
    r=execute([intent(5)])
    assert not r.tables['fills'] and r.tables['orders'][0]['status']=='cancelled_no_next_session'


def test_after_close_split_cannot_size_target_with_old_basis_close():
    action=split(suspended_dates=(),record_date=DAYS[2])
    r=execute([intent(),intent(2,qty=None,weight='0.5',oid='target')],
              market(['10','10','10','5','5','5']),[action])
    assert r.tables['orders'][1]['status']=='cancelled_missing_decision_close'


def test_no_prior_close_cannot_silently_value_new_holding_at_open():
    q=[replace(x,close=None) for x in market()]
    with pytest.raises(ValueError,match='last legal close'): execute([intent()],q)


def test_split_rounding_is_separate_explained_value():
    r=execute([intent(qty='3')],market(['10','10',None,'4','4','4']),
              [split(share_ratio='2.5',rounding='ceil_whole_share')])
    assert r.tables['holdings'][2]['shares']==8
    assert r.tables['daily'][2]['share_rounding_value']==2
    assert r.tables['daily'][2]['nav']==1002


def test_reconciliation_rejects_short_or_cash_borrowing_independently():
    from src.backtest.accounting import Book
    b=Book(D('100'),['A'],D('0.00000001'));b.prices['A']=D('10');b.shares['A']=D('-1')
    with pytest.raises(ValueError,match='shares'): b.check()
    b.shares['A']=D('20');b.cash=D('-100')
    with pytest.raises(ValueError,match='cash'): b.check()


def test_future_price_perturbation_leaves_past_journal_identical():
    a=execute([intent()]);q=market();q[3:]=[replace(x,open='50',close='50') for x in q[3:]]
    b=execute([intent()],q)
    assert a.tables['daily'][:3]==b.tables['daily'][:3]
    assert [x for x in a.tables['journal'] if x['date']<=DAYS[2]]==[x for x in b.tables['journal'] if x['date']<=DAYS[2]]

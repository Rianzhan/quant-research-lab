"""Late-known scenarios are synthetic, never a rewrite of the real 510500 notice."""
from dataclasses import replace
from decimal import Decimal as D
import pytest
from src.backtest.contracts import Action,Asset,OrderIntent,Policy,Quote
from src.backtest.pipeline import validate_result

DATES=['2022-08-24','2022-08-25','2022-08-26','2022-08-29','2022-08-30','2022-08-31']


def intent(i,oid,qty='100',code='A',weight=None):
    return OrderIntent(oid,DATES[i],code,'buy',DATES[i]+'T23:59:59+08:00','manual_test',requested_shares=qty,target_weight=weight)


def scenario(known='2022-08-29T23:59:59+08:00',ratio='2',reverse=False,late_price=None,other_weight=False):
    from src.backtest.phase6a5.engine import DelayedShareAction,run_research
    a=DelayedShareAction('delayed','A','split',known,DATES[2],phase='after_close',
        record_date=DATES[2],share_ratio=ratio,rounding='exact',evidence_sha256='a'*64,
        announced_at='2022-08-23T23:59:59+08:00')
    assets=[Asset('A',DATES[0]),Asset('B',DATES[0])]
    quotes=[Quote(d,c,'10' if c=='B' or i<3 else '5','10' if c=='B' or i<3 else '5')
            for i,d in enumerate(DATES) for c in ['A','B']]
    if late_price is not None:
        quotes=[replace(q,open=late_price,close=late_price) if q.asset_code=='A' and q.date>=DATES[3] else q for q in quotes]
    orders=[intent(0,'initial'),intent(2,'old','10'),intent(3,'new','10'),intent(2,'other','10','B')]
    if other_weight:
        orders.append(intent(2,'other_weight',None,'B','0.2'))
    args=[assets,DATES,quotes,orders,[a]]
    if reverse:args=[list(reversed(x)) for x in args]
    return run_research(Policy(initial_cash='10000',fee_rate='0',slippage='0'),*args)


def test_pending_cancel_no_chase_and_zero_nav_resolution():
    r=scenario();validate_result(r.result,initial_cash=D('10000'))
    assert [f['order_id'] for f in r.result.tables['fills']]==['initial','other','new']
    old=next(o for o in r.result.tables['orders'] if o['order_id']=='old')
    assert old['status']=='cancelled_pending_corporate_action'
    assert not [f for f in r.result.tables['fills'] if f['asset_code']=='A' and f['date']==DATES[3]]
    assert all(d['nav']==D('10000') for d in r.result.tables['daily'])
    h=[h for h in r.result.tables['holdings'] if h['asset_code']=='A']
    assert h[2]['shares']==100 and h[3]['shares']==200
    event=[j for j in r.result.tables['journal'] if j['kind']=='share_adjustment']
    assert len(event)==1 and event[0]['date']==DATES[3] and event[0]['nav_before']==event[0]['nav_after']


def test_future_ratio_cannot_change_pre_resolution_state():
    a=scenario();b=scenario(ratio='3')
    for table in ['daily','holdings','journal','orders']:
        key='decision_date' if table=='orders' else 'date'
        assert [x for x in a.result.tables[table] if x[key]<=DATES[2]]==[x for x in b.result.tables[table] if x[key]<=DATES[2]]
    assert all(x['ratio_visible'] is None for x in a.states if x['date']<=DATES[2])


def test_unresolved_and_later_quotes_cannot_change_pending_state():
    a=scenario(known='',ratio=None);b=scenario(known='',ratio=None,late_price='999')
    assert a.result.fingerprint()==b.result.fingerprint()
    assert a.states==b.states
    a=scenario();b=scenario(late_price='999')
    for table in ['daily','holdings','journal','orders']:
        key='decision_date' if table=='orders' else 'date'
        assert [x for x in a.result.tables[table] if x[key]<=DATES[2]]==[x for x in b.result.tables[table] if x[key]<=DATES[2]]


def test_other_asset_target_weight_cannot_use_pending_portfolio_nav():
    r=scenario(other_weight=True)
    order=next(o for o in r.result.tables['orders'] if o['order_id']=='other_weight')
    assert order['status']=='cancelled_pending_portfolio_valuation'
    assert order['frozen_shares']==0 and order['target_weight']=='0.2'
    assert not [f for f in r.result.tables['fills'] if f['order_id']=='other_weight']
    assert any(f['order_id']=='other' for f in r.result.tables['fills'])


def test_preopen_known_path_a_and_afterclose_path_b():
    a=scenario('2022-08-29T08:00:00+08:00');b=scenario()
    assert next(x for x in a.states if x['date']==DATES[3])['resolved_phase']=='pre_open'
    assert next(x for x in b.states if x['date']==DATES[3])['resolved_phase']=='close_after'


def test_unresolved_keeps_gate_false_and_stale_economic_value():
    r=scenario(known='',ratio=None)
    assert not r.resolved and all(d['nav']==10000 for d in r.result.tables['daily'])
    assert r.states[-1]['valuation_status']=='stale_economic_valuation'
    assert r.states[-1]['ratio_visible'] is None
    assert all(f['date']<DATES[2] for f in r.result.tables['fills'] if f['asset_code']=='A')


def test_reordering_and_repeated_hashes():
    a=scenario();b=scenario(reverse=True);c=scenario()
    assert a.result.fingerprint()==b.result.fingerprint()==c.result.fingerprint()
    assert a.states==b.states==c.states


def test_other_asset_is_not_contaminated_by_unknown_ratio():
    a=scenario();b=scenario(ratio='3')
    assert [x for x in a.result.tables['holdings'] if x['asset_code']=='B']==[x for x in b.result.tables['holdings'] if x['asset_code']=='B']


def test_market_inputs_not_changed_and_holdout_rows_refused():
    from src.backtest.phase6a5.engine import run_research
    quotes=[Quote(d,'A','10','10') for d in DATES];snapshot=list(quotes)
    r=run_research(Policy(),[Asset('A',DATES[0])],DATES,quotes,[intent(0,'x')])
    assert quotes==snapshot and set(r.result.tables)=={'orders','fills','daily','holdings','receivables','journal'}
    with pytest.raises(ValueError,match='Non-development'):
        run_research(Policy(),[Asset('A',DATES[0])],DATES,quotes+[Quote('2026-01-05','A','10','10')],[])


def test_real_known_split_separates_zero_conversion_from_rounding_compensation():
    from src.backtest.phase6a5.engine import run_research
    a=Action('real','A','split','2022-08-23T23:59:59+08:00',DATES[2],phase='after_close',
        record_date=DATES[2],share_ratio='1.14539',rounding='ceil_whole_share',evidence_sha256='a'*64)
    r=run_research(Policy(initial_cash='10000',fee_rate='0',slippage='0'),[Asset('A',DATES[0])],
        DATES[:3],[Quote(d,'A','10','10') for d in DATES[:3]],[intent(0,'x')],[a])
    postings=[j for j in r.result.tables['journal'] if j['kind']=='share_adjustment']
    assert postings[0]['expected_nav_change']==0
    assert postings[1]['reference']=='real:issuer_rounding_compensation'
    assert postings[1]['expected_nav_change']>0 and r.result.tables['holdings'][-1]['shares']==115
    validate_result(r.result,initial_cash=D('10000'))

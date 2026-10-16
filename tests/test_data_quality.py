"""Hand-built prices catch date, unit and cross-asset errors; no real data invented."""
import importlib
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
A = {'code': '510300', 'exchange': 'SSE', 'name': 'A', 'category': 'china_broad'}
B = {'code': '159915', 'exchange': 'SZSE', 'name': 'B', 'category': 'growth_technology'}

def api():
    assert (ROOT/'src/data/cleaner.py').exists(), 'data cleaner not implemented'
    return importlib.import_module('src.data.cleaner')

def raw(dates=('2024-01-02', '2024-01-03'), price=10):
    return pd.DataFrame({'date': list(dates), 'open': price, 'high': price+1,
                         'low': price-1, 'close': price, 'volume': 200, 'amount': price*200})

def clean(frame=None, asset=A, source='akshare_sina', **kw):
    return api().clean_prices(raw() if frame is None else frame, asset, source, 'snap-1', **kw)

def test_sort_without_filling_or_mutating_source():
    frame=raw(('2024-01-04','2024-01-02')); before=frame.copy(deep=True)
    result=clean(frame)
    assert result.date.tolist()==['2024-01-02','2024-01-04']
    assert result.volume.tolist()==[200,200]
    pd.testing.assert_frame_equal(frame,before)

@pytest.mark.parametrize('column,value', [('open',0),('high',8),('low',12),('volume',-1),
    ('amount',-1),('close',float('inf')),('close',None),('date','2024-01-02T23:00:00Z'),('date','bad')])
def test_invalid_market_rows_are_rejected(column,value):
    frame=raw(); frame[column]=frame[column].astype(object); frame.loc[0,column]=value
    with pytest.raises(api().DataQualityError): clean(frame)

def test_duplicate_dates_rejected_even_if_identical():
    with pytest.raises(api().DataQualityError,match='duplicate'): clean(raw(('2024-01-02','2024-01-02')))

def test_foreign_asset_and_adjustment_rejected():
    frame=raw(); frame['code']='sz.159915'
    with pytest.raises(api().DataQualityError,match='identity'): clean(frame)
    with pytest.raises(api().DataQualityError,match='adjust'): clean(price_basis='qfq')

def test_unbalanced_panel_preserves_early_history_and_asset_prices():
    a=clean(raw(price=10)); b=clean(raw(('2024-01-03',),price=100), B)
    panel=api().combine_assets([a,b], [A,B])
    assert len(panel)==3
    assert panel.groupby('asset_code').close.apply(list).to_dict()=={'159915':[100],'510300':[10,10]}
    assert not ((panel.asset_code=='159915') & (panel.date=='2024-01-02')).any()
    assert panel[['asset_code','exchange','asset_name','category','snapshot_id']].notna().all().all()

def test_panel_rejects_missing_asset_and_wrong_metadata():
    a=clean()
    with pytest.raises(api().DataQualityError,match='asset'): api().combine_assets([a],[A,B])
    a.loc[0,'exchange']='SZSE'
    with pytest.raises(api().DataQualityError,match='metadata'): api().combine_assets([a],[A])

def test_eastmoney_volume_lots_are_converted_to_shares():
    frame=raw().rename(columns={'date':'日期','open':'开盘','high':'最高','low':'最低','close':'收盘','volume':'成交量','amount':'成交额'})
    frame['成交量']=2
    result=clean(frame,source='akshare_em')
    assert result.volume.tolist()==[200,200]
    assert result.amount.tolist()==[2000,2000]

def test_missing_amount_stays_missing_and_zero_volume_is_observed():
    frame=raw().drop(columns='amount'); frame['volume']=0
    result=clean(frame)
    assert result.amount.isna().all() and result.volume.eq(0).all()

def test_listing_evidence_prevents_prelisting_rows():
    with pytest.raises(api().DataQualityError,match='listing'): clean(listing_date='2024-01-03')

def test_calendar_quality_keeps_missing_and_noncommon_dates_visible():
    q=importlib.import_module('src.data.quality') if (ROOT/'src/data/quality.py').exists() else None
    assert q is not None, 'quality auditor not implemented'
    a=clean(raw(('2024-01-02','2024-01-04'))); b=clean(raw(('2024-01-04',),price=100),B)
    p=api().combine_assets([a,b],[A,B])
    report=q.audit_panel(p,['2024-01-02','2024-01-03','2024-01-04'],[A,B],'2024-01-02','2024-01-04')
    assert report['common_dates']==['2024-01-04']
    assert report['noncommon_dates']==['2024-01-02']
    assert report['assets']['SSE:510300']['internal_missing_dates']==['2024-01-03']
    assert report['assets']['SZSE:159915']['leading_unobserved_dates']==['2024-01-02','2024-01-03']
    assert report['assets']['SZSE:159915']['listing_status']=='unverified'

def test_source_conflict_is_not_silently_accepted():
    q=importlib.import_module('src.data.quality') if (ROOT/'src/data/quality.py').exists() else None
    assert q is not None, 'quality auditor not implemented'
    a=clean(); b=clean(raw(price=20))
    result=q.compare_sources(a,b,price_tolerance=0.001,volume_relative_tolerance=0.02)
    assert result['conflict'] is True and result['overlap_rows']==2

def test_quality_flags_price_discontinuity_without_adjusting_it():
    q=importlib.import_module('src.data.quality')
    frame=raw(price=10.0); frame.loc[1,['open','high','low','close','amount']]=[2,2.1,1.9,2,400]
    p=clean(frame)
    report=q.audit_panel(p,['2024-01-02','2024-01-03'],[A],'2024-01-02','2024-01-03')
    assert report['assets']['SSE:510300']['large_price_break_dates']==['2024-01-03']
    assert p.close.tolist()==[10,2]

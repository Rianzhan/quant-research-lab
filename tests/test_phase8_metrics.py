"""Hand-derived NAV metrics; no market data or model labels."""
import importlib,math
from pathlib import Path
import numpy as np
import pytest

def api():
    assert (Path(__file__).parents[1]/'src/evaluation/phase8/metrics.py').exists(),'NAV metrics missing'
    return importlib.import_module('src.evaluation.phase8.metrics')

def test_returns_cagr_sample_volatility_and_sharpe_hand_calculation():
    m,curve,ep=api().nav_metrics(['2022-01-04','2022-01-05','2022-01-06'],['100','110','99'],'100')
    r=np.array([0,.1,-.1]);std=r.std(ddof=1)
    assert curve['daily_return'].tolist()==pytest.approx(r)
    assert m['total_return']==pytest.approx(-.01)
    assert m['cagr']==pytest.approx(.99**(252/3)-1)
    assert m['annualized_volatility']==pytest.approx(std*math.sqrt(252))
    assert m['sharpe_ratio']==pytest.approx(0)
    assert m['maximum_drawdown']==pytest.approx(.1)

def test_recovered_drawdown_peak_trough_and_durations():
    dates=['2022-01-03','2022-01-04','2022-01-05','2022-01-06','2022-01-07']
    m,curve,ep=api().nav_metrics(dates,[100,110,88,100,110],100,'2021-12-31')
    assert m['maximum_drawdown']==pytest.approx(.2)
    assert m['drawdown_peak_date']=='2022-01-04' and m['drawdown_trough_date']=='2022-01-05'
    assert m['recovery_date']=='2022-01-07' and m['drawdown_duration_sessions']==3
    assert ep.iloc[0]['underwater_sessions']==2 and ep.iloc[0]['duration_calendar_days']==3

def test_initial_loss_unrecovered_and_last_equal_peak():
    m,_,e=api().nav_metrics(['2022-01-04','2022-01-05'],[80,90],100,'2022-01-03')
    assert m['maximum_drawdown']==pytest.approx(.2) and m['recovery_date'] is None
    assert m['recovery_status']=='unrecovered' and m['drawdown_duration_sessions']==2
    assert e.iloc[0]['peak_observation']=='initial_capital'
    m,_,_=api().nav_metrics(['2022-01-04','2022-01-05','2022-01-06','2022-01-07'],[110,110,88,110],100)
    assert m['drawdown_peak_date']=='2022-01-05' and m['drawdown_duration_sessions']==2

def test_flat_single_and_no_drawdown_are_explicit_not_infinite():
    m,_,e=api().nav_metrics(['2022-01-04','2022-01-05'],[100,100],100)
    assert m['sharpe_ratio'] is None and 'zero_volatility' in m['undefined_reason']
    assert m['maximum_drawdown']==0 and e.empty and m['recovery_status']=='no_drawdown'
    m,_,_=api().nav_metrics(['2022-01-04'],[110],100)
    assert m['annualized_volatility'] is None and m['sharpe_ratio'] is None

def test_annual_first_day_uses_previous_year_balance_and_compounds():
    first,_,_=api().nav_metrics(['2022-12-30'],[110],100)
    second,_,_=api().nav_metrics(['2023-01-03','2023-01-04'],[99,121],110,'2022-12-30')
    assert second['total_return']==pytest.approx(.1)
    assert (1+first['total_return'])*(1+second['total_return'])-1==pytest.approx(.21)

def test_input_order_cannot_change_metric_or_episode_and_invalid_inputs_fail():
    args=(['2022-01-04','2022-01-05','2022-01-06'],[100,90,95],100)
    a=api().nav_metrics(*args);b=api().nav_metrics(args[0][::-1],args[1][::-1],100)
    assert a[0]==b[0];assert a[1].equals(b[1]);assert a[2].equals(b[2])
    for dates,nav,initial in [(['2026-01-05'],[100],100),(['2022-01-04']*2,[100,100],100),(['2022-01-04'],[0],100),(['2022-01-04'],[float('nan')],100),([],[],100)]:
        with pytest.raises(ValueError):api().nav_metrics(dates,nav,initial)

def test_equal_decimal_trough_keeps_first_date_beyond_float_precision():
    m,_,_=api().nav_metrics(['2022-01-04','2022-01-05','2022-01-06','2022-01-07'],
        ['100','99.999999999999999999','99.999999999999999999','100'],'100')
    assert m['drawdown_trough_date']=='2022-01-05'
def test_nonzero_constant_returns_have_zero_volatility_and_undefined_sharpe():
    from src.evaluation.phase8.metrics import nav_metrics
    metrics,curve,_=nav_metrics(['2022-01-04','2022-01-05','2022-01-06'],['110','121','133.1'],'100')
    assert list(curve.daily_return)==[.1,.1,.1]
    assert metrics['annualized_volatility']==0
    assert metrics['sharpe_ratio'] is None
    assert metrics['undefined_reason']=='zero_volatility'

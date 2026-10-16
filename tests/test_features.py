"""Known numerical values and perturbation tests pin the causal feature contract."""
import importlib
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def api(module='builder'):
    assert (ROOT/f'src/features/{module}.py').exists(), 'Phase 3 feature implementation missing'
    return importlib.import_module('src.features.'+module)


def test_hand_calculated_returns_momentum_means_and_volatility():
    p = pd.Series([10.,12.,9.,15.,18.,21.])
    returns = api('returns').historical_returns(p,[1,5])
    assert returns.return_5d.iloc[-1] == pytest.approx(1.1)
    assert returns.return_5d.iloc[:5].isna().all()
    momentum = api('momentum').momentum(p,[5])
    pd.testing.assert_series_equal(momentum.momentum_5d,returns.return_5d,check_names=False)
    means = api('technical').moving_averages(p,[2,3],[[2,3]])
    assert means.ma_diff_2_3.iloc[4] == pytest.approx(2.5)
    assert means.ma_3d.iloc[:2].isna().all()
    volatility = api('volatility').historical_volatility(p,[3],1)
    assert volatility.volatility_3d.iloc[3] == pytest.approx(np.std([.2,-.25,2/3],ddof=1))
    assert volatility.volatility_3d.iloc[:3].isna().all()


def test_wilder_rsi_seed_recursion_and_degenerate_cases():
    technical = api('technical')
    actual = technical.wilder_rsi(pd.Series([10.,12.,11.,14.,14.,12.]),2)
    assert actual.iloc[:2].isna().all()
    assert actual.iloc[2:].tolist() == pytest.approx([200/3,800/9,800/9,32])
    assert technical.wilder_rsi(pd.Series([10.]*20),14).iloc[14:].eq(50).all()
    assert technical.wilder_rsi(pd.Series(range(1,21),dtype=float),14).iloc[14:].eq(100).all()
    assert technical.wilder_rsi(pd.Series(range(20,0,-1),dtype=float),14).iloc[14:].eq(0).all()


def test_ema_and_macd_use_visible_arithmetic_seed_and_reset_at_missing():
    technical = api('technical'); p=pd.Series([1.,2.,4.,8.,16.])
    ema = technical.seeded_ema(p,3)
    assert ema.iloc[:2].isna().all()
    assert ema.iloc[2:].tolist() == pytest.approx([7/3,31/6,127/12])
    macd = technical.macd(p,2,3,2)
    assert macd.macd_line.iloc[2] == pytest.approx(5/6)
    assert macd.macd_signal.iloc[3] == pytest.approx(37/36)
    assert macd.macd_hist.iloc[3] == pytest.approx(7/36)
    gap=technical.seeded_ema(pd.Series([1.,2.,3.,np.nan,10.,20.,30.]),3)
    assert gap.iloc[3:6].isna().all() and gap.iloc[6]==20


def test_volume_does_not_fill_or_skip_missing():
    v=pd.Series([100.,200.,np.nan,400.,800.,1600.])
    result=api('volume').volume_features(v,1,3)
    assert result.volume_change_1d.iloc[[0,2,3]].isna().all()
    assert result.volume_change_1d.iloc[[1,4,5]].eq(1).all()
    assert result.volume_ratio_3d.iloc[:5].isna().all()
    assert result.volume_ratio_3d.iloc[5]==pytest.approx(12/7)


@pytest.mark.parametrize('cut',[20,61,95])
def test_future_prices_or_truncation_never_change_past_features(make_panel,phase3_config,cut):
    panel,calendar=make_panel(event_at=60)
    before=api().build_features(panel,phase3_config,calendar)
    future=panel.date>calendar[cut]
    changed=panel.copy()
    changed.loc[future,['close','research_close']]*=7
    changed.loc[future,['volume','research_volume']]*=13
    after=api().build_features(changed,phase3_config,calendar)
    pd.testing.assert_frame_equal(before.loc[before.date<=calendar[cut]],after.loc[after.date<=calendar[cut]],check_exact=True)
    truncated=api().build_features(panel.loc[~future],phase3_config,calendar)
    pd.testing.assert_frame_equal(before.loc[before.date<=calendar[cut]].reset_index(drop=True),truncated,check_exact=True)


def test_other_asset_and_long_format_order_cannot_contaminate_target(make_panel,phase3_config):
    panel,calendar=make_panel(event_at=60)
    before=api().build_features(panel,phase3_config,calendar)
    changed=panel.copy();other=changed.asset_code=='510500'
    changed.loc[other,['close','research_close']]*=20
    changed.loc[other,['volume','research_volume']]*=15
    after=api().build_features(changed.iloc[::-1],phase3_config,calendar)
    pd.testing.assert_frame_equal(before.loc[before.asset_code=='510300'],after.loc[after.asset_code=='510300'],check_exact=True)
    alone=api().build_features(panel.loc[panel.asset_code=='510300'],phase3_config,calendar)
    pd.testing.assert_frame_equal(before.loc[before.asset_code=='510300'].reset_index(drop=True),alone,check_exact=True)


@pytest.mark.parametrize('boundary',['action','calendar_gap'])
def test_every_feature_restarts_after_event_or_missing_session(make_panel,phase3_config,boundary):
    panel,calendar=make_panel(event_at=60 if boundary=='action' else None)
    start=60
    if boundary=='calendar_gap':
        panel=panel.loc[panel.date!=calendar[59]]
    whole=api().build_features(panel,phase3_config,calendar)
    fresh=api().build_features(panel.loc[panel.date>=calendar[start]],phase3_config,calendar)
    pd.testing.assert_frame_equal(whole.loc[whole.date>=calendar[start]].reset_index(drop=True),fresh,check_exact=True)
    first=fresh.loc[fresh.date==calendar[start]]
    assert first[[s['name'] for s in api().feature_specs(phase3_config)]].isna().all().all()


@pytest.mark.parametrize('invalid',['flag_false_positive_value','flag_true_nan'])
def test_invalid_volume_masks_current_and_dependent_windows_only(make_panel,phase3_config,invalid):
    panel,calendar=make_panel()
    before=api().build_features(panel,phase3_config,calendar)
    mask=(panel.asset_code=='510300')&(panel.date==calendar[70])
    if invalid=='flag_false_positive_value': panel.loc[mask,'volume_valid']=False
    else: panel.loc[mask,'research_volume']=np.nan
    after=api().build_features(panel,phase3_config,calendar)
    asset=after.loc[after.asset_code=='510300'].reset_index(drop=True)
    assert asset.volume_change_1d.iloc[70:72].isna().all()
    assert asset.volume_ratio_20d.iloc[70:90].isna().all()
    assert pd.notna(asset.volume_ratio_20d.iloc[90])
    price_cols=[s['name'] for s in api().feature_specs(phase3_config) if not s['uses_volume']]
    pd.testing.assert_frame_equal(before[price_cols],after[price_cols],check_exact=True)


def test_all_four_registered_zero_volume_dates_remain_missing(make_panel,phase3_config):
    panel,calendar=make_panel(days=520,start='2023-01-02')
    asset=next(a for a in phase3_config['universe']['assets'] if a['code']=='159915')
    first=panel.asset_code=='510300'
    for column,key in [('asset_code','code'),('exchange','exchange'),('asset_name','name'),('category','category')]:
        panel.loc[first,column]=asset[key]
    dates=['2023-12-12','2023-12-14','2023-12-15','2024-11-01']
    mask=(panel.asset_code=='159915')&panel.date.isin(dates)
    # Synthetic prices only; the four real anomaly dates exercise exact-date masking.
    panel.loc[mask,['volume','research_volume']]=[0,np.nan]
    panel.loc[mask,'volume_valid']=False
    out=api().build_features(panel,phase3_config,calendar)
    target=(out.asset_code=='159915')&out.date.isin(dates)
    assert len(out.loc[target])==4
    assert out.loc[target,['volume_change_1d','volume_ratio_20d']].isna().all().all()


@pytest.mark.parametrize('fault',['holdout','duplicate','segment_reversal','unknown_asset','adjusted_price','bad_boolean'])
def test_invalid_panel_fails_explicitly(make_panel,phase3_config,fault):
    panel,calendar=make_panel(event_at=60)
    if fault=='holdout': panel.loc[0,'date']='2026-01-05'
    if fault=='duplicate': panel=pd.concat([panel,panel.iloc[:1]],ignore_index=True)
    if fault=='segment_reversal': panel.loc[(panel.asset_code=='510300')&(panel.date==calendar[70]),'price_segment']=0
    if fault=='unknown_asset': panel.loc[0,'asset_code']='999999'
    if fault=='adjusted_price': panel.loc[0,'research_close']*=2
    if fault=='bad_boolean': panel['volume_valid']=panel.volume_valid.astype(str)
    with pytest.raises(ValueError): api().build_features(panel,phase3_config,calendar)

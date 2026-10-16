"""Hand arithmetic and causal perturbations of the registered ratio features."""
import importlib
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import yaml
from phase4_helpers import validation_policy
from src.features.builder import build_features
from src.utils.research_rows import KEYS

ROOT=Path(__file__).resolve().parents[1]
RATIOS=['close_over_ma_5d','close_over_ma_20d','close_over_ma_60d','ma_ratio_5_20','ma_ratio_20_60',
        'macd_line_over_close','macd_signal_over_close','macd_hist_over_close']


def api():
    assert (ROOT/'src/features/scale_free.py').exists(), 'Scale-free implementation missing'
    return importlib.import_module('src.features.scale_free')


@pytest.fixture
def scale_registration():
    return yaml.safe_load((ROOT/'config/scale_free.yaml').read_text(encoding='utf-8'))


def from_panel(panel,config,calendar):
    """Rebuild causal Phase3 dependencies for synthetic perturbation tests."""
    return build_features(panel,config,calendar).merge(panel[KEYS+['research_close']],on=KEYS,validate='one_to_one')


@pytest.fixture
def hand_frame(validation_policy):
    row={name:.01 for name in validation_policy['feature_registration']['shared_raw_whitelist']}
    row.update(date='2021-01-04',exchange='SSE',asset_code='510300',price_segment=0,research_close=12.,
               ma_5d=10.,ma_20d=8.,ma_60d=6.,ma_diff_5_20=2.,ma_diff_20_60=2.,
               macd_line=3.,macd_signal=-1.5,macd_hist=4.5)
    return pd.DataFrame([row])


def test_ratio_arithmetic_and_shared_model_inputs(hand_frame,phase3_config,validation_policy,scale_registration):
    result=api().build_scale_free(hand_frame,['2021-01-04'],phase3_config,validation_policy,scale_registration)
    assert result[RATIOS].iloc[0].tolist()==pytest.approx([.2,.5,1.,.25,1/3,.25,-.125,.375])
    assert list(result)==KEYS+scale_registration['shared_whitelist'] and len(result.columns)==21
    assert 'classification_target' not in result and 'ma_5d' not in result
    bad=dict(scale_registration,shared_whitelist=['classification_target']+scale_registration['shared_whitelist'][1:])
    with pytest.raises(ValueError): api().build_scale_free(hand_frame,['2021-01-04'],phase3_config,validation_policy,bad)
    bad=dict(scale_registration,model_scopes=['pooled'])
    with pytest.raises(ValueError): api().build_scale_free(hand_frame,['2021-01-04'],phase3_config,validation_policy,bad)


@pytest.mark.parametrize('dependency,invalid,affected',[
    ('ma_20d',np.nan,['close_over_ma_20d','ma_ratio_5_20','ma_ratio_20_60']),
    ('research_close',np.nan,['close_over_ma_5d','close_over_ma_20d','close_over_ma_60d',*RATIOS[-3:]]),
    ('macd_signal',np.nan,['macd_signal_over_close']),
    ('ma_5d',0.,['close_over_ma_5d','ma_ratio_5_20']),
    ('ma_60d',-1.,['close_over_ma_60d','ma_ratio_20_60']),
])
def test_missing_or_nonpositive_price_dependencies_remain_missing(hand_frame,phase3_config,validation_policy,scale_registration,dependency,invalid,affected):
    hand_frame.loc[0,dependency]=invalid
    result=api().build_scale_free(hand_frame,['2021-01-04'],phase3_config,validation_policy,scale_registration)
    assert result[affected].isna().all().all()
    assert not np.isinf(result[scale_registration['shared_whitelist']].to_numpy()).any()


@pytest.mark.parametrize('boundary',['action','calendar_gap'])
def test_ratios_restart_all_required_history_after_boundary(make_panel,phase3_config,validation_policy,scale_registration,boundary):
    panel,cal=make_panel(days=145,event_at=70 if boundary=='action' else None)
    if boundary=='calendar_gap': panel=panel.loc[panel.date!=cal[69]]
    before=panel.copy(deep=True)
    full=api().build_scale_free(from_panel(panel,phase3_config,cal),cal,phase3_config,validation_policy,scale_registration)
    fresh=api().build_scale_free(from_panel(panel.loc[panel.date>=cal[70]],phase3_config,cal),cal,phase3_config,validation_policy,scale_registration)
    pd.testing.assert_frame_equal(full.loc[full.date>=cal[70]].reset_index(drop=True),fresh,check_exact=True)
    one=fresh.loc[fresh.asset_code=='510300'].reset_index(drop=True)
    for name,warmup in zip(RATIOS,[4,19,59,19,59,25,33,33]):
        assert one[name].iloc[:warmup].isna().all()
        assert pd.notna(one[name].iloc[warmup])
    pd.testing.assert_frame_equal(panel,before)


@pytest.mark.parametrize('cut',[20,69,110])
def test_future_prices_and_truncation_cannot_change_past_ratios(make_panel,phase3_config,validation_policy,scale_registration,cut):
    panel,cal=make_panel(days=145,event_at=70)
    def build(rows):
        return api().build_scale_free(from_panel(rows,phase3_config,cal),cal,phase3_config,validation_policy,scale_registration)
    before=build(panel);changed=panel.copy()
    changed.loc[changed.date>cal[cut],['close','research_close']]*=7
    changed.loc[changed.date>cal[cut],['volume','research_volume']]*=13
    after=build(changed);keep=before.date<=cal[cut]
    pd.testing.assert_frame_equal(before.loc[keep],after.loc[keep],check_exact=True)
    pd.testing.assert_frame_equal(before.loc[keep].reset_index(drop=True),build(panel.loc[panel.date<=cal[cut]]),check_exact=True)


def test_other_asset_and_row_order_do_not_contaminate_ratios(make_panel,phase3_config,validation_policy,scale_registration):
    panel,cal=make_panel(days=145,event_at=70)
    before=api().build_scale_free(from_panel(panel,phase3_config,cal),cal,phase3_config,validation_policy,scale_registration)
    changed=panel.copy();changed.loc[changed.asset_code=='510500',['close','research_close']]*=20
    after=api().build_scale_free(from_panel(changed,phase3_config,cal).iloc[::-1],cal,phase3_config,validation_policy,scale_registration)
    pd.testing.assert_frame_equal(before.loc[before.asset_code=='510300'],after.loc[after.asset_code=='510300'],check_exact=True)
    # Positive rescaling within each asset must not change any of the 17 inputs.
    pd.testing.assert_frame_equal(before,after,check_exact=False,rtol=1e-10,atol=1e-10)


def test_invalid_volume_propagates_without_changing_price_ratios(make_panel,phase3_config,validation_policy,scale_registration):
    panel,cal=make_panel()
    before=api().build_scale_free(from_panel(panel,phase3_config,cal),cal,phase3_config,validation_policy,scale_registration)
    bad=(panel.asset_code=='510300')&(panel.date==cal[70])
    panel.loc[bad,'volume_valid']=False;panel.loc[bad,'research_volume']=np.nan
    after=api().build_scale_free(from_panel(panel,phase3_config,cal),cal,phase3_config,validation_policy,scale_registration)
    one=after.loc[after.asset_code=='510300'].reset_index(drop=True)
    assert one.volume_change_1d.iloc[70:72].isna().all() and one.volume_ratio_20d.iloc[70:90].isna().all()
    pd.testing.assert_frame_equal(before[RATIOS],after[RATIOS],check_exact=True)


@pytest.mark.parametrize('fault',['holdout','holdout_calendar','duplicate','null_asset','unknown_asset','bad_segment','inf'])
def test_invalid_feature_keys_and_holdout_are_rejected(hand_frame,phase3_config,validation_policy,scale_registration,fault):
    cal=['2021-01-04']
    if fault=='holdout': hand_frame.loc[0,'date']='2026-01-05'
    if fault=='holdout_calendar': cal.append('2026-01-05')
    if fault=='duplicate': hand_frame=pd.concat([hand_frame,hand_frame],ignore_index=True)
    if fault=='null_asset': hand_frame.loc[0,'asset_code']=None
    if fault=='unknown_asset': hand_frame.loc[0,'asset_code']='999999'
    if fault=='bad_segment': hand_frame.loc[0,'price_segment']=-1
    if fault=='inf': hand_frame.loc[0,'macd_line']=np.inf
    with pytest.raises(ValueError): api().build_scale_free(hand_frame,cal,phase3_config,validation_policy,scale_registration)

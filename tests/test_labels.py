"""Targets use exchange-session dates, never row-count shortcuts or missing=down."""
import copy
import importlib
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

ROOT=Path(__file__).resolve().parents[1]


def api():
    assert (ROOT/'src/labels/forward_return.py').exists(), 'Label implementation missing'
    return importlib.import_module('src.labels.forward_return')


def test_exact_five_session_return_and_missing_tail_are_not_class_zero(make_panel,phase3_config):
    panel,calendar=make_panel(days=20)
    result=api().build_labels(panel,phase3_config,calendar)
    for code,group in panel.groupby('asset_code'):
        out=result.loc[result.asset_code==code].reset_index(drop=True)
        prices=group.close.to_numpy()
        np.testing.assert_allclose(out.future_return_5d.iloc[:-5],prices[5:]/prices[:-5]-1)
        pd.testing.assert_series_equal(out.future_return_5d,out.regression_target,check_names=False)
        assert out.label_valid.iloc[:-5].all() and not out.label_valid.iloc[-5:].any()
        assert out.classification_target.iloc[-5:].isna().all()
        assert str(out.classification_target.dtype)=='Int8'
        assert out.label_end_date.iloc[0]==calendar[5]


def test_zero_return_is_valid_classification_zero(make_panel,phase3_config):
    panel,calendar=make_panel(days=15)
    panel[['close','research_close']]=10.
    out=api().build_labels(panel,phase3_config,calendar)
    assert out.loc[out.label_valid,'classification_target'].eq(0).all()
    assert out.loc[~out.label_valid,'classification_target'].isna().all()


def test_label_never_crosses_corporate_action_and_has_no_cross_asset_pollution(make_panel,phase3_config):
    panel,calendar=make_panel(days=25,event_at=12)
    out=api().build_labels(panel,phase3_config,calendar)
    event_window=out.date.isin(calendar[7:12])
    assert out.loc[event_window,'future_return_5d'].isna().all()
    assert out.loc[event_window,'label_invalid_reason'].eq('corporate_action_boundary').all()
    assert out.loc[out.date==calendar[12],'label_valid'].all()
    changed=panel.copy();other=changed.asset_code=='510500'
    changed.loc[other,['close','research_close']]*=100
    after=api().build_labels(changed.iloc[::-1],phase3_config,calendar)
    pd.testing.assert_frame_equal(out.loc[out.asset_code=='510300'],after.loc[after.asset_code=='510300'],check_exact=True)


def test_suspension_target_is_missing_not_shifted_to_next_quote(make_panel,phase3_config):
    panel,calendar=make_panel(days=20)
    panel=panel.loc[~((panel.asset_code=='510300')&(panel.date==calendar[5]))]
    out=api().build_labels(panel,phase3_config,calendar)
    first=out.loc[(out.asset_code=='510300')&(out.date==calendar[0])].iloc[0]
    assert first.label_end_date==calendar[5]
    assert first.label_invalid_reason=='target_quote_missing'
    assert pd.isna(first.classification_target)
    # A suspension inside the interval does not change an otherwise defined endpoint return.
    next_row=out.loc[(out.asset_code=='510300')&(out.date==calendar[1])].iloc[0]
    prices=panel.loc[panel.asset_code=='510300'].set_index('date').close
    assert next_row.future_return_5d==pytest.approx(prices[calendar[6]]/prices[calendar[1]]-1)


def test_configured_horizon_is_used_without_changing_rebalance_frequency(make_panel,phase3_config):
    panel,calendar=make_panel(days=20)
    c=copy.deepcopy(phase3_config);c['timing']['prediction_horizon_days']=3
    out=api().build_labels(panel,c,calendar)
    assert 'future_return_3d' in out and 'future_return_5d' not in out
    assert out.iloc[0].label_end_date==calendar[3]
    assert c['timing']['rebalance_frequency_days']==5


def test_labels_reject_holdout_even_when_caller_supplies_it(make_panel,phase3_config):
    panel,calendar=make_panel(days=20)
    panel.loc[0,'date']='2026-01-02'
    with pytest.raises(ValueError): api().build_labels(panel,phase3_config,calendar)

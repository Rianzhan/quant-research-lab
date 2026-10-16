"""Independent hand-checked boundaries, global dates, availability and statistics."""
import copy
import importlib
import math
import numpy as np
import pandas as pd
import pytest
from phase4_helpers import ROOT, validation_policy


def api(name='walk_forward'):
    assert (ROOT/f'src/validation/{name}.py').exists(), 'Phase4 split/audit missing'
    return importlib.import_module('src.validation.'+name)


@pytest.fixture
def split_input(phase3_config,validation_policy):
    c=phase3_config;p=copy.deepcopy(validation_policy)
    p['split']['test_windows']=[{'id':'test_2022','start':'2022-01-01','end':'2022-01-31'},
                              {'id':'test_2023','start':'2023-01-01','end':'2023-01-20'}]
    p['eligibility'].update(min_train_samples_per_asset=5,min_test_samples_per_asset=1)
    cal=pd.bdate_range('2021-12-20','2023-02-03').strftime('%Y-%m-%d').tolist()
    records=[]
    for asset in c['universe']['assets'][:2]:
        for i,day in enumerate(cal[:-5]):
            target=.02 if i%3==0 else -.01
            records.append({'date':day,'exchange':asset['exchange'],'asset_code':asset['code'],
                'price_segment':0,'research_close':10.,
                **{name:1. for name in p['feature_registration']['shared_raw_whitelist']},
                'label_valid':True,'label_end_date':cal[i+5],
                'regression_target':target,'classification_target':int(target>0)})
    frame=pd.DataFrame(records)
    # Irregular observed histories; the generic splitter uses the supplied audited
    # label endpoints, not row shifts. Full real-label semantics are the loader's job.
    frame=frame.loc[~((frame.asset_code=='510500')&frame.date.isin(['2021-12-29','2022-01-04']))]
    return frame.reset_index(drop=True),cal,c,p


def test_global_purge_uses_real_endpoints_including_equality(split_input):
    f,cal,c,p=split_input; m=api().build_membership(f,cal,c,p)
    first=m.loc[m.fold_id=='test_2022']
    assert first.partition.value_counts().to_dict()=={'test':41,'train':10,'purged':9}
    assert set(first.loc[first.partition=='purged','date'])=={'2021-12-27','2021-12-28','2021-12-29','2021-12-30','2021-12-31'}
    assert first.loc[first.partition=='train','label_end_date'].max()=='2021-12-31'
    assert (first.loc[first.partition=='train','date']<'2022-01-03').all()
    assert len(first.loc[(first.date=='2021-12-27')&(first.partition=='purged')])==2
    # A real endpoint can invalidate an earlier row too; deleting last five rows cannot.
    f.loc[(f.asset_code=='510300')&(f.date=='2021-12-20'),'label_end_date']='2022-01-03'
    changed=api().build_membership(f,cal,c,p)
    assert len(changed.loc[(changed.fold_id=='test_2022')&(changed.partition=='purged')])==10


def test_pooled_and_each_asset_share_exact_test_calendar_and_rows(split_input):
    f,cal,c,p=split_input; m=api().build_membership(f,cal,c,p)
    pooled=api().select_samples(f,m,'test_2022','test')
    separate=pd.concat([api().select_samples(f,m,'test_2022','test',('SSE',code)) for code in ['510300','510500']])
    keys=['date','exchange','asset_code']
    pd.testing.assert_frame_equal(pooled[keys],separate[keys].sort_values(keys).reset_index(drop=True))
    assert len(pooled)==41 and pooled.date.max()=='2022-01-31'
    train=api().select_samples(f,m,'test_2022','train')
    assert not train.date.str.startswith(('2022','2023')).any()


def test_shuffled_input_and_repeated_build_are_identical(split_input):
    f,cal,c,p=split_input; first=api().build_membership(f,cal,c,p)
    other=api().build_membership(f.sample(frac=1,random_state=987),cal,c,p)
    pd.testing.assert_frame_equal(first,other,check_exact=True)
    pd.testing.assert_frame_equal(first,api().build_membership(f,cal,c,p),check_exact=True)


def test_insufficient_asset_is_explicit_and_excluded_from_both_scopes(split_input):
    f,cal,c,p=split_input
    f=f.loc[~((f.asset_code=='510500')&(f.date=='2021-12-20'))]
    m=api().build_membership(f,cal,c,p)
    excluded=m.loc[(m.fold_id=='test_2022')&(m.asset_code=='510500')]
    assert not excluded.included.any() and excluded.asset_status.eq('insufficient').all()
    assert excluded.exclusion_reason.str.contains('train_samples_4_below_5').all()
    assert set(api().select_samples(f,m,'test_2022','test').asset_code)=={'510300'}
    with pytest.raises(ValueError,match='unavailable'):
        api().select_samples(f,m,'test_2022','test',('SSE','510500'))
    audit=api('audit').audit_folds(f,m,cal,c,p)
    absent=next(x for x in audit['assets'] if x['fold_id']=='test_2022' and x['asset_code']=='588000')
    assert absent['status']=='unavailable' and absent['train_after_purge']==0 and absent['test_samples']==0


def test_test_label_distribution_cannot_change_split_or_eligibility(split_input):
    f,cal,c,p=split_input; baseline=api().build_membership(f,cal,c,p)
    f.loc[f.date>='2022-01-01','regression_target']=.75
    f.loc[f.date>='2022-01-01','classification_target']=1
    pd.testing.assert_frame_equal(baseline,api().build_membership(f,cal,c,p))


def test_future_test_availability_never_changes_historical_training_membership(split_input):
    """Removing future quotes must not select a different historical training pool."""
    f,cal,c,p=split_input
    baseline=api().build_membership(f,cal,c,p)
    future_removed=f.loc[~((f.asset_code=='510500')&(f.date>='2022-01-01'))]
    altered=api().build_membership(future_removed,cal,c,p)
    before=api().select_samples(f,baseline,'test_2022','train')
    after=api().select_samples(future_removed,altered,'test_2022','train')
    pd.testing.assert_frame_equal(before,after,check_exact=True)
    audit=api('audit').audit_folds(future_removed,altered,cal,c,p)['folds'][0]
    assert audit['train_after_purge']==10 and audit['test_samples']==21
    assert audit['training_assets']==['SSE:510300','SSE:510500']
    assert audit['evaluation_assets']==['SSE:510300']


def test_unknown_fold_raises_but_registered_empty_fold_remains_valid(split_input):
    f,cal,c,p=split_input; m=api().build_membership(f,cal,c,p)
    with pytest.raises(ValueError,match='Unknown fold'):
        api().select_samples(f,m,'typo','test')
    names=p['feature_registration']['shared_raw_whitelist']
    f.loc[:,names]=np.nan
    empty=api().build_membership(f,cal,c,p)
    assert api().select_samples(f,empty,'test_2022','test').empty
    # A serialized empty table needs the explicit registered fold IDs restored.
    empty.attrs.clear()
    assert api().select_samples(f,empty,'test_2022','test',registered_folds=['test_2022','test_2023']).empty


@pytest.mark.parametrize('fault',['holdout_row','holdout_endpoint','holdout_calendar','null_asset','unknown_asset','duplicate','early_endpoint','missing_valid_label','inf_feature'])
def test_bad_rows_cannot_enter_any_fold(split_input,fault):
    f,cal,c,p=split_input
    if fault=='holdout_row': f.loc[0,'date']='2026-01-05'
    if fault=='holdout_endpoint': f.loc[0,'label_end_date']='2026-01-05'
    if fault=='holdout_calendar': cal=cal+['2026-01-05']
    if fault=='null_asset': f.loc[0,'asset_code']=None
    if fault=='unknown_asset': f.loc[0,'asset_code']='999999'
    if fault=='duplicate': f=pd.concat([f,f.iloc[:1]],ignore_index=True)
    if fault=='early_endpoint': f.loc[0,'label_end_date']='2021-12-20'
    if fault=='missing_valid_label': f.loc[0,'regression_target']=np.nan
    if fault=='inf_feature': f.loc[0,'return_1d']=np.inf
    with pytest.raises(ValueError): api().build_membership(f,cal,c,p)


def test_missing_features_are_not_imputed_into_fold(split_input):
    f,cal,c,p=split_input;f.loc[(f.asset_code=='510300')&(f.date=='2022-01-05'),'volume_ratio_20d']=np.nan
    m=api().build_membership(f,cal,c,p)
    assert not ((m.fold_id=='test_2022')&(m.asset_code=='510300')&(m.date=='2022-01-05')).any()


def test_audit_statistics_match_hand_values_and_empty_is_undefined(validation_policy):
    f=pd.DataFrame({'regression_target':[-.1,0.,.2],'classification_target':[0,0,1]})
    result=api('audit').target_statistics(f,validation_policy)
    assert result['positive_count']==1 and result['negative_count']==2
    assert result['positive_ratio']==pytest.approx(1/3)
    assert result['regression_mean']==pytest.approx(1/30) and result['regression_median']==0
    assert result['regression_std']==pytest.approx(math.sqrt(7/300))
    assert result['q05']==pytest.approx(-.09) and result['q95']==pytest.approx(.18)
    assert api('audit').target_statistics(f.iloc[:0],validation_policy)['positive_ratio'] is None
    assert api('audit').target_statistics(f.iloc[:1],validation_policy)['regression_std'] is None


def test_audit_counts_reconcile_and_empty_participation_is_explicit(split_input):
    f,cal,c,p=split_input;m=api().build_membership(f,cal,c,p)
    audit=api('audit').audit_folds(f,m,cal,c,p)
    row=audit['folds'][0]
    assert (row['train_before_purge'],row['train_after_purge'],row['purged_samples'],row['test_samples'])==(19,10,9,41)
    assert row['train_actual_start']=='2021-12-20' and row['train_actual_end']=='2021-12-24'
    assert row['train_label_end_max']=='2021-12-31' and row['test_start_date']=='2022-01-03'
    p['eligibility']['min_train_samples_per_asset']=10000
    m=api().build_membership(f,cal,c,p);empty=api('audit').audit_folds(f,m,cal,c,p)['folds'][0]
    assert empty['status']=='unavailable' and empty['train_after_purge']==0 and empty['test_samples']==0


def test_full_purged_kfold_is_explicitly_not_implemented():
    with pytest.raises(NotImplementedError): api('purged_split').PurgedKFold().split(None)

"""Static date-context copies; all formulas and event/gap reset rules unchanged."""
import numpy as np
import pandas as pd
from src.data.cleaner import date_strings
from src.data.research_policy import require
from src.utils.config import validate_config
from src.utils.research_rows import ASSET, KEYS
from src.features.builder import feature_specs
from src.features.scale_free import validate_registration
from src.features.returns import historical_returns
from src.features.momentum import momentum
from src.features.volatility import historical_volatility
from src.features.volume import volume_features
from src.features.technical import moving_averages, wilder_rsi, macd
from src.final_holdout.training import FEATURES

EXPECTED_FEATURES = {'return_windows':[1,5,20],'momentum_windows':[5,20,60],
    'moving_average_windows':[5,20,60],'moving_average_diff_pairs':[[5,20],[20,60]],
    'volatility_windows':[5,20],'volatility_ddof':1,'volume_change_periods':1,
    'volume_average_window':20,'rsi_window':14,'macd_fast':12,'macd_slow':26,'macd_signal':9}

def prepare_rows(panel: pd.DataFrame, config: dict, calendar: list[str]) -> pd.DataFrame:
    """Validate and canonically sort observed rows; mark causal calculation runs.

    An exchange-session gap starts a new calculation run, without creating any
    missing price row. Positions are audit metadata and never model features.
    """
    validate_config(config)
    calendar = list(calendar)  # Validated by build_context_features before dispatch.
    required = {*KEYS, 'asset_name', 'category', 'close', 'research_close', 'volume',
                'research_volume', 'volume_valid', 'price_basis', 'research_price_policy'}
    require(not panel.empty and panel.columns.is_unique and required <= set(panel), 'Missing research columns/rows')
    frame = panel.copy(deep=True)
    # pandas groupby drops null keys by default; reject them before any grouping.
    require(frame[ASSET].notna().all().all(), 'Missing asset identity')
    frame['date'] = date_strings(frame.date)
    # Exact context membership is enforced below; old config remains unmodified.
    require(frame.date.isin(calendar).all(), 'Quote date is not in supplied domestic calendar')
    require(not frame.duplicated(['date', *ASSET]).any(), 'Duplicate asset date')
    require(frame.price_basis.eq('none').all() and frame.research_price_policy.eq('unadjusted_event_segmented_v1').all(),
            'Unapproved research price basis')
    for col in ['close', 'research_close', 'volume', 'research_volume', 'price_segment']:
        frame[col] = pd.to_numeric(frame[col], errors='raise')
    require(np.isfinite(frame[['close','research_close','volume','price_segment']]).all().all(), 'Nonfinite required field')
    require(frame.close.gt(0).all() and frame.research_close.eq(frame.close).all(), 'Research price must equal positive raw close')
    require(frame.volume.ge(0).all(), 'Negative raw volume')
    require(frame.price_segment.ge(0).all() and frame.price_segment.eq(np.floor(frame.price_segment)).all(), 'Invalid price segment')
    require(frame.volume_valid.map(lambda v: isinstance(v,(bool,np.bool_))).all(), 'volume_valid must be Boolean')
    frame['price_segment'] = frame.price_segment.astype('int64')
    valid_volume = frame.volume_valid & np.isfinite(frame.research_volume) & frame.research_volume.gt(0)
    require(frame.loc[valid_volume,'research_volume'].eq(frame.loc[valid_volume,'volume']).all(), 'Research volume was silently replaced')
    frame['_effective_volume'] = frame.research_volume.where(valid_volume, np.nan)
    frame = frame.sort_values(['date', *ASSET], kind='stable').reset_index(drop=True)
    expected = {(a['exchange'],a['code']): a for a in config['universe']['assets']}
    positions = dict(zip(calendar, range(len(calendar))))
    for name in ['_asset_position', '_segment_position', '_run_id', '_run_position']:
        frame[name] = 0
    for identity, group in frame.groupby(ASSET, sort=False):
        require(identity in expected, 'Unknown research asset')
        asset = expected[identity]
        require(group.asset_name.eq(asset['name']).all() and group.category.eq(asset['category']).all(), 'Asset metadata conflict')
        require(group.date.ge(asset['data_available_from']).all(), 'Prelisting/pre-observation values')
        require(group.price_segment.is_monotonic_increasing, 'Corporate-action segments must not reverse or recur')
        changed = group.price_segment.diff().ne(0) | group.date.map(positions).diff().ne(1)
        run = changed.cumsum()-1
        frame.loc[group.index, '_asset_position'] = np.arange(len(group))
        frame.loc[group.index, '_segment_position'] = group.groupby('price_segment',sort=False).cumcount()
        frame.loc[group.index, '_run_id'] = run
        frame.loc[group.index, '_run_position'] = group.groupby(run,sort=False).cumcount()
    return frame

def build_features(panel: pd.DataFrame, config: dict, calendar: list[str]) -> pd.DataFrame:
    """Build only explicitly listed predictors; asset/action/gap state never leaks."""
    rows=prepare_rows(panel,config,calendar); f=config['features']
    names=[s['name'] for s in feature_specs(config)]
    result=rows[KEYS].copy()
    for name in names:
        result[name]=np.nan
    for _,group in rows.groupby([*ASSET,'price_segment','_run_id'],sort=False):
        close=group.research_close
        parts=[historical_returns(close,f['return_windows']),momentum(close,f['momentum_windows']),
               moving_averages(close,f['moving_average_windows'],f['moving_average_diff_pairs']),
               historical_volatility(close,f['volatility_windows'],f['volatility_ddof']),
               volume_features(group._effective_volume,f['volume_change_periods'],f['volume_average_window']),
               wilder_rsi(close,f['rsi_window']).to_frame(f'rsi_{f["rsi_window"]}'),
               macd(close,f['macd_fast'],f['macd_slow'],f['macd_signal'])]
        calculated=pd.concat(parts,axis=1)
        result.loc[group.index,names]=calculated[names]
    require(not np.isinf(result[names].to_numpy()).any(),'Nonfinite feature overflow')
    return result

def build_scale_free(frame: pd.DataFrame, calendar: list[str], config: dict,
                     validation: dict, registration: dict) -> pd.DataFrame:
    """Transform same-row dependencies, preserving all keys and dependency NaNs.

    Production callers must use the pinned-input adapter. Rolling/recursive
    warmups live in the accepted Phase3 columns; no rolling state, fill or
    target is used here. Synthetic tests rebuild those columns before perturbing.
    """
    validate_registration(registration,validation,config)
    spec=validation['feature_registration'];raw=spec['shared_raw_whitelist']
    require(not frame.empty and frame.columns.is_unique and set(KEYS+raw+['research_close'])<=set(frame),
            'Missing scale-free dependency rows/columns')
    rows=frame[KEYS+raw+['research_close']].copy(deep=True)
    require(rows[ASSET].notna().all().all(),'Missing asset identity')
    rows['date']=date_strings(rows.date)
    require(rows.date.isin(calendar).all(),'Holdout/out-of-development observations are forbidden')
    require(not rows.duplicated(['date',*ASSET]).any(),'Duplicate asset date')
    for name in [*raw,'research_close','price_segment']:
        rows[name]=pd.to_numeric(rows[name],errors='raise')
    require(not np.isinf(rows[[*raw,'research_close']].to_numpy(dtype=float,na_value=np.nan)).any(),'Nonfinite feature dependency')
    require(np.isfinite(rows.price_segment).all() and rows.price_segment.ge(0).all()
            and rows.price_segment.eq(np.floor(rows.price_segment)).all(),'Invalid segment identity')
    rows=rows.sort_values(KEYS,kind='stable').reset_index(drop=True)
    assets={(a['exchange'],a['code']):a for a in config['universe']['assets']}
    for identity,group in rows.groupby(ASSET,sort=False):
        require(identity in assets,'Unknown research asset')
        require(group.date.ge(assets[identity]['data_available_from']).all(),'Pre-observation values')
        require(group.price_segment.is_monotonic_increasing,'Corporate-action segments must not reverse or recur')
    result=rows[KEYS].copy()
    replacements=spec['planned_replacements']
    for old in raw:
        if old not in replacements:
            result[old]=rows[old]
            continue
        item=replacements[old];numerator=rows[item['numerator']];denominator=rows[item['denominator']]
        if item['numerator'] in ['research_close','ma_5d','ma_20d','ma_60d']:
            numerator=numerator.where(numerator.gt(0))
        denominator=denominator.where(denominator.gt(0))
        with np.errstate(over='ignore',divide='ignore',invalid='ignore'):
            result[item['name']]=numerator/denominator-item['subtract']
    names=registration['shared_whitelist']
    require(not np.isinf(result[names].to_numpy(dtype=float,na_value=np.nan)).any(),'Scale-free ratio overflow')
    return result[KEYS+names]

def build_context_features(panel: pd.DataFrame, config: dict, calendar: list[str],
        validation: dict, registration: dict, *, synthetic: bool=False) -> tuple[pd.DataFrame,pd.DataFrame]:
    """Causal same-asset/segment/gap features; future in-memory fixtures only.

    Pre2026 observations are retained for warmup. There is no label argument,
    forward fill, date shift, synthetic quote generation, or payload reader.
    """
    require(type(synthetic) is bool,'Synthetic flag must be Boolean')
    require(config['features']==EXPECTED_FEATURES,'Frozen feature parameters changed')
    require(tuple(registration['shared_whitelist'])==FEATURES,'Changed feature order')
    dates=date_strings(pd.Series(calendar,dtype=str))
    require(len(dates)>0 and dates.is_unique and dates.is_monotonic_increasing,'Invalid explicit calendar')
    require(dates.between('2018-01-01','2026-09-30' if synthetic else '2025-12-31').all(),'Unauthorized future context')
    expected=[('ma_5d','close_over_ma_5d','research_close','ma_5d',1),
        ('ma_20d','close_over_ma_20d','research_close','ma_20d',1),
        ('ma_60d','close_over_ma_60d','research_close','ma_60d',1),
        ('ma_diff_5_20','ma_ratio_5_20','ma_5d','ma_20d',1),
        ('ma_diff_20_60','ma_ratio_20_60','ma_20d','ma_60d',1),
        ('macd_line','macd_line_over_close','macd_line','research_close',0),
        ('macd_signal','macd_signal_over_close','macd_signal','research_close',0),
        ('macd_hist','macd_hist_over_close','macd_hist','research_close',0)]
    require(validation['feature_registration']['planned_replacements']=={old:dict(name=n,numerator=a,denominator=b,subtract=s)
        for old,n,a,b,s in expected},'Frozen ratio formulas changed')
    raw=build_features(panel,config,dates.tolist())
    frame=raw.merge(panel[KEYS+['research_close']],on=KEYS,validate='one_to_one')
    ratios=build_scale_free(frame,dates.tolist(),config,validation,registration)
    return raw,ratios

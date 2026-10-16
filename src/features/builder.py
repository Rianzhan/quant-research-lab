"""Pure, development-only feature construction; no targets, fit or time split."""
import numpy as np
import pandas as pd
from src.data.research_policy import require
from src.features.returns import historical_returns
from src.features.momentum import momentum
from src.features.volatility import historical_volatility
from src.features.volume import volume_features
from src.features.technical import moving_averages, wilder_rsi, macd
from src.utils.research_rows import ASSET, KEYS, prepare_rows


def feature_specs(config: dict) -> list[dict]:
    """Ordered predictor whitelist and dependency lengths for missingness audits."""
    f=config['features']; result=[]
    def add(name, lookback, formula, meaning, volume=False, duplicate=None):
        result.append({'name':name,'lookback':lookback,'uses_volume':volume,
                       'formula':formula,'financial_meaning':meaning,'duplicate_of':duplicate})
    for n in f['return_windows']:
        add(f'return_{n}d',n,f'P[t]/P[t-{n}]-1','过去固定交易日价格变化比例')
    for n in f['momentum_windows']:
        add(f'momentum_{n}d',n,f'P[t]/P[t-{n}]-1','过去价格趋势的方向与幅度',duplicate=f'return_{n}d' if n in f['return_windows'] else None)
    for n in f['moving_average_windows']:
        add(f'ma_{n}d',n-1,f'mean(P[t-{n-1}:t])','近期平均价格水平，单位为元/份')
    for a,b in f['moving_average_diff_pairs']:
        add(f'ma_diff_{a}_{b}',b-1,f'MA({a})-MA({b})','短期价格水平相对长期的偏离，单位为元/份')
    for n in f['volatility_windows']:
        add(f'volatility_{n}d',n,f'std(last {n} one-session simple returns, ddof={f["volatility_ddof"]})','近期日收益波动，非年化')
    p,n=f['volume_change_periods'],f['volume_average_window']
    add(f'volume_change_{p}d',p,f'V[t]/V[t-{p}]-1; all dependency volumes valid','交易量相对前期变化',True)
    add(f'volume_ratio_{n}d',n-1,f'V[t]/mean(last {n} valid V including t)','当前交易活跃度相对近期平均',True)
    r=f['rsi_window'];slow=f['macd_slow'];sig=f['macd_signal']
    add(f'rsi_{r}',r,f'100*WilderGain({r})/(WilderGain({r})+WilderLoss({r})); flat=50','近期上涨与下跌幅度的相对强弱')
    add('macd_line',slow-1,f'EMA({f["macd_fast"]})-EMA({slow}); arithmetic seeds','快慢趋势均值之差，价格单位')
    add('macd_signal',slow+sig-2,f'EMA({sig}, MACD line); arithmetic seed','趋势差的平滑参考线')
    add('macd_hist',slow+sig-2,'MACD line - signal; multiplier=1','趋势差偏离参考线的程度')
    return result


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

"""NAV-only performance and explicitly censored drawdown episodes."""
from datetime import date
from decimal import Decimal,localcontext
import math
import numpy as np
import pandas as pd
from .common import decimal,require

EPISODE_COLUMNS=['episode','peak_date','peak_observation','peak_nav','trough_date','trough_nav',
    'maximum_drawdown','recovery_date','end_date','recovery_status','duration_sessions',
    'duration_calendar_days','underwater_sessions','peak_to_trough_sessions']

def _episodes(dates: list[str],values: list[Decimal],initial: Decimal,base_date: str) -> tuple[list[float],pd.DataFrame]:
    """Latest equal high, earliest equal low, first recovery; never infer beyond the end."""
    peak=initial;peak_i=-1;peak_date=base_date;current=None;trough=initial;rows=[];curve=[]
    def finish(end_i: int,recovered: bool) -> None:
        assert current is not None
        row={**current,'episode':len(rows)+1,'recovery_date':dates[end_i] if recovered else None,
            'end_date':dates[end_i],'recovery_status':'recovered' if recovered else 'unrecovered',
            'duration_sessions':end_i-peak_i,
            'duration_calendar_days':(date.fromisoformat(dates[end_i])-date.fromisoformat(peak_date)).days}
        rows.append(row)
    for i,value in enumerate(values):
        if value>=peak:
            if current is not None:finish(i,True);current=None
            peak=value;peak_i=i;peak_date=dates[i];curve.append(0.0)
            continue
        depth=float(1-value/peak);curve.append(-depth)
        if current is None:
            trough=value
            current={'peak_date':peak_date,'peak_observation':'initial_capital' if peak_i==-1 else 'session_close',
                'peak_nav':float(peak),'trough_date':dates[i],'trough_nav':float(value),'maximum_drawdown':depth,
                'underwater_sessions':0,'peak_to_trough_sessions':i-peak_i}
        if value<trough:
            trough=value
            current.update(trough_date=dates[i],trough_nav=float(value),maximum_drawdown=depth,peak_to_trough_sessions=i-peak_i)
        current['underwater_sessions']+=1
    if current is not None:finish(len(dates)-1,False)
    return curve,pd.DataFrame(rows,columns=EPISODE_COLUMNS)

def nav_metrics(dates: list[str],nav: list,initial: object,base_date: str | None=None) -> tuple[dict,pd.DataFrame,pd.DataFrame]:
    """252-session CAGR/vol/Sharpe; all returns include their frozen starting NAV."""
    require(len(dates)==len(nav)>0,'Nonempty aligned NAV series required')
    pairs=sorted(zip(dates,nav));dates=[x[0] for x in pairs]
    require(len(set(dates))==len(dates),'Duplicate NAV date')
    require(all(date.fromisoformat(d).isoformat()==d and '2018-01-01'<=d<='2025-12-31' for d in dates),'Non-development NAV date')
    base_date=base_date or dates[0];require(date.fromisoformat(base_date).isoformat()==base_date and base_date<=dates[0],'Invalid initial observation date')
    with localcontext() as ctx:
        ctx.prec=40;initial=decimal(initial);values=[decimal(x[1]) for x in pairs]
        require(initial>0 and all(v>0 for v in values),'NAV must be positive')
        previous=[initial]+values[:-1];exact_returns=[v/p-1 for v,p in zip(values,previous)]
        returns=np.array([float(r) for r in exact_returns])
        require(np.isfinite(returns).all(),'Nonfinite NAV returns')
        total=float(values[-1]/initial-1);cagr=float(values[-1]/initial)**(252/len(values))-1
        dd,episodes=_episodes(dates,values,initial,base_date)
        curve=pd.DataFrame({'date':dates,'nav':[float(v) for v in values],'previous_nav':[float(v) for v in previous],
            'daily_return':returns,'cumulative_nav':[float(v/initial) for v in values],'drawdown':dd})
    constant=all(r==exact_returns[0] for r in exact_returns)
    std=(0.0 if constant else float(returns.std(ddof=1))) if len(returns)>1 else None
    vol=std*math.sqrt(252) if std is not None else None
    sharpe=float(returns.mean())/std*math.sqrt(252) if std is not None and std>0 else None
    undefined='insufficient_sessions' if std is None else 'zero_volatility' if std==0 else ''
    result={'start_date':dates[0],'end_date':dates[-1],'sessions':len(dates),'initial_nav':float(initial),
        'terminal_nav':float(values[-1]),'total_return':total,'cagr':cagr,'annualized_volatility':vol,
        'sharpe_ratio':sharpe,'maximum_drawdown':0.0,'drawdown_peak_date':None,'drawdown_trough_date':None,
        'recovery_date':None,'recovery_status':'no_drawdown','drawdown_duration_sessions':0,
        'drawdown_duration_calendar_days':0,'longest_drawdown_duration_sessions':0,
        'longest_drawdown_duration_calendar_days':0,'longest_drawdown_peak_date':None,
        'longest_drawdown_end_date':None,'longest_drawdown_recovery_status':'no_drawdown','undefined_reason':undefined}
    if not episodes.empty:
        worst=episodes.loc[episodes.maximum_drawdown.idxmax()];longest=episodes.loc[episodes.duration_sessions.idxmax()]
        result.update(maximum_drawdown=float(worst.maximum_drawdown),drawdown_peak_date=worst.peak_date,
            drawdown_trough_date=worst.trough_date,recovery_date=worst.recovery_date,recovery_status=worst.recovery_status,
            drawdown_duration_sessions=int(worst.duration_sessions),drawdown_duration_calendar_days=int(worst.duration_calendar_days),
            longest_drawdown_duration_sessions=int(longest.duration_sessions),longest_drawdown_duration_calendar_days=int(longest.duration_calendar_days),
            longest_drawdown_peak_date=longest.peak_date,longest_drawdown_end_date=longest.end_date,
            longest_drawdown_recovery_status=longest.recovery_status)
    return result,curve,episodes

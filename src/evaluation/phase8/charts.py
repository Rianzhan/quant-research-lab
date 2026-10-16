"""Deterministic descriptive charts; plots consume evaluation tables, not market files."""
from io import BytesIO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
from .common import STRATEGIES,FEES,PRIMARY

COLORS={'A':'#2864a5','B':'#d06b28','C':'#2d865d'}
FOOTER='Frozen development accounts 2022-2025 | no optimization | 2026 remains sealed'

def _save(fig,name: str,out: dict) -> None:
    fig.tight_layout(rect=(0,.035,1,.98));fig.text(.5,.012,FOOTER,ha='center',fontsize=8,color='#555555')
    buffer=BytesIO();fig.savefig(buffer,format='png',dpi=130,metadata={'Software':'quant-lab Phase8'})
    out['charts/'+name+'.png']=buffer.getvalue();plt.close(fig)

def _percent(ax) -> None:
    ax.yaxis.set_major_formatter(PercentFormatter(1))

def _grouped(frame,key: str,ax,title: str,percent: bool=True) -> None:
    years=['2022','2023','2024','2025'];x=np.arange(4)
    for i,s in enumerate(STRATEGIES):
        values=frame[frame.strategy==s].set_index('year').loc[years,key]
        ax.bar(x+(i-1)*.25,values,.24,label=s,color=COLORS[s])
    ax.set_xticks(x,years);ax.set_title(title);ax.axhline(0,color='#555',lw=.6);ax.legend()
    if percent:_percent(ax)

def render(tables: dict[str,pd.DataFrame]) -> dict[str,bytes]:
    """All twelve figures use fixed strategy order and no fitted display parameters."""
    style={'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,
           'axes.grid':True,'grid.alpha':.2,'axes.axisbelow':True,'savefig.facecolor':'white'}
    with plt.rc_context(style):
        return _render(tables)

def _render(t: dict[str,pd.DataFrame]) -> dict[str,bytes]:
    out={};p=t['performance_metrics'];primary=p[p.fee_rate==PRIMARY].set_index('strategy').loc[list(STRATEGIES)]
    daily=t['daily_return_drawdown'];daily=daily[daily.fee_rate==PRIMARY]
    for column,title,file in [('cumulative_nav','Cumulative NAV | fee 0.03%, slippage 0.02% per side','01_cumulative_nav'),
                              ('drawdown','Drawdown below running peak | main cost','02_drawdown')]:
        fig,ax=plt.subplots(figsize=(10,5))
        for s in STRATEGIES:
            rows=daily[daily.strategy==s];ax.plot(pd.to_datetime(rows.date),rows[column],label=s,color=COLORS[s],lw=1.5)
        ax.set_title(title);ax.legend();ax.set_ylabel('Initial NAV = 1' if column=='cumulative_nav' else 'Drawdown')
        ax.set_xlim(pd.to_datetime(daily.date.min()),pd.to_datetime(daily.date.max()))
        if column=='drawdown':_percent(ax)
        _save(fig,file,out)
    annual=t['annual_metrics'];annual=annual[annual.fee_rate==PRIMARY]
    fig,ax=plt.subplots(figsize=(10,5));_grouped(annual,'total_return',ax,'Annual returns | 2022 starts Jan 4; main cost')
    _save(fig,'03_annual_returns',out)
    for metric,title,file,is_pct in [('sharpe_ratio','Sharpe | 252 sessions, risk-free = 0','04_sharpe_by_fee',False),
                                   ('maximum_drawdown','Maximum drawdown | positive loss magnitude','05_max_drawdown_by_fee',True)]:
        fig,ax=plt.subplots(figsize=(9,5));x=np.arange(3)
        for i,fee in enumerate(FEES):
            values=p[p.fee_rate==fee].set_index('strategy').loc[list(STRATEGIES),metric]
            ax.bar(x+(i-1.5)*.19,values,.18,label=f'fee {float(fee):.2%}',color=plt.cm.Blues(.35+i*.18))
        ax.set_xticks(x,STRATEGIES);ax.set_title(title);ax.legend()
        if is_pct:_percent(ax)
        _save(fig,file,out)
    fig,axes=plt.subplots(2,3,figsize=(13,8))
    panels=[('terminal_nav','Terminal NAV (CNY)',False),('total_return','Total return',True),('cagr','CAGR (252 sessions)',True),
            ('sharpe_ratio','Sharpe',False),('maximum_drawdown','Max drawdown (loss)',True),('total_direct_trading_friction','Direct friction (CNY)',False)]
    for ax,(key,title,is_pct) in zip(axes.flat,panels):
        for s in STRATEGIES:
            rows=p[p.strategy==s].set_index('fee_rate').loc[list(FEES)]
            ax.plot([float(f) for f in FEES],rows[key],'-o',color=COLORS[s],label=s)
        ax.set_title(title);ax.set_xticks([float(f) for f in FEES],[f'{float(f):.2%}' for f in FEES]);ax.set_xlabel('Single-side fee')
        if is_pct:_percent(ax)
    axes[0,0].legend();fig.suptitle('Four frozen cost scenarios | slippage always 0.02% per side')
    _save(fig,'06_cost_sensitivity',out)
    fig,axes=plt.subplots(1,2,figsize=(11,5))
    axes[0].bar(STRATEGIES,primary.cumulative_turnover,color=[COLORS[s] for s in STRATEGIES]);axes[0].set_title('Cumulative turnover (times NAV)')
    axes[1].bar(STRATEGIES,primary.explicit_fees,label='Explicit fees',color='#365b89')
    axes[1].bar(STRATEGIES,primary.slippage_cost,bottom=primary.explicit_fees,label='Slippage',color='#df9551')
    axes[1].set_title('Direct friction (CNY) | main cost');axes[1].legend();_save(fig,'07_turnover_friction',out)
    fig,axes=plt.subplots(1,2,figsize=(10,5))
    for ax,key,title in zip(axes,['average_gross_exposure','average_cash_ratio'],['Average invested exposure','Average cash including restricted proceeds']):
        ax.bar(STRATEGIES,primary[key],color=[COLORS[s] for s in STRATEGIES]);ax.set_ylim(0,1.05);ax.set_title(title);_percent(ax)
    _save(fig,'08_exposure_cash',out)
    cov=t['coverage_effect_summary'];fig,axes=plt.subplots(1,2,figsize=(11,6))
    for ax,s in zip(axes,['A','B']):
        rows=cov[(cov.strategy==s)&(cov.grouping=='year_asset')]
        matrix=rows.pivot(index='asset_code',columns='year',values='prediction_coverage').sort_index()
        im=ax.imshow(matrix.to_numpy(),vmin=0,vmax=1,cmap='YlGnBu',aspect='auto')
        ax.set_xticks(range(4),matrix.columns);ax.set_yticks(range(len(matrix)),matrix.index);ax.grid(False);ax.set_title(s+' | frozen prediction coverage')
        for i in range(len(matrix)):
            for j in range(4):
                v=matrix.iloc[i,j];ax.text(j,i,f'{v:.1%}',ha='center',va='center',color='white' if v>.7 else 'black',fontsize=9)
    fig.colorbar(im,ax=axes.tolist(),fraction=.025,pad=.02).set_label('C does not require model predictions')
    # Explicit axes positioning avoids tight_layout warnings from a shared colorbar.
    fig.subplots_adjust(left=.08,right=.87,wspace=.32,bottom=.12,top=.9)
    fig.text(.5,.02,FOOTER,ha='center',fontsize=8)
    buffer=BytesIO();fig.savefig(buffer,format='png',dpi=130,metadata={'Software':'quant-lab Phase8'})
    out['charts/09_prediction_coverage.png']=buffer.getvalue();plt.close(fig)
    y=t['yearly_strategy_comparison'];y=y[y.fee_rate==PRIMARY];fig,ax=plt.subplots(figsize=(10,5));x=np.arange(4)
    for i,s in enumerate(['A','B']):
        values=y[y.strategy==s].sort_values('year');ax.bar(x+(i-.5)*.3,values.total_return_difference,.29,label=s+' - C',color=COLORS[s])
    ax.set_xticks(x,['2022','2023','2024','2025']);ax.axhline(0,color='#444');_percent(ax);ax.legend();ax.set_title('Annual return differences | main cost (percentage points)')
    _save(fig,'10_yearly_return_difference',out)
    fig,axes=plt.subplots(2,2,figsize=(12,8));ac=t['annual_nav_curves'];ac=ac[ac.fee_rate==PRIMARY]
    for ax,year in zip(axes.flat,['2022','2023','2024','2025']):
        for s in STRATEGIES:
            rows=ac[(ac.year==year)&(ac.strategy==s)];ax.plot(pd.to_datetime(rows.date),rows.cumulative_nav,label=s,color=COLORS[s])
        ax.set_title(year+' | year-start balance = 1');ax.tick_params(axis='x',rotation=25)
        ax.set_xlim(pd.to_datetime(rows.date.min()),pd.to_datetime(rows.date.max()))
    axes[0,0].legend();_save(fig,'11_annual_cumulative_nav',out)
    fig,axes=plt.subplots(1,2,figsize=(11,5));cost=t['benchmark_comparison']
    for ax,key,title in zip(axes,['total_return_difference','cagr_difference'],['Total return difference vs C','CAGR difference vs C']):
        for s in ['A','B']:
            rows=cost[cost.strategy==s].set_index('fee_rate').loc[list(FEES)]
            ax.plot([float(f) for f in FEES],rows[key],'-o',label=s+' - C',color=COLORS[s])
        ax.axhline(0,color='#444',lw=.7);ax.set_xticks([float(f) for f in FEES],[f'{float(f):.2%}' for f in FEES]);ax.set_title(title);_percent(ax);ax.legend()
    _save(fig,'12_cost_relative_advantage',out)
    return out

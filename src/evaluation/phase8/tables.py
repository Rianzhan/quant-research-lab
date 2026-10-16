"""Reconciled descriptive tables derived solely from frozen account observations."""
from collections import defaultdict,Counter
from decimal import Decimal,localcontext
import numpy as np
import pandas as pd
from .common import STRATEGIES,FEES,account_id,decimal,require
from .metrics import nav_metrics

EPS=Decimal('0.00000001')

def _equal(a: object,b: object,label: str) -> None:
    require(abs(decimal(a)-decimal(b))<=EPS,'Reconciliation failed: '+label)

def _dated(rows: list[dict]) -> dict[str,dict]:
    result={r['date']:r for r in rows}
    require(len(result)==len(rows),'Duplicate daily observation')
    return result

def reconcile_account(account: dict) -> list[dict]:
    """Check saved accounting identities; do not reconstruct execution decisions."""
    days=sorted(account['daily'],key=lambda r:r['date']);dates=[d['date'] for d in days]
    require(len(dates)==len(set(dates))==account['summary']['sessions'],'Daily coverage changed')
    costs=_dated(account['costs']);turnover=_dated(account['turnover'])
    require(set(dates)==set(costs)==set(turnover),'Cost/turnover calendars differ')
    holdings=defaultdict(list);seen=set();codes={h['asset_code'] for h in account['holdings']}
    for h in account['holdings']:
        key=(h['date'],h['asset_code']);require(key not in seen and h['date'] in dates,'Invalid holdings key')
        seen.add(key);holdings[h['date']].append(h)
    fills=defaultdict(list);orders={o['order_id']:o for o in account['orders']}
    require(len(orders)==len(account['orders']),'Duplicate order id')
    for f in account['fills']:
        require(f['date'] in dates and f['order_id'] in orders,'Fill has no matching date/order')
        require(orders[f['order_id']]['execution_date']==f['date']>f['decision_date'],'Fill timing changed')
        fills[f['date']].append(f)
    with localcontext() as ctx:
        ctx.prec=40;previous=decimal(account['summary']['initial_nav']);fees=slip=Decimal(0)
        for d in days:
            date=d['date'];nav=decimal(d['nav']);cash=decimal(d['cash']);mv=decimal(d['gross_asset_value'])
            require(nav>0 and cash>=0 and 0<=mv<=nav,'Invalid NAV/cash/exposure')
            _equal(d['previous_nav'],previous,'previous NAV');_equal(nav,cash+decimal(d['dividend_receivable'])+mv,'NAV identity')
            require({h['asset_code'] for h in holdings[date]}==codes,'Incomplete holdings panel')
            for h in holdings[date]:
                require(decimal(h['shares'])>=0 and decimal(h['market_value'])>=0,'Negative holding')
                _equal(h['market_value'],decimal(h['shares'])*decimal(h['valuation_price'] or '0'),'Holding value')
            _equal(mv,sum((decimal(h['market_value']) for h in holdings[date]),Decimal(0)),'Holdings aggregate')
            _equal(d['held_asset_count'],sum(decimal(h['shares'])>0 for h in holdings[date]),'Holding count')
            for field,fill_field in [('fees_paid','fee'),('slippage_cost','slippage_cost'),('executed_notional','executed_notional')]:
                require(decimal(d[field])>=0,'Negative trading cost/turnover')
                _equal(d[field],sum((decimal(f[fill_field]) for f in fills[date]),Decimal(0)),field+' fills')
            for field in ['fees_paid','slippage_cost','cumulative_explicit_fees','cumulative_slippage_cost','net_account_value','same_fill_gross_account_value']:
                _equal(d[field],costs[date][field],'cost table '+field)
            for field in ['pre_trade_nav','executed_notional','turnover']:_equal(d[field],turnover[date][field],'turnover table '+field)
            _equal(d['turnover'],decimal(d['executed_notional'])/decimal(d['pre_trade_nav']),'turnover formula')
            fees+=decimal(d['fees_paid']);slip+=decimal(d['slippage_cost'])
            _equal(d['cumulative_explicit_fees'],fees,'cumulative fee');_equal(d['cumulative_slippage_cost'],slip,'cumulative slip')
            _equal(d['same_fill_gross_account_value'],nav+fees+slip,'same-fill gross bridge')
            expected=decimal(d['market_revaluation'])+decimal(d['dividend_accrual'])+decimal(d['share_rounding_value'])-decimal(d['fees_paid'])-decimal(d['slippage_cost'])
            _equal(nav-previous,expected,'daily economic bridge');_equal(d['unexplained_pnl'],0,'unexplained PnL')
            require(d['daily_reconciliation']=='passed','Frozen reconciliation did not pass');previous=nav
        for field,value in [('terminal_nav',previous),('explicit_fee_total',fees),('slippage_cost_total',slip)]:
            _equal(account['summary'][field],value,'summary '+field)
    return days

def period_stats(days: list[dict],fills: list[dict]) -> dict:
    """Costs are actual monetary amounts; exposure includes idle sessions."""
    n=len(days);dates={d['date'] for d in days}
    with localcontext() as ctx:
        ctx.prec=40
        total=lambda key:sum((decimal(d[key]) for d in days),Decimal(0))
        fees=total('fees_paid');slip=total('slippage_cost')
        exposure=[decimal(d['gross_asset_value'])/decimal(d['nav']) for d in days]
        cash=[decimal(d['cash'])/decimal(d['nav']) for d in days]
        return {'average_gross_exposure':float(sum(exposure)/n),'average_cash_ratio':float(sum(cash)/n),
            'max_gross_exposure':float(max(exposure)),'cumulative_turnover':float(total('turnover')),
            'explicit_fees':float(fees),'slippage_cost':float(slip),'total_direct_trading_friction':float(fees+slip),
            'number_of_fills':sum(f['date'] in dates for f in fills),'sessions_invested':sum(v>0 for v in exposure)}

def weight_drift(strategy: str,days: list[dict],holdings: list[dict],coverage: list[dict]) -> pd.DataFrame:
    """Compare actual close weights with targets effective no earlier than the next session."""
    dates=sorted(d['date'] for d in days);nexts=dict(zip(dates,dates[1:]));updates=defaultdict(dict)
    for r in coverage:
        if r['strategy']==strategy and r['rebalance_date'] in nexts:
            updates[nexts[r['rebalance_date']]][r['asset_code']]=decimal(r['target_weight'])
    by_date={d['date']:d for d in days};targets={};rows=[]
    for h in sorted(holdings,key=lambda r:(r['date'],r['asset_code'])):
        date=h['date'];targets.update(updates.get(date,{}))
        actual=decimal(h['market_value'])/decimal(by_date[date]['nav']);target=targets.get(h['asset_code'],Decimal(0))
        rows.append({'date':date,'asset_code':h['asset_code'],'actual_weight':float(actual),
            'reference_target_weight':float(target),'weight_drift':float(actual-target),
            'above_target_cap':float(max(Decimal(0),actual-Decimal('.25')))})
    return pd.DataFrame(rows)

def coverage_summary(rows: list[dict]) -> pd.DataFrame:
    """Separate overlapping availability flags from mutually exclusive signal reasons."""
    groups=defaultdict(list);seen=set()
    for r in rows:
        key=(r['strategy'],r['rebalance_date'],r['asset_code']);require(key not in seen,'Duplicate coverage row');seen.add(key)
        for grouping,year,asset in [('overall','ALL','ALL'),('year',r['year'],'ALL'),('asset','ALL',r['asset_code']),('year_asset',r['year'],r['asset_code'])]:
            groups[(r['strategy'],grouping,year,asset)].append(r)
    result=[]
    for (strategy,grouping,year,asset),group in sorted(groups.items()):
        model=strategy!='C';n=len(group);present=sum(r['prediction_available']=='True' for r in group)
        missing=[r for r in group if r['prediction_available']=='False']
        tail=[r for r in missing if r['label_horizon_outside_development']=='True']
        reasons=Counter(r['missing_reason'] for r in group)
        result.append({'strategy':strategy,'grouping':grouping,'year':year,'asset_code':asset,'observations':n,
            'predictions_available':present if model else None,'missing_predictions':len(missing) if model else None,
            'prediction_coverage':present/n if model else None,
            'missing_prediction_zero_targets':sum(r['signal_reason']=='missing_prediction' and decimal(r['target_weight'])==0 for r in group),
            'threshold_zero_targets':sum(r['signal_reason']=='below_or_equal_hurdle' and decimal(r['target_weight'])==0 for r in group),
            'feature_incomplete_primary':reasons['feature_incomplete'],'insufficient_history_primary':reasons['fold_asset_insufficient'],
            'invalid_event_gap_label_primary':reasons['invalid_label_event_or_gap'],'end_label_primary':reasons['label_horizon_outside_development'],
            'not_tradable_primary':reasons['not_tradable'],'end_label_missing':len(tail),
            'end_label_and_incomplete':sum(r['feature_available']=='False' for r in tail),
            'eligible_targets':sum(r['strategy_eligible']=='True' for r in group)})
    return pd.DataFrame(result)

DIFFERENCES=['total_return','cagr','sharpe_ratio','maximum_drawdown','annualized_volatility','cumulative_turnover',
             'explicit_fees','slippage_cost','total_direct_trading_friction','average_gross_exposure']

def comparisons(performance: pd.DataFrame,annual: pd.DataFrame) -> tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    """Same-cost, same-date comparisons and telescoping annual wealth gaps, no ranking."""
    p=performance.set_index(['strategy','fee_rate']);a=annual.set_index(['strategy','fee_rate','year'])
    whole=[];yearly=[];sensitivity=[]
    for fee in FEES:
        c=p.loc['C',fee]
        for strategy in STRATEGIES:
            row=p.loc[strategy,fee]
            require(row.start_date==c.start_date and row.end_date==c.end_date and row.sessions==c.sessions,'Unpaired comparison window')
            diff={k+'_difference':row[k]-c[k] for k in DIFFERENCES}
            sensitivity.append({**row.to_dict(),'strategy':strategy,'fee_rate':fee,**diff})
            if strategy=='C':continue
            whole.append({'strategy':strategy,'benchmark':'C','fee_rate':fee,'slippage':'.0002',**diff})
            final_gap=decimal(a.loc[strategy,fee,'2025'].ending_nav_exact)-decimal(a.loc['C',fee,'2025'].ending_nav_exact);changes=[]
            for year in ['2022','2023','2024','2025']:
                x=a.loc[strategy,fee,year];b=a.loc['C',fee,year]
                start=decimal(x.starting_nav_exact)-decimal(b.starting_nav_exact)
                end=decimal(x.ending_nav_exact)-decimal(b.ending_nav_exact);change=end-start;changes.append(change)
                yearly.append({'strategy':strategy,'benchmark':'C','fee_rate':fee,'year':year,
                    **{k+'_difference':x[k]-b[k] for k in DIFFERENCES},
                    'wealth_gap_start':float(start),'wealth_gap_end':float(end),'wealth_gap_change':float(change),
                    'fraction_of_terminal_gap':float(change/final_gap) if abs(final_gap)>EPS else None})
            _equal(sum(changes),final_gap,'annual wealth-gap telescope')
    return pd.DataFrame(whole),pd.DataFrame(yearly),pd.DataFrame(sensitivity)

def evaluate(bundle: dict) -> dict[str,pd.DataFrame]:
    """No strategy runner, model, prediction or raw-market interfaces exist here."""
    expected={account_id(s,f) for s in STRATEGIES for f in FEES}
    require(set(bundle['accounts'])==expected,'Exactly twelve accounts required')
    performance=[];annual=[];curves=[];annual_curves=[];episodes=[];exposures=[];distributions=[];drifts=[];attribution=[];calendar=None
    for strategy in STRATEGIES:
        for fee in FEES:
            identity=account_id(strategy,fee);account=bundle['accounts'][identity];days=reconcile_account(account)
            dates=[d['date'] for d in days];initial=account['summary']['initial_nav']
            require(calendar is None or calendar==dates,'Account calendars differ');calendar=dates
            tags={'account_id':identity,'strategy':strategy,'fee_rate':fee,'slippage':'0.0002'}
            m,curve,ep=nav_metrics(dates,[d['nav'] for d in days],initial)
            stats=period_stats(days,account['fills']);performance.append({**tags,**m,**stats})
            curves.append(curve.assign(**tags));episodes.append(ep.assign(**tags))
            drift=weight_drift(strategy,days,account['holdings'],bundle['coverage']);drifts.append(drift.assign(**tags))
            counts=Counter(int(d['held_asset_count']) for d in days)
            exposures.append({**tags,**stats,'max_actual_asset_weight':float(drift.actual_weight.max()),
                'max_weight_above_25pct':float(drift.above_target_cap.max()),
                'asset_sessions_above_cap':int((drift.above_target_cap>1e-12).sum()),
                'max_absolute_weight_drift':float(drift.weight_drift.abs().max()),
                'mean_absolute_weight_drift':float(drift.weight_drift.abs().mean()),
                'average_holding_count':sum(k*v for k,v in counts.items())/len(days)})
            distributions.extend({**tags,'held_assets':k,'sessions':v,'fraction':v/len(days)} for k,v in sorted(counts.items()))
            friction=stats['total_direct_trading_friction'];gross=m['terminal_nav']+friction
            attribution.append({**tags,'explicit_fees':stats['explicit_fees'],'slippage_cost':stats['slippage_cost'],
                'cumulative_turnover':stats['cumulative_turnover'],'direct_friction':friction,
                'friction_over_initial_capital':friction/float(initial),'same_fill_gross_terminal_value':gross,
                'friction_over_same_fill_gross_value':friction/gross,'not_counterfactual_gross_strategy':True})
            by_year=defaultdict(list)
            for d in days:by_year[d['date'][:4]].append(d)
            require(set(by_year)=={'2022','2023','2024','2025'},'All four evaluation years required')
            previous_date=dates[0];compound=1.0
            for year,group in sorted(by_year.items()):
                start=group[0]['previous_nav'];y,ycurve,_=nav_metrics([d['date'] for d in group],[d['nav'] for d in group],start,previous_date)
                annual.append({**tags,'year':year,**y,**period_stats(group,account['fills']),
                    'calendar_year_return':y['total_return'],'starting_nav_exact':start,'ending_nav_exact':group[-1]['nav'],
                    'interval_status':'available_strategy_interval' if year=='2022' else 'calendar_year'})
                annual_curves.append(ycurve.assign(**tags,year=year));previous_date=group[-1]['date'];compound*=1+y['total_return']
            require(abs(compound-1-m['total_return'])<1e-10,'Annual returns do not compound to total return')
    p=pd.DataFrame(performance);a=pd.DataFrame(annual);benchmark,yearly,cost=comparisons(p,a)
    draw_columns=['account_id','strategy','fee_rate','maximum_drawdown','drawdown_peak_date','drawdown_trough_date',
        'recovery_date','recovery_status','drawdown_duration_sessions','drawdown_duration_calendar_days',
        'longest_drawdown_duration_sessions','longest_drawdown_duration_calendar_days','longest_drawdown_peak_date',
        'longest_drawdown_end_date','longest_drawdown_recovery_status']
    return {'performance_metrics':p,'annual_metrics':a,'benchmark_comparison':benchmark,'cost_sensitivity':cost,
        'drawdown_analysis':p[draw_columns].copy(),'exposure_analysis':pd.DataFrame(exposures),
        'coverage_effect_summary':coverage_summary(bundle['coverage']),'yearly_strategy_comparison':yearly,
        'daily_return_drawdown':pd.concat(curves,ignore_index=True),'annual_nav_curves':pd.concat(annual_curves,ignore_index=True),
        'drawdown_episodes':pd.concat(episodes,ignore_index=True),'holding_count_distribution':pd.DataFrame(distributions),
        'asset_weight_drift':pd.concat(drifts,ignore_index=True),'cost_attribution':pd.DataFrame(attribution),
        'coverage_cashification':pd.DataFrame(sorted([r for r in bundle['coverage'] if r['prediction_available']=='False'
            and r['label_horizon_outside_development']=='True'],key=lambda r:(r['strategy'],r['rebalance_date'],r['asset_code'])))}
